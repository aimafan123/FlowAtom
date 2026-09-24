"""Regression tests for split isolation, short flows and artifact compatibility.

Run with: PYTHONPATH=src python -m unittest discover -s tests -v
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace

import numpy as np
import pandas as pd
import torch

from flowatom.artifacts import (
    ArtifactError, file_sha256, fingerprint, load_embedding_cache,
    write_embedding_metadata,
)
from flowatom.atoms import (
    AtomVocabulary, ExtractionConfig, NearestCentroidClassifier,
    TraceAtomTable, concatenate_trace_atom_tables, extract_trace_atoms,
)
from flowatom.data import (
    build_closed_world_specs, iter_flow_sequences, validate_specs,
    validate_trace_frame, validate_trace_pools,
)
from flowatom.models import DFMiniEncoder, load_pretrained_encoder
from flowatom.models.checkpoints import CheckpointError
from flowatom.representation import flow_representation
from flowatom.training import WindowPredictorConfig, train_window_predictor, load_frozen_predictor, evaluate_frozen
from flowatom.training.window_predictor import NonSingletonBatchSampler, validate_training_inputs
from torch.utils.data import SequentialSampler

ROOT = Path(__file__).resolve().parents[1]


def trace_frame(train_count=3):
    rows = []
    for label in range(5):
        for split, count in (("train", train_count), ("test", 2)):
            for visit in range(count):
                rows.append({
                    "trace_id": len(rows) + 1, "label": label, "site": f"site-{label}",
                    "split": split, "test_order": visit,
                    "payload_flows": [[100], [20 + label * 100 + visit * 15] * 12],
                    "direction_flows": [[1], [1, -1] * 6],
                })
    return pd.DataFrame(rows)


def split_specs(frame):
    return build_closed_world_specs(frame, train_samples=15, validation_samples=5,
                                    test_counts={size: 1 for size in range(1, 6)})


def feature_contract(**overrides):
    result = {
        "encoder_sha256": "encoder-A", "vocabulary_sha256": "vocabulary-A",
        "representation": "signed_payload_length", "input_length": 32,
        "minimum_nonzero_payload_packets": 3, "confidence_threshold": 0.0,
        "max_flows_per_trace": 0, "flow_sampling_seed": None,
    }
    result.update(overrides)
    return result


def atom_table(frame, specs):
    return TraceAtomTable(
        trace_ids=frame.trace_id.to_numpy(), labels=frame.label.to_numpy(),
        peak=np.random.default_rng(1).random((len(frame), 3)).astype(np.float32),
        flow_count=np.ones(len(frame), dtype=np.int64), retained_flow_count=len(frame),
        total_flow_count=len(frame), input_length=32, contract=feature_contract(),
        atom_fit={"source": "trace_train_pool", "training_trace_ids": specs["trace_pools"]["train"],
                  "trace_pools_sha256": fingerprint(specs["trace_pools"])},
    )


class SplitTests(unittest.TestCase):
    def setUp(self):
        self.frame = trace_frame()
        self.specs = split_specs(self.frame)

    def test_pools_are_disjoint_complete_and_stable_under_row_reordering(self):
        pools = validate_trace_pools(self.specs)
        self.assertFalse(pools["train"] & pools["validation"])
        self.assertEqual(set.union(*pools.values()), set(self.frame.trace_id))
        self.assertEqual(self.specs, split_specs(self.frame.sample(frac=1, random_state=9)))
        validate_specs(self.specs, self.frame)

    def test_single_training_trace_fails_before_sampling(self):
        with self.assertRaisesRegex(ValueError, "at least two"):
            split_specs(trace_frame(train_count=1))

    def test_training_rejects_window_leakage_even_when_flag_claims_disjoint(self):
        table = atom_table(self.frame, self.specs)
        self.specs["validation"][0] = copy.deepcopy(self.specs["train"][0])
        with self.assertRaisesRegex(ValueError, "outside its pool"):
            validate_training_inputs(self.specs, table)

    def test_training_rejects_vocabulary_fitted_on_validation(self):
        table = atom_table(self.frame, self.specs)
        fit = copy.deepcopy(table.atom_fit)
        fit["training_trace_ids"] += self.specs["trace_pools"]["validation"]
        with self.assertRaisesRegex(ArtifactError, "Atom training trace pool"):
            validate_training_inputs(self.specs, replace(table, atom_fit=fit))

    def test_training_rejects_different_split_or_dataset(self):
        table = atom_table(self.frame, self.specs)
        fit = {**table.atom_fit, "trace_pools_sha256": "different"}
        with self.assertRaisesRegex(ArtifactError, "Atom training split"):
            validate_training_inputs(self.specs, replace(table, atom_fit=fit))
        self.specs["trace_dataset_sha256"] = "source-A"
        with self.assertRaisesRegex(ArtifactError, "training cache source"):
            validate_training_inputs(self.specs, replace(table, source_sha256="source-B"))

    def test_background_sites_can_share_unmonitored_label(self):
        frame = self.frame.copy()
        frame["label"] = -1
        validate_trace_frame(frame)


class SmallEncoder(torch.nn.Module):
    checkpoint_sha256 = "encoder-A"

    def encode(self, inputs):
        return inputs[:, :2]


class ShortFlowTests(unittest.TestCase):
    def setUp(self):
        self.vocabulary = AtomVocabulary(
            classifier=NearestCentroidClassifier(np.array([[10, 20], [20, 30]], dtype=np.float32)),
            classifier_type="nearest_centroid",
            provenance={**{key: feature_contract()[key] for key in (
                "encoder_sha256", "representation", "input_length", "minimum_nonzero_payload_packets")},
                "atom_fit": {"source": "independent_unlabeled_pretraining"}},
            vocabulary_sha256="vocabulary-A",
        )
        self.frame = pd.DataFrame([
            {"trace_id": 1, "label": 0, "payload_flows": [[1], [10, 20, 30], [20, 30, 40]],
             "direction_flows": [[1], [1, 1, 1], [1, 1, 1]]},
            {"trace_id": 2, "label": 1, "payload_flows": [[1, 0]], "direction_flows": [[1, 1]]},
        ])
        self.config = ExtractionConfig(input_length=32, minimum_nonzero_payload_packets=3, batch_size=1)

    def test_exact_threshold_and_zero_payload_removal(self):
        self.assertIsNone(flow_representation([0, 10, 20], [1, 1, -1], min_payload_packets=3))
        np.testing.assert_array_equal(
            flow_representation([0, 10, 20, 30], [1, 1, -1, 1], input_length=4, min_payload_packets=3),
            [10, -20, 30, 0],
        )

    def test_short_flows_skip_and_all_short_trace_survives(self):
        table = extract_trace_atoms(self.frame, SmallEncoder(), self.vocabulary, self.config)
        self.assertEqual(table.total_flow_count, 4)
        self.assertEqual(table.short_flow_count, 2)
        self.assertEqual(table.retained_flow_count, 2)
        np.testing.assert_array_equal(table.flow_count, [2, 0])
        np.testing.assert_array_equal(table.peak, [[1, 1], [0, 0]])
        self.assertEqual(len(list(iter_flow_sequences(self.frame, input_length=32, min_payload_packets=3))), 2)

    def test_budget_is_applied_after_short_flow_filtering(self):
        table = extract_trace_atoms(self.frame, SmallEncoder(), self.vocabulary,
                                    replace(self.config, max_flows_per_trace=1))
        np.testing.assert_array_equal(table.flow_count, [1, 0])
        self.assertEqual(table.short_flow_count, 2)
        self.assertEqual(table.peak[0].sum(), 1)

    def test_confidence_filter_updates_actual_contributing_counts(self):
        class SoftClassifier:
            classes_ = np.arange(2)
            n_features_in_ = 2
            def predict_proba(self, inputs):
                return np.tile([0.6, 0.4], (len(inputs), 1))
        vocab = replace(self.vocabulary, classifier=SoftClassifier(), classifier_type="xgboost")
        table = extract_trace_atoms(self.frame, SmallEncoder(), vocab,
                                    replace(self.config, confidence_threshold=0.9))
        self.assertEqual(table.retained_flow_count, 0)
        self.assertFalse(table.peak.any())
        self.assertFalse(table.flow_count.any())

    def test_malformed_alignment_is_not_silently_filtered(self):
        self.frame.at[0, "direction_flows"] = [[1], [1], [1, 1, 1]]
        with self.assertRaisesRegex(ValueError, "align"):
            extract_trace_atoms(self.frame, SmallEncoder(), self.vocabulary, self.config)

    def test_extraction_rejects_wrong_encoder_or_preprocessing(self):
        for key, value in (("encoder_sha256", "encoder-B"), ("input_length", 300),
                           ("representation", "direction_only"), ("minimum_nonzero_payload_packets", 10)):
            with self.subTest(key=key):
                vocab = replace(self.vocabulary, provenance={**self.vocabulary.provenance, key: value})
                with self.assertRaises(ArtifactError):
                    extract_trace_atoms(self.frame, SmallEncoder(), vocab, self.config)


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.frame = trace_frame()
        self.specs = split_specs(self.frame)
        self.table = atom_table(self.frame, self.specs)

    def test_cache_roundtrip_and_compatible_merge(self):
        path = self.root / "atoms.npz"
        self.table.save(path)
        loaded = TraceAtomTable.load(path)
        np.testing.assert_array_equal(loaded.peak, self.table.peak)
        self.assertEqual(loaded.contract, self.table.contract)
        combined = concatenate_trace_atom_tables([loaded, replace(loaded, trace_ids=loaded.trace_ids + 1000)])
        self.assertEqual(combined.trace_count, 2 * loaded.trace_count)

    def test_equal_dimensions_do_not_make_different_artifacts_compatible(self):
        for key, value in (("vocabulary_sha256", "vocabulary-B"), ("encoder_sha256", "encoder-B"),
                           ("minimum_nonzero_payload_packets", 4), ("max_flows_per_trace", 1),
                           ("flow_sampling_seed", 99)):
            with self.subTest(key=key):
                other = replace(self.table, trace_ids=self.table.trace_ids + 1000,
                                contract={**self.table.contract, key: value})
                with self.assertRaisesRegex(ArtifactError, "feature contract"):
                    concatenate_trace_atom_tables([self.table, other])
        for field, value in (("input_length", 300), ("representation", "direction_only"),
                             ("confidence_threshold", 0.5)):
            other = replace(self.table, trace_ids=self.table.trace_ids + 1000,
                            contract={**self.table.contract, field: value}, **{field: value})
            with self.assertRaisesRegex(ArtifactError, "feature contract"):
                concatenate_trace_atom_tables([self.table, other])

    def test_legacy_cache_fails_with_rebuild_message(self):
        path = self.root / "legacy.npz"
        np.savez(path, trace_ids=[1], labels=[0], peak=[[0.5, 0.5]], flow_count=[1])
        with self.assertRaisesRegex(ValueError, "rebuild"):
            TraceAtomTable.load(path)

    def test_embedding_cache_binds_content_shape_and_source(self):
        path = self.root / "embeddings.npy"
        np.save(path, np.ones((2, 3), dtype=np.float32))
        contract = {"encoder": "A", "source": "B", "split": "C", "minimum_packets": 3}
        write_embedding_metadata(path, contract, (2, 3))
        self.assertEqual(load_embedding_cache(path, contract, (2, 3)).shape, (2, 3))
        for key in contract:
            with self.subTest(key=key), self.assertRaises(ArtifactError):
                load_embedding_cache(path, {**contract, key: "changed"}, (2, 3))
        with self.assertRaises(ArtifactError):
            load_embedding_cache(path, contract, (3, 2))
        np.save(path, np.zeros((2, 3), dtype=np.float32))
        with self.assertRaisesRegex(ArtifactError, "content"):
            load_embedding_cache(path, contract, (2, 3))
        Path(str(path) + ".json").unlink()
        with self.assertRaisesRegex(ArtifactError, "rebuild"):
            load_embedding_cache(path, contract, (2, 3))

    def test_checkpoint_length_check_and_legacy_feature_dictionary(self):
        model = DFMiniEncoder(input_length=32)
        path = self.root / "encoder.pt"
        torch.save({"input_length": 32, "state_dict": model.state_dict()}, path)
        with self.assertRaisesRegex(CheckpointError, "input_length"):
            load_pretrained_encoder(path, 300)
        loaded = load_pretrained_encoder(path, 32)
        self.assertEqual(loaded.checkpoint_sha256, file_sha256(path))
        torch.save(model.state_dict(), path)
        self.assertEqual(load_pretrained_encoder(path, 32).input_length, 32)

    def test_vocabulary_roundtrip_and_replaced_model_rejected(self):
        provenance = ShortFlowTests()
        provenance.setUp()
        vocab = provenance.vocabulary
        vocab.save(self.root)
        loaded = AtomVocabulary.load(self.root)
        self.assertEqual(loaded.vocabulary_sha256, vocab.vocabulary_sha256)
        np.savez_compressed(self.root / "atom_centroids.npz", cluster_centers=np.array([[100, 200], [200, 300]]))
        with self.assertRaisesRegex(ArtifactError, "vocabulary files"):
            AtomVocabulary.load(self.root)

    def test_singleton_tail_is_merged_without_losing_samples(self):
        for count in (2, 3, 4, 5, 7, 13):
            with self.subTest(count=count):
                sampler = NonSingletonBatchSampler(SequentialSampler(range(count)), 2, False)
                batches = list(sampler)
                self.assertTrue(all(len(batch) >= 2 for batch in batches))
                self.assertEqual([i for batch in batches for i in batch], list(range(count)))
                self.assertEqual(len(sampler), len(batches))

    def test_training_roundtrip_frozen_contract_and_run_integrity(self):
        # 15 windows with batch_size=7 deliberately produces a singleton tail.
        result = train_window_predictor(self.specs, self.table, self.root,
            WindowPredictorConfig(epochs=1, batch_size=7, hidden_dim=8, device="cpu"))
        self.assertEqual(result["feature_contract"], self.table.contract)
        predictor = load_frozen_predictor(self.root)
        evaluated = evaluate_frozen(predictor, self.specs, self.table)
        self.assertEqual(evaluated["overall"]["samples"], 5)
        wrong = replace(self.table, contract={**self.table.contract, "vocabulary_sha256": "other"})
        with self.assertRaisesRegex(ArtifactError, "predictor/cache"):
            evaluate_frozen(predictor, self.specs, wrong)
        (self.root / "standardizer.npz").write_bytes(b"different run")
        with self.assertRaisesRegex(ArtifactError, "standardizer"):
            load_frozen_predictor(self.root)


class AtomBuildCLITests(unittest.TestCase):
    """Exercise the release entry points with raw short flows and real artifacts."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.tmp.cleanup)
        cls.root = Path(cls.tmp.name)
        cls.frame = trace_frame()
        cls.frame.to_parquet(cls.root / "traces.parquet", index=False)
        torch.manual_seed(1)
        encoder = DFMiniEncoder(input_length=32)
        torch.save({"input_length": 32, "state_dict": encoder.state_dict()}, cls.root / "encoder.pt")
        (cls.root / "config.yaml").write_text("""
encoder:
  input_length: 32
  minimum_nonzero_payload_packets: 3
atom_vocabulary:
  clusters: 3
  minimum_cluster_size: 1
  kmeans_n_init: 1
  kmeans_batch_size: 32
  xgboost:
    n_estimators: 2
    max_depth: 2
closed_world:
  train_samples: 15
  validation_samples: 5
  test_samples: {1: 1, 2: 1, 3: 1, 4: 1, 5: 1}
""")
        cls.run_script("build_mixture_specs.py", "--traces", cls.root / "traces.parquet",
                       "--output", cls.root / "specs.json", "--config", cls.root / "config.yaml")
        cls.arguments = ["--traces", cls.root / "traces.parquet", "--specs", cls.root / "specs.json",
                         "--encoder", cls.root / "encoder.pt", "--config", cls.root / "config.yaml",
                         "--output-dir", cls.root / "vocab", "--keep-embeddings", "--device", "cpu"]
        cls.run_script("build_atom_vocabulary.py", *cls.arguments)

    @classmethod
    def run_script(cls, script, *args, success=True):
        result = subprocess.run([sys.executable, str(ROOT / "scripts" / script), *map(str, args)],
                                text=True, capture_output=True, cwd=ROOT)
        if success and result.returncode:
            raise AssertionError(result.stdout + result.stderr)
        return result

    def test_atom_build_excludes_validation_and_test_and_skips_short_flows(self):
        specs = json.loads((self.root / "specs.json").read_text())
        vocab = AtomVocabulary.load(self.root / "vocab")
        training = set(vocab.provenance["atom_fit"]["training_trace_ids"])
        self.assertFalse(training & set(specs["trace_pools"]["validation"]))
        self.assertFalse(training & set(specs["trace_pools"]["test"]))
        self.assertEqual(np.load(self.root / "vocab/embeddings.npy").shape[0], len(training))
        self.run_script("build_atom_vocabulary.py", *self.arguments)

    def test_all_short_training_data_fails_before_creating_embeddings(self):
        frame = self.frame.copy(deep=True)
        frame["payload_flows"] = [[[1]] for _ in range(len(frame))]
        frame["direction_flows"] = [[[1]] for _ in range(len(frame))]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frame.to_parquet(root / "short.parquet", index=False)
            self.run_script("build_mixture_specs.py", "--traces", root / "short.parquet",
                            "--output", root / "specs.json", "--config", self.root / "config.yaml")
            result = self.run_script("build_atom_vocabulary.py", "--traces", root / "short.parquet",
                "--specs", root / "specs.json", "--encoder", self.root / "encoder.pt",
                "--config", self.root / "config.yaml", "--output-dir", root / "vocab", success=False)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("no eligible training flows", result.stderr)
            self.assertFalse((root / "vocab/embeddings.npy").exists())

    def test_specs_cannot_be_reused_for_different_trace_data(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "other.parquet"
            frame = self.frame.copy(deep=True)
            frame["test_order"] += 1
            frame.to_parquet(path, index=False)
            args = list(self.arguments)
            args[args.index("--traces") + 1] = path
            result = self.run_script("build_atom_vocabulary.py", *args, success=False)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("specs trace dataset mismatch", result.stderr)

    def test_stale_embedding_cache_is_rejected_by_cli(self):
        result = self.run_script("build_atom_vocabulary.py", *self.arguments, "--max-flows", 5, success=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("embedding source/configuration mismatch", result.stderr)

    def test_missing_split_specs_is_an_actionable_cli_error(self):
        args = list(self.arguments)
        index = args.index("--specs")
        del args[index:index+2]
        result = self.run_script("build_atom_vocabulary.py", *args, success=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--traces requires --specs", result.stderr)


if __name__ == "__main__":
    unittest.main()
