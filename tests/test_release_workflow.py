"""Boundary tests for installed entry points and reproducible run orchestration."""

import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from flowatom.cli import main as cli_main
from flowatom.cli.run_experiment import main as run_main, validate_background
from flowatom.cli.summarize_results import main as summarize_main
from flowatom.config import DEFAULT_CONFIG_PATH, load_config

ROOT = Path(__file__).resolve().parents[1]


class ReleaseTests(unittest.TestCase):
    def test_packaged_configs_match_source_examples(self):
        for name in ("mainline.yaml", "smoke.yaml"):
            self.assertEqual(
                (ROOT / "configs" / name).read_bytes(),
                (DEFAULT_CONFIG_PATH.parent / name).read_bytes(),
            )
        with tempfile.TemporaryDirectory() as directory:
            command = "from flowatom.config import load_config; assert load_config()['encoder']['input_length'] == 300"
            env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
            subprocess.run([sys.executable, "-c", command], cwd=directory, env=env, check=True)

    def test_help_does_not_import_torch(self):
        code = """import sys
from flowatom.cli import main
try:
    main(['--help'])
except SystemExit as error:
    assert error.code == 0
assert 'torch' not in sys.modules
"""
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("build-atom-vocabulary", result.stdout)

    def test_dispatch_passes_arguments_without_reparsing(self):
        with patch("flowatom.cli.importlib.import_module") as module:
            module.return_value.main.return_value = 7
            self.assertEqual(cli_main(["run", "--seed", "2026"]), 7)
            module.return_value.main.assert_called_once_with(["--seed", "2026"])

    def _inputs(self, root):
        (root / "traces.parquet").touch()
        (root / "encoder.pt").touch()
        return [
            "--traces",
            str(root / "traces.parquet"),
            "--encoder",
            str(root / "encoder.pt"),
            "--config",
            str(ROOT / "configs/smoke.yaml"),
            "--output-dir",
            str(root / "run"),
        ]

    def test_dry_run_propagates_seed_and_creates_nothing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(run_main(self._inputs(root) + ["--seed", "17", "--dry-run"]), 0)
            plan = json.loads(output.getvalue())["stages"]
            for name in ("split", "vocabulary", "train"):
                self.assertEqual(plan[name][plan[name].index("--seed") + 1], "17")
            self.assertFalse((root / "run").exists())

    def test_existing_run_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self._inputs(root)
            (root / "run").mkdir()
            sentinel = root / "run/model.pt"
            sentinel.write_bytes(b"valuable checkpoint")
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                run_main(args)
            self.assertEqual(sentinel.read_bytes(), b"valuable checkpoint")

    def test_failure_records_stage_and_stops_dependencies(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self._inputs(root)
            with (
                patch("flowatom.cli.run_experiment.subprocess.run") as child,
                patch(
                    "flowatom.cli.run_experiment.platform.platform", return_value="test-platform"
                ),
            ):
                child.return_value.returncode = 9
                with (
                    contextlib.redirect_stdout(io.StringIO()),
                    contextlib.redirect_stderr(io.StringIO()),
                ):
                    self.assertEqual(run_main(args), 1)
                self.assertEqual(child.call_count, 1)
            state = json.loads((root / "run/manifest.json").read_text())
            self.assertEqual(state["status"], "failed")
            self.assertEqual(state["stages"][0]["returncode"], 9)
            self.assertTrue((root / "run/logs/split.log").exists())
            self.assertEqual(load_config(root / "run/config.yaml")["predictor"]["seeds"], [2025])

    def test_background_rejects_monitored_sites_and_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            monitored = root / "monitored.parquet"
            background = root / "background.parquet"
            pd.DataFrame({"trace_id": [1], "label": [0], "site": ["a"]}).to_parquet(monitored)
            for row in (
                {"trace_id": 2, "label": -1, "site": "a"},
                {"trace_id": 1, "label": -1, "site": "b"},
                {"trace_id": 2, "label": 0, "site": "b"},
            ):
                pd.DataFrame([row]).to_parquet(background)
                with self.assertRaises(ValueError):
                    validate_background(monitored, background)
            pd.DataFrame({"trace_id": [2], "label": [-1], "site": ["b"]}).to_parquet(background)
            validate_background(monitored, background)

    def test_summary_single_run_and_duplicate_input(self):
        with tempfile.TemporaryDirectory() as directory:
            metrics = Path(directory) / "metrics.json"
            metrics.write_text(json.dumps({"seed": 2025, "test": {"overall": {"micro_f1": 0.9}}}))
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                summarize_main(["--run", str(metrics)])
            self.assertIsNone(json.loads(output.getvalue())["std_micro_f1"])
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                summarize_main(["--run", str(metrics), "--run", str(metrics)])


if __name__ == "__main__":
    unittest.main()
