"""Tests for the trace schema and window specifications."""

import copy

import pytest

from flowatom.data import (
    DatasetValidationError,
    build_closed_world_specs,
    build_open_world_specs,
    validate_specs,
    validate_trace_frame,
)

from synthetic import make_traces, offset_trace_ids


def test_validate_trace_frame_summary(tiny_traces):
    summary = validate_trace_frame(tiny_traces)
    assert summary.traces == len(tiny_traces)
    assert summary.labels == 5
    assert summary.flows == sum(len(flows) for flows in tiny_traces["payload_flows"])
    assert summary.split_counts == {"train": 15, "test": 10}


def test_validate_trace_frame_detects_unaligned_flows(tiny_traces):
    frame = tiny_traces.copy()
    frame.at[0, "direction_flows"] = frame.at[0, "direction_flows"][:-1]
    with pytest.raises(DatasetValidationError):
        validate_trace_frame(frame)


def test_validate_trace_frame_detects_duplicate_ids(tiny_traces):
    frame = tiny_traces.copy()
    frame.at[1, "trace_id"] = frame.at[0, "trace_id"]
    with pytest.raises(DatasetValidationError):
        validate_trace_frame(frame)


def test_closed_world_specs_are_deterministic_and_valid(tiny_traces):
    specs = build_closed_world_specs(
        tiny_traces, seed=11, train_samples=40, validation_samples=20
    )
    again = build_closed_world_specs(
        tiny_traces, seed=11, train_samples=40, validation_samples=20
    )
    assert specs == again
    summary = validate_specs(specs, tiny_traces)
    assert summary["labels"] == 5
    assert summary["windows"]["train"] == 40
    assert summary["windows"]["validation"] == 20
    assert set(summary["count_distribution"]["train"]) == {"1", "2", "3", "4", "5"}


def test_validate_specs_detects_cross_split_leakage(tiny_traces):
    specs = build_closed_world_specs(
        tiny_traces, seed=3, train_samples=40, validation_samples=20
    )
    leaked = copy.deepcopy(specs)
    train_trace = int(tiny_traces[tiny_traces.split == "train"]["trace_id"].iloc[0])
    leaked["test"][0]["trace_ids"][0] = train_trace
    with pytest.raises(DatasetValidationError):
        validate_specs(leaked, tiny_traces)


def test_open_world_specs_grid_and_background():
    monitored = make_traces(num_sites=5, train_per_site=2, test_per_site=2)
    background = offset_trace_ids(make_traces(num_sites=0, train_per_site=0, test_per_site=0, background=5))
    background_ids = background["trace_id"].astype(int).tolist()
    specs = build_open_world_specs(
        monitored,
        background_ids,
        [0, 1, 2, 3],
        seed=5,
        samples_per_cell=4,
        monitored_counts=(1, 2),
        background_counts=(0, 2),
    )
    assert specs["train"] == [] and specs["validation"] == []
    assert len(specs["test"]) == 4 * 4
    assert len(specs["protocol"]["cells"]) == 4
    assert specs["protocol"]["background_used_for_training"] is False
    for row in specs["test"]:
        assert len(row["trace_ids"]) == row["monitored_websites_num"] + row["background_traces_num"]
        assert row["background_traces_num"] == 0 or set(row["trace_ids"]) & set(background_ids)
    summary = validate_specs(specs, monitored, background_trace_ids=background_ids)
    assert summary["windows"]["test"] == 16


def test_open_world_rejects_too_few_background_traces():
    monitored = make_traces(num_sites=2, train_per_site=1, test_per_site=1)
    with pytest.raises(DatasetValidationError):
        build_open_world_specs(
            monitored, [999], [0, 1], samples_per_cell=1,
            monitored_counts=(1,), background_counts=(3,),
        )
