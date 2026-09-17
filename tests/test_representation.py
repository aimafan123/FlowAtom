"""Tests for packet-sequence representations."""

import numpy as np
import pytest

from flowatom.representation import (
    RepresentationError,
    flow_representation,
    payload_direction_sequences,
    signed_flow_values,
)

PAYLOADS = [0, 100, 250, 0, 40, 1500, 0, 33]
DIRECTIONS = [1, 1, -1, -1, 1, -1, 1, -1]


def test_signed_payload_length_removes_zeros_and_encodes_direction():
    sequence = flow_representation(
        PAYLOADS, DIRECTIONS, representation="signed_payload_length",
        input_length=10, min_payload_packets=3,
    )
    assert sequence is not None
    assert sequence.dtype == np.int32
    assert sequence.tolist() == [100, -250, 40, -1500, -33, 0, 0, 0, 0, 0]


def test_payload_length_and_direction_only():
    lengths = flow_representation(
        PAYLOADS, DIRECTIONS, representation="payload_length",
        input_length=5, min_payload_packets=3,
    )
    signs = flow_representation(
        PAYLOADS, DIRECTIONS, representation="direction_only",
        input_length=5, min_payload_packets=3,
    )
    assert lengths.tolist() == [100, 250, 40, 1500, 33]
    assert signs.tolist() == [1, -1, 1, -1, -1]


def test_minimum_payload_packets_filters_flow():
    assert flow_representation(
        PAYLOADS, DIRECTIONS, input_length=10, min_payload_packets=9
    ) is None


def test_unaligned_sequences_raise():
    with pytest.raises(RepresentationError):
        payload_direction_sequences([1, 2], [1], input_length=4, min_payload_packets=1)


def test_unknown_representation_raises():
    with pytest.raises(RepresentationError):
        flow_representation(PAYLOADS, DIRECTIONS, representation="nope")


def test_signed_flow_values_align_lengths():
    values = signed_flow_values([[10, 20, 30]], [[1, -1]])
    assert values[0].tolist() == [10.0, -20.0]
