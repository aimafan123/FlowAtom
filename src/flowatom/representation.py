"""Packet-sequence representations for flow encoding.

FlowAtom removes zero-payload packets, keeps packet order, and represents a
flow as a fixed-length sequence of ``abs(payload length) * sign(direction)``.
Sequences longer than ``input_length`` are truncated and shorter sequences are
right-padded with zeros, matching the preprocessing used for pretraining and
for Atom extraction.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

REPRESENTATIONS = ("signed_payload_length", "payload_length", "direction_only")


class RepresentationError(ValueError):
    """Raised when a packet sequence violates the representation contract."""


def payload_direction_sequences(
    payloads: Sequence[float],
    directions: Sequence[float],
    *,
    input_length: int,
    min_payload_packets: int,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Return aligned absolute-length and direction arrays.

    Zero-payload packets are dropped because they carry no length evidence.
    ``None`` is returned when fewer than ``min_payload_packets`` packets remain,
    so callers can decide whether to skip the flow or fail loudly.
    """

    if len(payloads) != len(directions):
        raise RepresentationError("payload and direction sequences must align")
    if input_length <= 0:
        raise RepresentationError("input_length must be positive")
    retained = [
        (int(abs(float(payload))), 1 if float(direction) > 0 else -1)
        for payload, direction in zip(payloads, directions)
        if float(payload) != 0.0
    ]
    if len(retained) < min_payload_packets:
        return None
    lengths = np.zeros(input_length, dtype=np.int32)
    signs = np.zeros(input_length, dtype=np.int32)
    clipped = retained[:input_length]
    lengths[: len(clipped)] = [item[0] for item in clipped]
    signs[: len(clipped)] = [item[1] for item in clipped]
    return lengths, signs


def flow_representation(
    payloads: Sequence[float],
    directions: Sequence[float],
    *,
    representation: str = "signed_payload_length",
    input_length: int = 300,
    min_payload_packets: int = 10,
) -> Optional[np.ndarray]:
    """Build one controlled flow representation, or ``None`` when filtered."""

    if representation not in REPRESENTATIONS:
        raise RepresentationError(f"unsupported representation: {representation}")
    components = payload_direction_sequences(
        payloads,
        directions,
        input_length=input_length,
        min_payload_packets=min_payload_packets,
    )
    if components is None:
        return None
    lengths, signs = components
    if representation == "signed_payload_length":
        return lengths * signs
    if representation == "payload_length":
        return lengths
    return signs


def signed_payload_length_sequence(
    payloads: Sequence[float],
    directions: Sequence[float],
    *,
    input_length: int = 300,
    min_payload_packets: int = 10,
) -> Optional[np.ndarray]:
    """Convenience wrapper for the paper's signed payload-length sequence."""

    return flow_representation(
        payloads,
        directions,
        representation="signed_payload_length",
        input_length=input_length,
        min_payload_packets=min_payload_packets,
    )


def signed_flow_values(payload_flows, direction_flows):
    """Return per-flow signed payload values for external flow classifiers."""

    output = []
    for payload, direction in zip(payload_flows, direction_flows):
        length = min(len(payload), len(direction))
        output.append(
            np.asarray(payload, dtype=np.float32)[:length]
            * np.asarray(direction, dtype=np.float32)[:length]
        )
    return output
