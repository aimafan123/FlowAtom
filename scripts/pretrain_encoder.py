#!/usr/bin/env python3
"""Compatibility wrapper for ``flowatom pretrain-encoder``."""

import _bootstrap  # noqa: F401
from flowatom.cli.pretrain_encoder import main

if __name__ == "__main__":
    raise SystemExit(main())
