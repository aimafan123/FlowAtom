#!/usr/bin/env python3
"""Compatibility wrapper for ``flowatom make-synthetic-data``."""

import _bootstrap  # noqa: F401
from flowatom.cli.make_synthetic_data import main

if __name__ == "__main__":
    raise SystemExit(main())
