#!/usr/bin/env python3
"""Compatibility wrapper for ``flowatom evaluate-frozen``."""

import _bootstrap  # noqa: F401
from flowatom.cli.evaluate_frozen import main

if __name__ == "__main__":
    raise SystemExit(main())
