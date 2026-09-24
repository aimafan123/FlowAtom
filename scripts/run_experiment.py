#!/usr/bin/env python3
"""Compatibility wrapper for ``flowatom run``."""

import _bootstrap  # noqa: F401
from flowatom.cli.run_experiment import main

if __name__ == "__main__":
    raise SystemExit(main())
