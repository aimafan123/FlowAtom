#!/usr/bin/env python3
"""Compatibility wrapper for ``flowatom summarize-results``."""

import _bootstrap  # noqa: F401
from flowatom.cli.summarize_results import main

if __name__ == "__main__":
    raise SystemExit(main())
