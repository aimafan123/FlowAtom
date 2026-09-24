#!/usr/bin/env python3
"""Compatibility wrapper for ``flowatom build-trace-atoms``."""

import _bootstrap  # noqa: F401
from flowatom.cli.build_trace_atoms import main

if __name__ == "__main__":
    raise SystemExit(main())
