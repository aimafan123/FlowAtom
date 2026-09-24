#!/usr/bin/env python3
"""Compatibility wrapper for ``flowatom build-open-world-specs``."""

import _bootstrap  # noqa: F401
from flowatom.cli.build_open_world_specs import main

if __name__ == "__main__":
    raise SystemExit(main())
