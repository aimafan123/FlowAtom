#!/usr/bin/env python3
"""Compatibility wrapper for ``flowatom build-mixture-specs``."""

import _bootstrap  # noqa: F401
from flowatom.cli.build_mixture_specs import main

if __name__ == "__main__":
    raise SystemExit(main())
