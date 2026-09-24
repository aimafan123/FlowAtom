#!/usr/bin/env python3
"""Compatibility wrapper for ``flowatom materialize-pretraining-shards``."""

import _bootstrap  # noqa: F401
from flowatom.cli.materialize_pretraining_shards import main

if __name__ == "__main__":
    raise SystemExit(main())
