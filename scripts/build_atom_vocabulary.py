#!/usr/bin/env python3
"""Compatibility wrapper for ``flowatom build-atom-vocabulary``."""

import _bootstrap  # noqa: F401
from flowatom.cli.build_atom_vocabulary import main

if __name__ == "__main__":
    raise SystemExit(main())
