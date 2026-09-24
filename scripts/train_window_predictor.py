#!/usr/bin/env python3
"""Compatibility wrapper for ``flowatom train-window-predictor``."""

import _bootstrap  # noqa: F401
from flowatom.cli.train_window_predictor import main

if __name__ == "__main__":
    raise SystemExit(main())
