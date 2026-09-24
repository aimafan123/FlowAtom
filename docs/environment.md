# Environment

## CPU reference

From the repository root, create a clean virtual environment and install the
reference versions of the direct runtime dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install "torch==2.5.1" --index-url https://download.pytorch.org/whl/cpu
python -m pip install -c requirements/constraints-cpu.txt -e ".[all,dev]"
python -m pip check
```

## CUDA and paper runs

Install a PyTorch build suitable for your GPU and driver before installing
`.[all]`. Use the [official PyTorch installation instructions](https://pytorch.org/get-started/locally/)
to select the wheel index. Pass `--device cuda:0` to `flowatom run`; the default
is `cpu`. CUDA accelerates encoder computation, predictor training and the
XGBoost mapper. k-means still uses CPU resources.

Experimental environment: Linux, Python 3.10.20, PyTorch 2.1.2+cu118,
NumPy 1.26.4, scikit-learn 1.7.2, XGBoost 2.1.4, PyArrow 24.0.0 and an NVIDIA
RTX A6000.
