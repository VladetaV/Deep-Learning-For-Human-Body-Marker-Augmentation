# Alzheimer MRI Classification

## Environment setup

Create a project-local Python 3.12 virtual environment:

```bash
/usr/bin/python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The dependency file uses CPU-only PyTorch packages to avoid downloading the
much larger CUDA toolkit on machines without an NVIDIA GPU.

Register the environment as a Jupyter kernel:

```bash
python -m ipykernel install --user \
  --name alzheimer-classification \
  --display-name "Alzheimer Classification (.venv)"
```

Start JupyterLab from the activated environment:

```bash
jupyter lab
```

Open `alzheimer_classification.ipynb` and select the
`Alzheimer Classification (.venv)` kernel before running the cells.

For later sessions, only activation and Jupyter startup are needed:

```bash
source .venv/bin/activate
jupyter lab
```
