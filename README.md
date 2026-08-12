# Alzheimer MRI Classification

## Trained models and Git LFS

The trained CNN checkpoints are stored directly in the repository. The larger
ResNet-50 checkpoint, `models/alzheimer_resnet50_best.pt`, is approximately
99 MB and is stored using [Git Large File Storage (Git LFS)](https://git-lfs.com/).

Install Git LFS before cloning the repository. For example, on Ubuntu or
Debian:

```bash
sudo apt-get install git-lfs
```

On macOS with Homebrew, use `brew install git-lfs`. Then initialize Git LFS
once for your user account:

```bash
git lfs install
```

A normal `git clone` will then download the complete ResNet-50 checkpoint. For
an existing clone, retrieve any missing LFS objects with:

```bash
git pull
git lfs pull
```

Verify that the model was downloaded rather than left as a small LFS pointer:

```bash
git lfs ls-files
ls -lh models/alzheimer_resnet50_best.pt
```

The model file should be approximately 95 MiB. With `RETRAIN_RESNET = False`,
the notebook loads this checkpoint and skips the ResNet-50 hyperparameter
search and training.

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

## Google Colab

Install Git LFS, clone the repository, retrieve the trained ResNet-50 model and
install only the non-PyTorch dependencies:

```bash
!apt-get update -qq
!apt-get install -y git-lfs
!git lfs install
!git clone <repository-url>
%cd Deep-Learning-For-Human-Body-Marker-Augmentation
!git lfs pull
!pip install -r requirements-colab.txt
```

Do not install `requirements.txt` in Colab because it intentionally pins the
CPU-only PyTorch build. Colab's preinstalled CUDA-enabled PyTorch should be
used instead. Enable a GPU under **Runtime > Change runtime type** before
running the notebook.

On Colab, the notebook mounts Google Drive and stores in-progress ResNet-50
search state under:

```text
MyDrive/alzheimer-classification/models/
```

If `models/alzheimer_resnet50_best.pt` exists in the cloned repository, that
committed model takes precedence and is loaded without starting a new search.
If the dataset is stored at
`MyDrive/alzheimer-classification/Alzheimer_s Dataset`, the notebook copies it
to Colab's temporary local storage for faster training.

## Dataset layout

Download the Kaggle dataset `tourist55/alzheimers-dataset-4-class-of-images`
and extract it without combining or renaming its original folders. The
notebook expects this layout:

```text
Alzheimer_s Dataset/
├── train/
│   ├── NonDemented/
│   ├── VeryMildDemented/
│   ├── MildDemented/
│   └── ModerateDemented/
└── test/
    ├── NonDemented/
    ├── VeryMildDemented/
    ├── MildDemented/
    └── ModerateDemented/
```

The notebook verifies the expected 5,121/1,279 Kaggle split, creates
validation data only from the training directory, and leaves the original
test directory untouched for final evaluation.

For later sessions, only activation and Jupyter startup are needed:

```bash
source .venv/bin/activate
jupyter lab
```
