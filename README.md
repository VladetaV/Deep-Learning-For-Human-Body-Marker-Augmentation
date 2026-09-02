# Alzheimer MRI Classification

## About the project

This project was created as part of the Machine Learning course at the
Faculty of Mathematics, University of Belgrade. It explores deep learning
methods for classifying brain MRI images into four stages of Alzheimer's
disease: non-demented, very mildly demented, mildly demented and moderately
demented.

The project compares convolutional neural network baselines, an
ImageNet-pretrained ResNet-50, supervised contrastive learning and
self-supervised contrastive learning. It also examines targeted data
augmentation, class imbalance and classification with a reduced set of
labels. The implementation is based on and extends the experiments described
in [Deep Learning Methods for Alzheimer's Disease Prediction](https://cs230.stanford.edu/projects_fall_2022/reports/108.pdf)
by Fang Shu and Longling Tian.

### Authors

- [Jovan Vukićević](https://github.com/jvukicev)
- [Vladeta Vujačić](https://github.com/VladetaV)

## Trained models and Git LFS

The two small CNN checkpoints are stored directly in Git. The larger
ResNet-50 and contrastive-learning checkpoints are approximately 95 MiB each
and are stored using [Git Large File Storage (Git LFS)](https://git-lfs.com/).
These include the ImageNet-pretrained ResNet-50, the supervised contrastive
encoder and its classifiers, and the self-supervised SimCLR encoder and its
classifiers, including the one trained with 10% of the labels. Together, the
LFS-tracked checkpoints occupy approximately 750 MiB.

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

A normal `git clone` will then download all committed model checkpoints. For
an existing clone, retrieve any missing LFS objects with:

```bash
git pull
git lfs pull
```

Verify that the models were downloaded rather than left as small LFS
pointers:

```bash
git lfs ls-files
ls -lh models/*.pt
```

The large checkpoint files should be approximately 95 MiB each. When the
corresponding retraining flags are set to `False`, the notebook loads the
committed checkpoints and skips the associated searches or training runs.
These flags are `RETRAIN_MODELS`, `RETRAIN_RESNET`, `RETRAIN_CONTRASTIVE`,
`RETRAIN_CONTRASTIVE_CLASSIFIER` and `RETRAIN_SIMCLR`.

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

Install Git LFS, clone the repository, retrieve the trained ResNet-50 and
contrastive-learning models, and install only the non-PyTorch dependencies:

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

On Colab, the notebook mounts Google Drive and stores in-progress search state
and newly trained checkpoints for the ResNet-50 and contrastive-learning
experiments under:

```text
MyDrive/alzheimer-classification/models/
```

When a matching checkpoint exists in the cloned repository and its retraining
flag is `False`, the committed model takes precedence over the Google Drive
copy and is loaded without repeating its search or training run. If the
dataset is stored at
`MyDrive/alzheimer-classification/Alzheimer_s Dataset`, the notebook copies it
to Colab's temporary local storage for faster training.

## Dataset layout

Download the [Alzheimers Disease Dataset from Kaggle](https://www.kaggle.com/datasets/kumarln/alzheimers-disease-dataset)
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
