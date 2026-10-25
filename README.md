# Image classification basics

An basic, tutorial-style implementation of a feedforward neural network (Multilayer Perceptron) trained on the MNIST handwritten digit dataset

---

## Overview

The MNIST benchmark consists of $28 \times 28$ grayscale pixel images of handwritten digits ($0$ through $9$). This project implements a fully connected feedforward architecture that flattens spatial pixel matrices into 784-dimensional feature vectors, applies non-linear ReLU transformations across two hidden dense layers, and outputs class probabilities using a 10-way softmax distribution.

### Mathematical Formulation

1. **Affine Transformation (Layer $l$):**

$$z^{[l]} = W^{[l]} a^{[l-1]} + b^{[l]}$$

where $W^{[l]} \in \mathbb{R}^{d_{l} \times d_{l-1}}$ and $b^{[l]} \in \mathbb{R}^{d_{l}}$.
2. **Activation:**

$$a^{[l]} = \text{ReLU}(z^{[l]}) = \max(0, z^{[l]})$$

3. **Output Distribution (Softmax):**

$$\hat{y}_k = P(y = k \mid x) = \frac{e^{z_k^{[L]}}}{\sum_{j=0}^{9} e^{z_j^{[L]}}}$$

4. **Objective (Categorical Cross-Entropy):**

$$\mathcal{L}(y, \hat{y}) = -\sum_{k=0}^{9} y_k \log(\hat{y}_k)$$

---

## Network Architecture

| Layer | Type | Output Shape | Parameters | Activation |
| --- | --- | --- | --- | --- |
| `Input` | Input Layer | `(None, 784)` | 0 | — |
| `Dense_1` | Fully Connected | `(None, 128)` | 100,480 | ReLU |
| `Dense_2` | Fully Connected | `(None, 128)` | 16,512 | ReLU |
| `Dense_3` | Dense (Output) | `(None, 10)` | 1,290 | Softmax |

* **Total Trainable Parameters:** 118,282 (~462 KB)
* **Optimizer:** Adam ($\alpha = 0.001, \beta_1 = 0.9, \beta_2 = 0.999$)
* **Loss:** Categorical Cross-Entropy

---

## Directory Structure

```text
├── mnist_dense_classifier.py    # Main training and evaluation script
├── requirements.txt             # Dependency specification
├── README.md                    # Project documentation
└── assets/                      # Generated figures and sample predictions

```

---

## Getting Started

### Prerequisites

* Python 3.9+
* pip package manager

### Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
pip install -r requirements.txt

```

#### `requirements.txt`

```text
tensorflow>=2.10.0
numpy>=1.22.0
matplotlib>=3.5.0

```

### Running Locally

Execute the pipeline script:

```bash
python mnist_dense_classifier.py

```

### Running on Google Colab

Open a new Colab notebook, paste the script into a code cell, and install TensorFlow and Matplotlib if needed:

```python
!pip install tensorflow matplotlib numpy

```

---

## Pipeline Workflow

1. **Ingestion:** Loads the MNIST corpus (60,000 training samples, 10,000 test samples).
2. **Preprocessing:**
* Unrolls $28 \times 28$ image tensors to 784-element vectors.
* Normalizes grayscale intensity values from $[0, 255] \to [0.0, 1.0]$.
* Converts integer scalar labels into one-hot binary targets via `to_categorical(num_classes=10)`.


3. **Training & Validation:** Fits model parameters across mini-batches of size 64 over 10 epochs, withholding a 10% stratified validation slice.
4. **Evaluation:** Computes out-of-sample accuracy and loss over the uncorrupted 10,000-sample test partition.
5. **Diagnostics:**
* Plots a $5 \times 5$ prediction matrix color-coded by correctness (green = correct, red = error).
* Visualizes categorical softmax prediction entropy on individual test samples.

---

## Expected Results

| Metric | Training Split | Validation Split | Test Split |
| --- | --- | --- | --- |
| **Loss** | ~0.03 | ~0.08 | ~0.08 |
| **Accuracy** | ~99.0% | ~97.8% | ~97.5% - 98.0% |

