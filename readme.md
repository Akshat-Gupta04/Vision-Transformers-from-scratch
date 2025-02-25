# Vision Transformer for MNIST Image Classification

This repository contains an implementation of a Vision Transformer (ViT) built from scratch for image classification. The model has been tested on the MNIST dataset and uses a custom encoder layer with multi-head self-attention and a feed-forward network.

## Overview

The Vision Transformer (ViT) in this project:
- **Splits the input image into patches** using a convolutional layer (with kernel size and stride equal to the patch size).
- **Embeds the patches** into a sequence using a linear projection, prepending a learnable class token.
- **Adds positional embeddings** to the sequence.
- **Processes the sequence through a stack of custom encoder layers**, each comprising:
  - Multi-head self-attention.
  - A feed-forward block with GELU activation.
  - Residual connections and layer normalization.
- **Extracts the class token** and passes it through an MLP head to produce class logits.

The architecture is designed to be modular so that it can be easily modified or extended.

## Tested on MNIST

The model has been evaluated using the MNIST dataset for handwritten digit classification (10 classes). The MNIST images are 28×28 grayscale images. The dataset splits are handled using custom dataset classes:
- `MNISTTrainDataset`
- `MNISTValDataset`
- `MNISTSubmitDataset`

These classes are defined in **dataset.py** and include common transformations like normalization and (for training) random rotations.

## Project Structure

- **hyperparams.py**: Contains all the hyperparameter definitions and seed setups for reproducibility.
- **model.py**: Contains the complete Vision Transformer architecture including patch embedding, custom encoder layers, multi-head self-attention, feed-forward blocks, and the final MLP head.
- **dataset.py**: Contains custom PyTorch dataset classes for training, validation, and test (submission) data.
- **train.py**: Main training script that:
  - Loads the MNIST CSV files.
  - Instantiates the dataset objects and corresponding DataLoaders.
  - Trains the Vision Transformer, validates the model, and plots loss/accuracy curves.
  - Evaluates the test data and displays a grid of sample predictions.
- **run.ipynb**: A Jupyter Notebook version of the training script for those who prefer working interactively.

## Dependencies

The project requires Python 3 and the following packages:

- torch
- torchvision
- numpy
- pandas
- matplotlib
- tqdm

Install the dependencies using pip:

```bash
pip install torch torchvision numpy pandas matplotlib tqdm# Vision Transformer for MNIST Image Classification

This repository contains an implementation of a Vision Transformer (ViT) built from scratch for image classification. The model has been tested on the MNIST dataset and uses a custom encoder layer with multi-head self-attention and a feed-forward network.

## Overview

The Vision Transformer (ViT) in this project:
- **Splits the input image into patches** using a convolutional layer (with kernel size and stride equal to the patch size).
- **Embeds the patches** into a sequence using a linear projection, prepending a learnable class token.
- **Adds positional embeddings** to the sequence.
- **Processes the sequence through a stack of custom encoder layers**, each comprising:
  - Multi-head self-attention.
  - A feed-forward block with GELU activation.
  - Residual connections and layer normalization.
- **Extracts the class token** and passes it through an MLP head to produce class logits.

The architecture is designed to be modular so that it can be easily modified or extended.

## Tested on MNIST

The model has been evaluated using the MNIST dataset for handwritten digit classification (10 classes). The MNIST images are 28×28 grayscale images. The dataset splits are handled using custom dataset classes:
- `MNISTTrainDataset`
- `MNISTValDataset`
- `MNISTSubmitDataset`

These classes are defined in **dataset.py** and include common transformations like normalization and (for training) random rotations.

## Project Structure

- **hyperparams.py**: Contains all the hyperparameter definitions and seed setups for reproducibility.
- **model.py**: Contains the complete Vision Transformer architecture including patch embedding, custom encoder layers, multi-head self-attention, feed-forward blocks, and the final MLP head.
- **dataset.py**: Contains custom PyTorch dataset classes for training, validation, and test (submission) data.
- **train.py**: Main training script that:
  - Loads the MNIST CSV files.
  - Instantiates the dataset objects and corresponding DataLoaders.
  - Trains the Vision Transformer, validates the model, and plots loss/accuracy curves.
  - Evaluates the test data and displays a grid of sample predictions.
- **run.ipynb**: A Jupyter Notebook version of the training script for those who prefer working interactively.

## Dependencies

The project requires Python 3 and the following packages:

- torch
- torchvision
- numpy
- pandas
- matplotlib
- tqdm

Install the dependencies using pip:

```bash
pip install torch torchvision numpy pandas matplotlib tqdm