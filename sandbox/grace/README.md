# Grace – Onboarding Project

A small PyTorch neural network that classifies handwritten digits (0–9) from the MNIST dataset.

## What it does

- Downloads MNIST into `sandbox/grace/data/` (ignored by Git)
- Trains a fully connected network (784 → 128 → 10) for 3 epochs with the Adam optimizer
- Prints the training loss and test accuracy after each epoch (about 97% after 3 epochs)

## How to run

From the repository root:

    uv run sandbox/grace/train.py