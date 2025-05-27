# Learning Rate Scaling for Qwen Models with μP

This repository provides a sample project for optimizing the learning rate of Qwen models using **μP (Maximal Update Parametrization)**, a powerful framework for hyperparameter transfer across model scales.

## Overview

Training large models often involves expensive hyperparameter tuning. μP helps mitigate this by allowing you to tune on a small model and transfer the optimal settings to larger models. This repository demonstrates how to:

* Set up a μP-compatible Qwen model
* Verify the correctness of μP scaling behavior
* Identify an appropriate learning rate via scaling experiments

## Setup

First, build and launch the Docker environment:

```bash
make build
make run
```

Then, generate the base model and shape configuration. You can edit the model settings in `setting.yml`.

```bash
python create_base_shapes.py
```

This step defines the base shapes required by μP to align the parameterization across different model widths.

## μP Coordination Check

To verify that μP scaling is working correctly, run the following script:

```bash
python coord_check.py
```

This script examines how the scale of activations (referred to as "coordinates") changes as the model width increases. Ideally, the coordinate scale should remain stable if μP is properly applied.

You can visualize the output results using the provided Jupyter notebook.

<img width="765" alt="coord_check" src="https://github.com/user-attachments/assets/98d5a3fa-24d8-43af-94f4-e3bd3d8eb229" />

## Learning Rate Scaling Check

Next, run a learning rate sweep across various model widths to identify a scale-invariant learning rate:

```bash
python lr_check.py
```

This script trains models with widths ranging from 80 to 640 for 3000 steps using different learning rates. Results can again be visualized with the provided notebook.

<img width="774" alt="lr_check" src="https://github.com/user-attachments/assets/41776c37-6312-468f-a963-8a03c42827bb" />

## Notes

* The experiments in this repository assume that the Qwen model is already μP-compatible. If not, refer to μP documentation to modify the parameterization accordingly.
* Ensure `set_base_shapes()` is called correctly on all μP layers before training.
