# ESE3060 Final Project – CIFAR-10 Speedrun

Small-scale replication and extension of the **Airbench94** CIFAR-10 “speedrun” setup, with a focus on

- fast training to ~94% test accuracy on CIFAR-10,
- careful timing and logging over many runs, and  
- exploring small, **targeted tweaks** to the original recipe, especially in the mid-resolution convolutional block.

This repo contains our baseline implementation plus several ablation / modification experiments.

---

## Repository structure

At a high level:

- `models/`
  - `baseline.py` – faithful reproduction of the original `airbench94.py` training script.
  - `width_rebalance.py` – **narrower mid-resolution block** (block2 width reduced from 256→128).
  - `block2_group4.py` – **grouped mid-resolution conv** (block2, conv2 uses 4 groups).
  - `alt_translate.py` – structured translation schedule for data augmentation.
  - `orbit_aug.py` – more aggressive “orbit-style’’ crop/flip augmentation over epochs.
  - `dynamic_ls.py` – **dynamic label smoothing** schedule over training.
  - `whitening_unfreeze.py` – late unfreezing of the patch-whitening conv layer.
  - `hyp_sweep.ipynb` – Jupyter notebook for small hyperparameter sweeps (e.g., over
    learning rate, label smoothing, and block widths) that drives the CLI arguments in
    `width_rebalance.py` and `block2_group4.py`.
- `logs/`
  - Created automatically when you run any model script.
  - Structure: `logs/<exp_name>/<timestamp>/log.pt` and `log.txt`.
  - Each `log.pt` stores:
    - the exact training code snapshot,
    - per-run accuracy and timing,
    - and per-epoch history (train loss/acc, val acc, cumulative time).

The short written report for the assignment is `ESE3060_Final_Project_Part_1.pdf` in the repo
root.

---

## Prerequisites

- Python 3.8+
- NVIDIA GPU (A100/H100 recommended)
- CUDA 11.7 or later

### Dependencies

Install all required packages:

```bash
pip install -r requirements.txt
