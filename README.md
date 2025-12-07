# ESE3060 Final Project – CIFAR-10 Speedrun

Small-scale replication and extension of the **Airbench94** CIFAR-10 “speedrun” setup, with a focus on:
- Fast training to ~94% test accuracy on CIFAR-10
- Careful timing and logging over many runs
- Exploring small, **targeted tweaks** to the original recipe

This repo contains our baseline implementation plus several ablation / modification experiments.

---

## Repository Structure

At a high level:

- `models/`
  - `baseline.py` – our faithful reproduction of the airbench94 training script.
  - `alt_translate.py` – variant with a more structured translation schedule for data augmentation.
  - `dynamic_ls.py` – variant with **dynamic label smoothing** over training.
  - `whiten_unfreeze.py` – variant that **late-unfreezes** the patch-whitening conv layer.
- `logs/`
  - Created automatically when you run any model script.
  - Structure: `logs/<exp_name>/<timestamp>/log.pt` and `log.txt`.
- `requirements.txt`
  - 

### Prerequisites
- Python 3.8+
- NVIDIA GPU (A100/H100 recommended)
- CUDA 11.7 or later

### Dependencies
Install all required packages:
```bash
pip install -r requirements.txt
```

## Models / Variants

This directory contains the baseline Airbench94 reproduction and several controlled variants exploring augmentation, loss shaping, and whitening behavior.  
All variants inherit the same training loop, optimizer structure, and logging framework unless otherwise noted.

---

### `baseline.py`

**Goal:**  
Faithful reproduction of **Airbench94**.

**Key features:**
- 2×2 patch-whitening convolution as the first layer:
  - Initialized from training patch covariance
  - Weights frozen throughout training
  - Biases trained for a small number of early epochs
- Fully decoupled hyperparameters for:
  - Learning rate
  - Momentum
  - Weight decay
- Label smoothing (default: `0.2`)
- Test-Time Augmentation (TTA), controlled by `hyp['net']['tta_level']`

**Logging:**
- `train_loss`
- `train_acc`
- `val_acc`
- `tta_val_acc`
- Cumulative training time (seconds)

This model serves as the **reference point** for all comparisons.

---

### `alt_translate.py`

**Idea:**  
Preserve the same augmentation *space* as the baseline, but make translation deterministic at the epoch level.

**Implementation:**
- Uses a fixed list of `(dy, dx)` translation offsets
- Each epoch applies **one specific offset to all images**
- Replaces per-image random cropping with a structured, cyclic pattern

**Motivation / Hypothesis:**
- More predictable memory access patterns
- Potential throughput improvements
- Minimal impact on accuracy

**Observed behavior (on our hardware):**
- Slightly reduced training time compared to baseline
- Accuracy comparable to baseline

---

### `dynamic_ls.py`

**Idea:**  
Replace constant label smoothing with a **schedule**.

**Method:**
- Start with `label_smoothing`
- Linearly anneal to `label_smoothing_final` by the end of training
- Intended to:
  - Stabilize early optimization
  - Encourage sharper predictions near convergence

**Implementation details:**
- Custom `label_smoothing_loss()` returning per-example loss
- At each step:
  - Compute training progress = `current_steps / total_train_steps`
  - Linearly interpolate between start and final smoothing
- Logs effective label smoothing per epoch

**Outcome:**
- Accuracy and runtime very close to baseline
- Largely a **negative result**, but included for completeness and analysis

---

### `whiten_unfreeze.py`

**Idea:**  
Investigate the effect of **unfreezing the whitening convolution weights** later in training.

**Background:**
- The Airbench94 setup keeps whitening weights frozen (bias-only training)

**Implementation:**
- New hyperparameter: `hyp['opt']['whiten_unfreeze_epoch']`
- Whitening convolution weights:
  - Frozen initially
  - Unfrozen after the specified epoch
- Separate optimizer parameter group for whitening weights
- Training history records whether whitening weights are trainable each epoch

**Outcome:**
- Increased training time
- Degraded accuracy relative to baseline
- Indicates that frozen whitening is important for both speed and performance

Despite underperforming, this variant helps quantify how sensitive the model is to deviations from the original whitening design.

---