<h1 align="center">Hybrid-Precision ACMPC</h1>

<p align="center">
  <em>A faithful reproduction of <strong>Actor-Critic Model Predictive Control</strong> - plus an honest attempt to make it fast. [Pardon in case of any mistake :) ]</em>
</p>

<p align="center">
  <a href="https://rpg.ifi.uzh.ch/docs/TRO25_ACMPC_Romero.pdf"><img src="https://img.shields.io/badge/paper-TRO_2025-a8611a" alt="Paper"/></a>
  <a href="https://arxiv.org/abs/2306.09852"><img src="https://img.shields.io/badge/arXiv-2306.09852-b31b1b" alt="arXiv"/></a>
  <a href="#quickstart"><img src="https://img.shields.io/badge/python-3.10-2a78d6" alt="Python 3.10"/></a>
  <a href="#quickstart"><img src="https://img.shields.io/badge/PyTorch-2.5-eb6834" alt="PyTorch 2.5"/></a>
  <a href="docs/report.html"><img src="https://img.shields.io/badge/report-interactive-4c9a2a" alt="Report"/></a>
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/90a03305-100a-4966-ab81-0ecc121f3d54" alt="ACMPC closed-loop rollout on an 8-gate circular track" width="100%"/>
  <br>
  <em>A trained ACMPC policy flying an 8-gate circular track — real closed-loop rollout, no cherry-picking.</em>
<p>

---

## TL;DR

- **Reproduced the paper's central robustness claim.** ACMPC beats its model-free counterpart ACMLP by **+20.7 pp** out-of-distribution and **+48.7 pp** under wind, at n = 150–300 episodes.


---

## Overview

**ACMPC** (Romero, Aljalbout, Song & Scaramuzza, *IEEE T-RO 2025*) embeds a differentiable Model Predictive Controller inside an actor-critic RL loop. Instead of emitting an action, the policy network emits a **cost function**; a differentiable box-constrained iLQR solver (`mpc.pytorch`) turns that cost into the actual control — while staying differentiable end to end, so PPO's policy gradient flows straight through the solve and back into the network weights.

This repository:

1. **Reproduces ACMPC and ACMLP** from the authors' released algorithmic core (`acmpc_public`), which ships the dynamics model and policies but **no training environment** — the paper trains inside a private Flightmare wrapper. Two paper-named tracks (`circular`, `split_s`) and a `hover` sanity task were built from scratch on the same dynamics the MPC uses internally.
2. **Validates against the paper's actual claims**, not just "it trains" — reproducing the Table I robustness edge with real statistics, across two independent wind models.
3. **Builds a hybrid FP16/FP32 precision variant** and measures, honestly, whether it delivers the latency win it was designed for. (It doesn't. The reason is the interesting part.)


---

## Architecture

|  | **ACMLP** | **ACMPC** | **ACMPC — Hybrid** |
|---|---|---|---|
| **Actor output** | action, directly | cost function `(Q, p)` | same as ACMPC |
| **Network -> motors** | nothing | differentiable MPC solve | fp16 network -> fp32 MPC solve *(unchanged)* |
| **Precision** | fp32 | fp32 | fp16 trunk, fp32 solver |
| **Critic** | plain MLP | plain MLP (no solver) | plain MLP |
| **Real-time capable?** | yes | no - ~300–600 ms/step | no - same solver dominates |

---

## Reproducing the paper's claim

| Test | ACMLP | ACMPC | Gap | Paper's Table I *(different track/sim)* |
|---|---:|---:|---:|---|
| Nominal (in-distribution) | 100.0% | 100.0% | — | not published |
| Widened start position (OOD) | 59.3% | **80.0%** | **+20.7 pp** | 74.8% → 90.4% |
| Wind disturbance | 47.3% | **96.0%** | **+48.7 pp** | 6.5% → 83.3% |

Measured at n = 150–300 episodes. The wind result sits at **~12+ standard errors** - not noise - and was cross-validated on two independent wind disturbance models.

**This did not come easily.** The first attempt, at a 2M-timestep budget, reproduced the *opposite* result: ACMPC losing to ACMLP under wind, consistently. Chasing it down — a mechanism diagnostic, then a more physically realistic wind model, then more training budget — found the real cause: **undertraining**. The paper reports ~11.5 hours of ACMPC training; the initial reproduction used ~24 minutes. Scaling to 8M steps, after also catching and fixing a **silent PPO training collapse** (an unguarded KL-divergence spike; `target_kl` was never enabled), recovered the paper's pattern cleanly.



https://github.com/user-attachments/assets/52e5dc94-5668-4c8c-8a97-f3ef8a3e8fe1



---

## Trying to make it fast

The hybrid-precision premise was that the network is the expensive part. Profiling the real solver said otherwise.


- **The trunk network is under 0.1% of per-step latency**, at every batch size tested. The differentiable-MPC solve — which *must* stay fp32, provably, since its own cost values already overflow fp16's range on ordinary rollouts — dominates completely at **300–600 ms per call**. FP16 in the trunk measured **1.04×** at batch 1 and **0.69× (slower)** at batch 64.
- **Attacking the solver directly** — skipping a provably redundant gradient-setup pass during inference, and cutting PNQP's inner iteration cap from 20 to 1 — held accuracy completely flat (100% success, identical return, at every setting) but bought only **1.17×**. The 234 ms bottleneck isn't iteration count; it's **fixed sequential structure**: a 5-step Python loop over many small, launch-overhead-heavy GPU ops.

> **Verdict.** The hybrid split is correctly designed, correctly implemented, and behavior-preserving — 200-episode closed-loop parity at **112.17 vs. 112.20** return, 100% = 100% success. But it does **not** solve ACMPC's real-time latency problem. That problem lives in the solver's sequential kernel-launch structure, not in numeric precision. Closing it for real would need CUDA-graph capture or a native reimplementation of the solve loop — a materially larger effort than precision tuning.

---

## Quickstart

### Prerequisites

- Python 3.10
- PyTorch 2.5+ with CUDA 12.1
- A CUDA-capable GPU (strongly recommended)

### Installation

```bash
git clone --recurse-submodules https://github.com/your-username/hybrid-AC_MPC.git
cd hybrid-precision-acmpc

python3 -m venv venv && source venv/bin/activate

pip install "pip==24.0"                 # gym==0.21's metadata needs this — see CLAUDE.md
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install "gym==0.21" numpy scipy matplotlib setproctitle pandas tensorboard --no-build-isolation
pip install -e acmpc_public/mpc.pytorch -e acmpc_public/stable-baselines3
```

### Train and evaluate

```bash
# train ACMPC on the circular track
python scripts/train.py \
    --policy acmpc --env circular \
    --timesteps 2000000 --n-envs 256 --horizon 5

# evaluate the resulting checkpoint
python scripts/eval.py \
    --policy acmpc --env circular \
    --checkpoint runs/acmpc_circular_seed0/model.zip
```

### Full reproduction sweep

```bash
./scripts/run_reproduction.sh          # all 6 base configs
```

**Flags.** `--policy` is `acmlp` or `acmpc`. `--env` is `hover`, `circular`, or `split_s`.

> ⚠️ **Long runs should go through `scripts/train_chunked.py`, not `train.py`.** It restarts the training process periodically to work around a real memory leak in `mpc.pytorch`. 

---

## Repository layout

```
hybrid-precision-acmpc/
├── README.md, HISTORY.md, PLAN.md, CLAUDE.md   the four docs — start with whichever matches your question
├── docs/            interactive report (report.html) + figures
├── acmpc_public/    authors' reference release (dynamics, MPC solver, SB3 fork)
├── envs/            our training environments — acmpc_public ships none
├── hybrid/          the FP16-trunk / FP32-solver policy variants
├── scripts/         every runnable script (train / eval / plot / profile / sweep + shell drivers)
└── runs/            checkpoints, tensorboard logs, trajectory plots, flight videos
```

---

## Citation

This work builds directly on the paper's algorithmic core. If you use ACMPC itself, please cite:

```bibtex
@article{romero2025acmpc,
  author  = {Romero, Angel and Aljalbout, Elie and Song, Yunlong and Scaramuzza, Davide},
  title   = {Actor-Critic Model Predictive Control: Differentiable Optimization
             meets Reinforcement Learning for Agile Flight},
  journal = {IEEE Transactions on Robotics},
  year    = {2025},
}
```

## Acknowledgements

Built on [`uzh-rpg/acmpc_public`](https://github.com/uzh-rpg/acmpc_public), [`locuslab/mpc.pytorch`](https://github.com/locuslab/mpc.pytorch), and a fork of [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3).
