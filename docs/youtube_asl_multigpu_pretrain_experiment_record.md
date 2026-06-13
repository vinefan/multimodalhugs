# YouTube-ASL Multi-GPU Pretrain Experiment Record

## Scope

This document records the multi-GPU SignCLIP pretraining experiments on
`YouTube-ASL`, with emphasis on distributed-training behavior and follow-up
diagnostic directions.

## Current Multi-GPU Setup

- GPUs: `4 x H100`
- Per-device train batch size: `128`
- Global train batch size: `512`
- Per-device eval batch size: `64`
- Global eval batch size: `256`
- Distributed strategy: DDP with cross-GPU gathered embeddings
- Distributed negatives: enabled
- Learning rate: `5e-5`
- Warmup steps: `1000`

## Observed Failure Pattern

For most of the run:

- train loss remained near `6.235`
- eval loss remained near `5.545`
- neither loss showed meaningful improvement

These values closely match the random-classification baselines:

- train: `ln(512) = 6.238`
- eval: `ln(256) = 5.545`

This indicates that the model assigned approximately uniform probability to
the candidates and did not reliably separate positive pairs from negatives.

## Potential Cause

The current distributed-negative implementation gathers embeddings from all
GPUs so that each local batch uses the full global batch as candidates.

For each GPU:

- local queries: `128`
- global candidates: `512`
- contrastive matrix: `128 x 512`

The local embeddings retain their computation graph, while embeddings gathered
from other GPUs are detached from their original computation graphs on the
receiving GPU. Remote embeddings still participate in the softmax and affect
the gradient of local queries, but they do not receive candidate-side gradients
from that GPU's loss.

This behavior is not yet confirmed as the root cause. It is recorded as a
priority hypothesis because the single-GPU run learned successfully while the
multi-GPU run remained almost exactly at the global random-loss baseline.

## Attempt Direction 1: Disable Cross-GPU Negatives

Run a controlled four-GPU experiment without gathering embeddings for the
contrastive loss.

Expected behavior:

- each GPU computes a local `128 x 128` contrastive matrix
- each sample has `127` local negatives
- each GPU computes its own local loss and parameter gradients
- DDP averages the model-parameter gradients across the four GPUs
- the random train-loss baseline becomes `ln(128) = 4.852`

Keep all other settings unchanged so that distributed-negative gathering is
the only experimental variable.

Interpretation:

- if loss falls below `4.852`, while the gathered version remains near
  `6.238`, the cross-GPU negative path is strongly implicated
- if loss remains near `4.852`, investigate embedding collapse, ineffective
  parameter updates, data pairing, and optimizer behavior instead

Useful diagnostics for both variants:

- `logit_scale.exp()`
- mean positive-pair cosine similarity
- mean negative-pair cosine similarity
- sign/text embedding standard deviation
- gradient norm

### Direction 1 H100 Smoke Configuration

The queued H100 diagnostic smoke run uses:

- `4 x H100`
- per-device batch size: `128`
- per-device eval batch size: `64`
- local contrastive matrix per rank: `128 x 128`
- global samples processed per optimizer step: `512`
- cross-GPU negatives: disabled
- maximum training steps: `300`
- warmup steps: `10`
- training-loss logging interval: `10` steps
- evaluation interval: `50` steps
- checkpoint saving: disabled
- dataloader workers: `1` per rank, `4` total

Primary checks:

- train loss should begin near `ln(128) = 4.852`, not `ln(512) = 6.238`
- eval loss should begin near `ln(64) = 4.159`, not `ln(256) = 5.545`
- gradient norm should remain meaningfully non-zero
- train and eval loss should start moving below their local random baselines

### Direction 1 V100 Smoke Configuration

The separately named low-priority V100 run uses:

- `4 x V100`
- per-device train and eval batch size: `32`
- local contrastive matrix per rank: `32 x 32`
- global samples processed per optimizer step: `128`
- dataloader workers: `1` per rank, `4` total
- maximum training steps: `300`
- training-loss logging interval: `10` steps
- evaluation interval: `50` steps
- checkpoint saving: disabled

Its train and eval random-loss baseline is `ln(32) = 3.466`.

### Direction 1 Low-Priority H100 Smoke Configuration

To reproduce the original H100 batch setting without waiting for the standard
partition, a separately named low-priority run uses:

- `4 x H100` on `lowprio`
- CPU memory request: `80G`
- time limit: `2 hours`
- per-device train batch size: `128`
- per-device eval batch size: `64`
- dataloader workers: `1` per rank, `4` total
- all training and evaluation intervals identical to the standard H100 smoke

The output directory, W&B run name, Slurm job name, and log files are separate
from both the queued standard H100 run and the completed V100 run.

### Direction 1 V100 Batch-128 Run

An additional V100 run tests whether the local-negative result holds at the
same per-device training batch size as the H100 experiment:

- `4 x V100` on `lowprio`
- CPU memory request: `64G`
- time limit: `3 hours`
- per-device train batch size: `128`
- per-device eval batch size: `64`
- global samples processed per optimizer step: `512`
- dataloader workers: `2` per rank, `8` total
- maximum training steps: `1000`
- training-loss logging interval: `10` steps
- evaluation interval: `50` steps
- checkpoint saving: disabled

Expected random-loss baselines:

- train: `ln(128) = 4.852`
- eval: `ln(64) = 4.159`

This run may exceed V100 device memory. An early CUDA OOM would establish that
the batch-128 comparison requires a higher-memory GPU or a different memory
strategy.
