# PopSign Pretrain Experiment Record

## Scope

This document records the current PopSign SignCLIP pretraining run on the
server branch workflow. The goal was to verify that the migrated Hugging Face
style SignCLIP pipeline can be set up, pretrained, and evaluated on PopSign,
and to compare the resulting retrieval metrics against the reported paper
numbers.

## Dataset and Setup

- Dataset root:
  `/shares/iict-sp2.ebling.cl.uzh/common/popsign_v1_0/game`
- Metadata root:
  `/shares/iict-sp2.ebling.cl.uzh/common/popsign_v1_0/game/metadata`
- Text format:
  - `encoder_prompt = <en> <ase>`
  - `output = label`
- Evaluation direction:
  - `v2t` only

## Config Files

- `configs/signclip_setup_popsign.server.yaml`
- `configs/signclip_pretrain_popsign.server.yaml`
- `configs/signclip_eval_popsign_initial.server.yaml`
- `configs/signclip_eval_popsign_final.server.yaml`

## Training Settings

- Train batch size: `256`
- Eval batch size: `128`
- Learning rate: `5e-5`
- Weight decay: `0.0`
- Epochs: `25`
- Warmup steps: `1000`
- Train ordering: `default`
- W&B project: `multimodalhugs-signclip-popsign`

## Initial Evaluation

Before pretraining, the setup model was evaluated on the PopSign validation
split.

| Metric | Initial |
|---|---:|
| eval_loss | 4.85383 |
| v2t_r@1 | 0.00132 |
| v2t_r@5 | 0.01338 |
| v2t_r@10 | 0.03205 |
| v2t_median_r | 127.0 |
| v2t_mean_r | 127.70353 |

## Final Evaluation

After full pretraining, the final model was evaluated again on the same
validation split.

| Metric | Final |
|---|---:|
| eval_loss | 4.64087 |
| v2t_r@1 | 0.87695 |
| v2t_r@5 | 0.96938 |
| v2t_r@10 | 0.97910 |
| v2t_median_r | 1.0 |
| v2t_mean_r | 2.51734 |

## Comparison to Reported Paper Numbers

The current run is already in the same range as the paper-side reference shown
during manual review.

| Metric | Current | Paper |
|---|---:|---:|
| v2t_r@1 | 0.877 | 0.83 |
| v2t_r@5 | 0.969 | 0.97 |
| v2t_r@10 | 0.979 | 0.99 |
| v2t_median_r | 1 | 1 |

## Practical Takeaway

The current result looks broadly successful:

- The untrained model starts near random retrieval behavior.
- Full pretraining dramatically improves `v2t` retrieval quality.
- Final `MedianR = 1` matches the paper-level target.
- `R@1` is already slightly above the reference number, while `R@5` and
  `R@10` are very close.

## Current Engineering Caveats

Two implementation details were adjusted in a pragmatic way to get the current
pipeline running and should be revisited later for a cleaner reproduction.

### 1. Sign position embedding limit

The sign-side maximum position embedding:  bumped from the
original `256` boundary to `258`.

Reason:

- the current sign tower adds special tokens around the pose sequence

Current config:

- `sign_max_position_embeddings: 258`
  in `configs/signclip_setup_popsign.server.yaml`

### 2. Pose feature dimension

The current MMH PopSign processor emits `534`-dimensional pose tokens, so the
setup was adjusted to use `534` instead of the earlier fingerclip
`609`-dimensional assumption.

Current config:

- `sign_input_dim: 534`
  in `configs/signclip_setup_popsign.server.yaml`

