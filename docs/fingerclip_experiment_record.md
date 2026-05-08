# Fingerclip Experiment Record

## Scope

This document records the `fingerclip` branch experiments used to validate the
Hugging Face style SignCLIP migration inside MultimodalHugs.

The goal of these runs was not only to confirm that the migrated pipeline can
train end-to-end, but also to compare a fairseq-inspired train ordering strategy
against the default Hugging Face style ordering on the same hand-only
fingerclip data.

## Dataset and Setup

### Data variant

- Dataset variant: `fingerclip_hand_fairseqish`
- Pose source: hand-only landmarks
- Text backbone: `bert-base-uncased`
- Sign input dimension: `126`
- Number of grouping labels for round-robin ordering: `35`

### Splits

- Train: `1397` samples
- Validation: `151` samples
- Test: `153` samples

### Shared training/eval settings

- Train batch size: `35`
- Eval batch size: `2`
- Learning rate: `5e-5`
- Weight decay: `0.0`
- Seed: `42`
- Save strategy: `no` for the final 25-epoch A/B runs

## Config Files

The following configs correspond to the recorded experiments and are intended to
be kept alongside this report:

- `configs/sign_clip_setup_hand_fairseqish.local.yaml`
- `configs/sign_clip_train_hand_fairseqish_eval.local.yaml`
- `configs/sign_clip_train_hand_fairseqish_ab_default.local.yaml`
- `configs/sign_clip_train_hand_fairseqish_ab_round_robin.local.yaml`
- `configs/sign_clip_train_hand_fairseqish_ab_default_25ep_eval.local.yaml`
- `configs/sign_clip_train_hand_fairseqish_ab_round_robin_25ep_eval.local.yaml`

## W&B Runs

Main 25-epoch runs:

- A / default ordering:
  [sign_clip_hand_fairseqish_ab_default_25ep](https://wandb.ai/xf-uzh-university-of-z-rich/multimodalhugs-signclip/runs/d7o0at1j)
- B / fairseq round robin:
  [sign_clip_hand_fairseqish_ab_round_robin_25ep](https://wandb.ai/xf-uzh-university-of-z-rich/multimodalhugs-signclip/runs/6h89wn76)

## Initial Retrieval Baseline

The untrained setup model was evaluated before any A/B training.

| Metric | Initial |
|---|---:|
| eval_loss | 0.68653 |
| t2v_p@1 | 0.00000 |
| t2v_p@5 | 0.04706 |
| t2v_p@10 | 0.03529 |
| t2v_r@1 | 0.00000 |
| t2v_r@5 | 0.05298 |
| t2v_r@10 | 0.07947 |
| t2v_median_r | 27.50000 |
| t2v_mean_r | 39.85294 |
| v2t_r@1 | 0.03311 |
| v2t_r@5 | 0.17219 |
| v2t_r@10 | 0.31126 |
| v2t_median_r | 16.00000 |
| v2t_mean_r | 16.37086 |

## Pilot Result: 10-Epoch A/B

The 10-epoch full-dataset A/B was used as a pilot before the longer 25-epoch
comparison.

| Metric | A Final (10ep) | B Final (10ep) |
|---|---:|---:|
| eval_loss | 0.01535 | 0.01592 |
| t2v_p@1 | 0.94118 | 0.97059 |
| t2v_p@5 | 0.72353 | 0.72353 |
| t2v_p@10 | 0.41765 | 0.41765 |
| t2v_r@1 | 0.21192 | 0.21854 |
| t2v_r@5 | 0.81457 | 0.81457 |
| t2v_r@10 | 0.94040 | 0.94040 |
| t2v_median_r | 1.00000 | 1.00000 |
| t2v_mean_r | 1.14706 | 1.11765 |
| v2t_r@1 | 0.88742 | 0.87417 |
| v2t_r@5 | 1.00000 | 1.00000 |
| v2t_r@10 | 1.00000 | 1.00000 |
| v2t_median_r | 1.00000 | 1.00000 |
| v2t_mean_r | 1.21192 | 1.23841 |

Pilot takeaway:

- Fairseq-style round-robin ordering accelerated early training.
- Final 10-epoch results were already very close between A and B.

## Main Result: 25-Epoch A/B

### Experiment definitions

- **A**: default train ordering
- **B**: fairseq-style `round_robin` train ordering grouped by `output`

### Per-epoch losses

| Epoch | A train | A eval | B train | B eval |
|---|---:|---:|---:|---:|
| 1 | 3.1918 | 0.29679 | 2.9579 | 0.15703 |
| 2 | 1.9495 | 0.08269 | 1.4040 | 0.05565 |
| 3 | 1.3804 | 0.04462 | 0.8306 | 0.02553 |
| 4 | 1.1103 | 0.05943 | 0.5964 | 0.02746 |
| 5 | 0.9975 | 0.03078 | 0.5149 | 0.02578 |
| 6 | 0.9472 | 0.02980 | 0.4272 | 0.01556 |
| 7 | 0.9386 | 0.03313 | 0.3522 | 0.01470 |
| 8 | 0.8854 | 0.01713 | 0.3406 | 0.01941 |
| 9 | 0.8525 | 0.01689 | 0.2875 | 0.01528 |
| 10 | 0.7977 | 0.01915 | 0.2452 | 0.01623 |
| 11 | 0.8025 | 0.01901 | 0.2409 | 0.01563 |
| 12 | 0.7771 | 0.02103 | 0.2307 | 0.01473 |
| 13 | 0.7737 | 0.01291 | 0.1808 | 0.01670 |
| 14 | 0.7544 | 0.01305 | 0.1730 | 0.01185 |
| 15 | 0.7213 | 0.01847 | 0.1515 | 0.01546 |
| 16 | 0.7211 | 0.01081 | 0.1388 | 0.01515 |
| 17 | 0.6747 | 0.01199 | 0.1231 | 0.01348 |
| 18 | 0.6824 | 0.01073 | 0.1022 | 0.01716 |
| 19 | 0.6618 | 0.01149 | 0.0923 | 0.01736 |
| 20 | 0.6720 | 0.01215 | 0.0902 | 0.01425 |
| 21 | 0.6424 | 0.01178 | 0.1002 | 0.01547 |
| 22 | 0.6348 | 0.01202 | 0.0736 | 0.01467 |
| 23 | 0.6628 | 0.01120 | 0.0607 | 0.01447 |
| 24 | 0.6544 | 0.01103 | 0.0656 | 0.01413 |
| 25 | 0.6297 | 0.01133 | 0.0478 | 0.01516 |

### Final retrieval metrics

| Metric | Initial | A Final (25ep) | B Final (25ep) |
|---|---:|---:|---:|
| eval_loss | 0.68653 | 0.01133 | 0.01516 |
| t2v_p@1 | 0.00000 | 1.00000 | 0.97059 |
| t2v_p@5 | 0.04706 | 0.77647 | 0.78235 |
| t2v_p@10 | 0.03529 | 0.44118 | 0.43529 |
| t2v_r@1 | 0.00000 | 0.22517 | 0.21854 |
| t2v_r@5 | 0.05298 | 0.87417 | 0.88079 |
| t2v_r@10 | 0.07947 | 0.99338 | 0.98013 |
| t2v_median_r | 27.50000 | 1.00000 | 1.00000 |
| t2v_mean_r | 39.85294 | 1.00000 | 1.02941 |
| v2t_r@1 | 0.03311 | 0.94702 | 0.94040 |
| v2t_r@5 | 0.17219 | 1.00000 | 0.98675 |
| v2t_r@10 | 0.31126 | 1.00000 | 1.00000 |
| v2t_median_r | 16.00000 | 1.00000 | 1.00000 |
| v2t_mean_r | 16.37086 | 1.09934 | 1.14570 |

## Interpretation

### What was learned

1. The migrated SignCLIP pipeline trains stably end-to-end on fingerclip.
2. Round-robin ordering clearly accelerates optimization in the
   early and middle training stages.
3. After a longer 25-epoch run, the default ordering slightly outperformed the
   round-robin ordering on the final retrieval metrics that matter most in this
   setup. (overfitted)

### Practical conclusion

For the current fingerclip hand-only fairseqish setup:

- `round_robin` is useful as a convergence-speed manipulation.
- `default` ordering produced the strongest final 25-epoch result overall.



## Recommended next usage

If this branch is used for follow-up reporting or comparison:

- Treat the 25-epoch A/B as the primary experiment record.
- Treat the 10-epoch A/B as a pilot sanity-check result.
- Use the 25-epoch W&B runs for supervisor-facing visualization.
