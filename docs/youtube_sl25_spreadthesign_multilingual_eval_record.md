# YouTube-SL-25 -> SpreadTheSign Multilingual Evaluation Record

## Experiment

- Date: 2026-09-27 (UZH local time)
- Task: zero-shot sign-to-text retrieval (`v2t`)
- Model: YouTube-SL-25 clean max256, softmax/CLIP, batch 128, learning rate `5e-5`
- Checkpoint: `/home/faxu/scratch/signclip/runs/youtube_sl25_clean_max256_softmax_b128_130k/train/checkpoint-36000`
- Processor: `/home/faxu/scratch/signclip/setup/youtube_sl25_clean_max256_v1/setup/sign_clip_processor`
- Code commit: `13a6343`

Each text/sign-language pair was evaluated independently. Text candidates were
deduplicated within the pair using the complete `<text_language>
<sign_language> text` string. Retrieval scores were computed in chunks rather
than by constructing one combined multilingual retrieval matrix.

## Data preparation

| Item | Count |
| --- | ---: |
| Selected rows | 95,963 |
| Kept rows | 95,535 |
| Rejected rows | 428 |
| Missing pose files | 6 |
| Poses longer than 256 frames | 422 |

All selected pose files were fully parsed before evaluation. No unreadable
pose remained in the evaluation metadata.

## Results

| Pair | Samples | Text candidates | R@1 | R@5 | R@10 | MedianR | MeanR | Loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `<en> <ase>` | 12,383 | 11,093 | 2.544% | 6.137% | 8.689% | 675 | 1,683.54 | 8.359 |
| `<en> <ins>` | 6,052 | 5,318 | 1.421% | 3.652% | 5.701% | 699 | 1,154.22 | 8.137 |
| `<pl> <pso>` | 13,795 | 12,659 | 0.188% | 0.805% | 1.298% | 4,019 | 4,581.99 | 10.598 |
| `<de> <gsg>` | 18,070 | 15,939 | 0.277% | 0.747% | 1.157% | 4,682 | 5,505.18 | 10.202 |
| `<en> <bfi>` | 15,951 | 15,006 | 0.301% | 0.972% | 1.624% | 2,491 | 3,787.96 | 9.614 |
| `<it> <ise>` | 19,571 | 17,132 | 0.194% | 0.720% | 1.175% | 4,531 | 5,572.08 | 9.823 |
| `<ja> <jsl>` | 9,713 | 8,929 | 0.175% | 0.474% | 0.947% | 2,845 | 3,139.64 | 9.473 |
| **Macro** | - | - | **0.729%** | **1.930%** | **2.941%** | **2,848.9** | **3,632.09** | **9.458** |
| **Sample-weighted** | 95,535 | - | **0.607%** | **1.642%** | **2.502%** | - | **4,087.38** | **9.640** |

Macro is the unweighted mean of the seven pair-level metrics, so each language
pair contributes equally. Sample-weighted metrics weight each pair by its
number of evaluation samples. A weighted MedianR is intentionally omitted
because averaging pair-level medians does not produce the pooled median.

The candidate pools contain 5,318--17,132 unique texts, so these R@K values
must not be compared directly with PopSign retrieval over only 250 classes.

## Execution

| Job | Purpose | State | Exit code | Runtime |
| --- | --- | --- | --- | ---: |
| `6567063` | Full metadata validation | `COMPLETED` | `0:0` | 00:25:05 |
| `6567064` | Seven-pair evaluation | `COMPLETED` | `0:0` | 00:29:44 |

Only non-fatal `pixi.lock` format and oneDNN informational warnings appeared
in stderr. No traceback, CUDA error, or unreadable evaluation sample occurred.

## Artifacts

```text
/home/faxu/scratch/signclip/metadata/spreadthesign_multilingual_youtube_sl25/summary.json
/home/faxu/scratch/signclip/evals/spreadthesign_multilingual_zeroshot_youtube_sl25_checkpoint36000/summary.json
/home/faxu/scratch/signclip/evals/spreadthesign_multilingual_zeroshot_youtube_sl25_checkpoint36000/summary.tsv
```
