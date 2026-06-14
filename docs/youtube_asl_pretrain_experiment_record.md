# YouTube-ASL Pretrain Experiment Record

## Scope

This document records the current exploratory setup work for SignCLIP
pretraining on `YouTube-ASL`.

At this stage, the focus is not yet on final training results. The main goal is
to understand the metadata structure, define a usable SignCLIP TSV conversion,
and measure the cleaned pose-length distribution before writing the first
server-side setup and pretraining configs.

## Data Sources

- Original metadata:
  `/shares/iict-sp2.ebling.cl.uzh/common/YouTube-ASL/metadata.tsv`
- Shared downloads root:
  `/shares/iict-sp2.ebling.cl.uzh/common/YouTube-ASL/downloads`
- Clean SignCLIP-ready split directory:
  `/shares/iict-sp2.ebling.cl.uzh/common/YouTube-ASL/metadata_signclip_clean_v2`

## Metadata Conversion Notes

The original root metadata uses columns:

- `file`
- `offset`
- `duration`
- `utf8`
- `mp4_full_duration`

The SignCLIP conversion script maps that structure into the current
`pose2text`-style TSV layout:

- `signal`
- `signal_start`
- `signal_end`
- `encoder_prompt`
- `decoder_prompt`
- `output`

Key normalization rules:

- replace the original user-local prefix
  `/home/gsantm/common/YouTube-ASL/downloads`
  with
  `/shares/iict-sp2.ebling.cl.uzh/common/YouTube-ASL/downloads`
- replace `.mp4` with `.pose`
- interpret the source metadata as:
  - `signal_start = offset`
  - `signal_end = offset + duration`

The last point is important: an earlier attempt incorrectly used
`signal_end = duration`, which caused many invalid negative-length pose reads.

## Cleaning and Split Policy

The current cleaned split set was generated with:

- grouped split by `signal`
- ratios:
  - train `98%`
  - validation `1%`
  - test `1%`
- minimum output characters:
  - `1`
- maximum segment duration:
  - `15000 ms`

## Clean Split Sizes

The current `metadata_signclip_clean_v2` split sizes are:

| Split | Rows |
|---|---:|
| train | 569,749 |
| validation | 4,859 |
| test | 5,842 |

## Pose Length Distribution

Pose lengths below were measured with the current SignCLIP pose processing path
(`reduce_holistic_poses=true`, no frame skipping), using the clean TSV splits.

### Train

- sample size analyzed: `4,951 / 5,000`
- failed rows in sample: `49`
- failure type:
  - `FileNotFoundError` only

Frame-length summary:

- min: `1`
- p50: `115`
- p90: `241`
- p95: `289`
- max: `449`
- mean: `132.6`

Frame-cap checks with `max_frames = 256` and `sign_max_position_embeddings = 258`:

- would be filtered by `max_frames=256`:
  - `397 / 4951 = 8.02%`
- exactly at `256` frames:
  - `7 / 4951 = 0.14%`
- would exceed position embeddings `258` with 2 special tokens:
  - `397 / 4951 = 8.02%`

### Validation

- sample size analyzed: `4,859 / 4,859`
- failed rows: `0`

Frame-length summary:

- min: `10`
- p50: `111`
- p90: `250`
- p95: `302`
- max: `450`
- mean: `133.0`

Frame-cap checks with `max_frames = 256` and `sign_max_position_embeddings = 258`:

- would be filtered by `max_frames=256`:
  - `449 / 4859 = 9.24%`
- exactly at `256` frames:
  - `6 / 4859 = 0.12%`
- would exceed position embeddings `258` with 2 special tokens:
  - `449 / 4859 = 9.24%`

### Test

- sample size analyzed: `5,000 / 5,000`
- failed rows: `0`

Frame-length summary:

- min: `10`
- p50: `94`
- p90: `229`
- p95: `279`
- max: `451`
- mean: `117.4`

Frame-cap checks with `max_frames = 256` and `sign_max_position_embeddings = 258`:

- would be filtered by `max_frames=256`:
  - `346 / 5000 = 6.92%`
- exactly at `256` frames:
  - `4 / 5000 = 0.08%`
- would exceed position embeddings `258` with 2 special tokens:
  - `346 / 5000 = 6.92%`

## Current Interpretation

Based on the current clean split statistics:

- `YouTube-ASL` looks like a viable large-scale sentence-level pretraining
  corpus for SignCLIP
- `max_frames = 256` looks acceptable as a first baseline
- around `7%` to `9%` of samples would be filtered by that frame cap, which is
  noticeable but not obviously too aggressive for a first experiment
- very few samples sit exactly at the `256` boundary, which suggests the frame
  cap is not heavily squeezing a large mass of borderline samples

## Current Caveat

There is still a small amount of data incompleteness in the shared pose assets.

Observed symptom:

- some metadata entries point to `.pose` files that do not exist under the
  shared downloads root

In the current train sample analysis:

- `49 / 5000` sampled rows failed with `FileNotFoundError`

This should be treated as a dataset integrity issue rather than a processor or
timing-field issue.

## Fixed Training-Loss Probe

The single-GPU runs show a train-loss drop approximately every `4024` steps,
which matches one epoch:

`515023 / 128 ~= 4024`

To distinguish a real epoch-boundary model change from changing training
batches, a diagnostic run evaluates one deterministic batch of `128` training
pairs throughout training.

The probe:

- selects the samples once with a fixed seed
- preserves their order and in-batch negative pairs
- pre-collates the batch before training
- switches the model to evaluation mode
- disables gradient computation and parameter updates
- records `fixed_train_loss` every `100` steps
- records an additional point at exact epoch boundaries

Interpretation:

- smooth fixed loss with stair-stepped ordinary train loss points to changing
  batches, negative composition, or train-loss statistics
- an epoch-boundary jump in both losses points to a real model or training-state
  change
- falling fixed loss with rising validation loss is direct evidence of
  training-set overfitting
