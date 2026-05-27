# YouTube-SL Pretrain Experiment Record

## Current Scope

This note records the currently confirmed data status for `YouTube-SL-25` before any preprocessing or training setup work.

## Dataset Root

- Shared root:
  - `/shares/iict-sp2.ebling.cl.uzh/common/YouTube-SL-25`

## Top-Level Structure

- `ase/`
- `non-ase/`

Observed sizes:

- `ase`: about `193G`
- `non-ase`: about `698G`

## ASE Part

Confirmed structure:

- `ase/downloads/<video_id>/`

Observed files inside each video directory:

- `<video_id>.mp4`
- `<video_id>.mp3`
- `<video_id> (en).srt`
- `<video_id> (en).json`

Confirmed metadata content from subtitle files:

- segment-level timestamps
- English text

Current conclusion:

- `ase` currently provides `video + audio + aligned English text`
- `ase` does **not** currently provide `.pose` files
- `ase` is therefore **not yet** in the `pose2text` format required by the current SignCLIP training pipeline

## Non-ASE Part

Confirmed structure:

- `non-ase/downloads/<video_id>/`

Observed files inside each video directory:

- `<video_id>.mp4`
- `<video_id>.mp3`

Current conclusion:

- `non-ase` currently provides only raw audio/video assets
- no subtitle metadata has been confirmed
- no `.pose` files have been confirmed
- `non-ase` is further away from the current `pose2text` training format than `ase`

## Training Readiness

Current SignCLIP pretraining in this repository expects:

- `.pose` inputs
- segment-level aligned text
- final TSVs in the `pose2text`-style format

`YouTube-SL-25` is therefore **not yet ready** for direct SignCLIP pretraining.

## Immediate Implication

The next stage for `YouTube-SL-25` is **data asset preparation**, not training configuration.

Priority:

1. `ase`: convert `video + subtitle timing` into `pose + text`
2. `non-ase`: first identify or recover text metadata, then consider pose extraction

