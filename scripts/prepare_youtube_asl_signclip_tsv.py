#!/usr/bin/env python3
"""
Convert YouTube-ASL metadata TSV files into SignCLIP-compatible
train/validation/test TSVs.

Supported input schemas:

1. Original root metadata.tsv
   file  offset  duration  utf8  mp4_full_duration

2. Downloads metadata.tsv
   source_signal  source_start  source_end  input_text  source_prompt
   generation_prompt  output_text

Generated output columns:
    signal  signal_start  signal_end  encoder_prompt  decoder_prompt  output

Transformations:
    - replace the original gsantm-local prefix with the shared downloads prefix
    - replace `.mp4` file suffixes with `.pose`

Split strategy:
    - group by signal path
    - shuffle groups with a deterministic seed
    - assign groups to train / validation / test according to provided ratios
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


ROOT_METADATA_COLUMNS = [
    "file",
    "offset",
    "duration",
    "utf8",
    "mp4_full_duration",
]

DOWNLOADS_METADATA_COLUMNS = [
    "source_signal",
    "source_start",
    "source_end",
    "input_text",
    "source_prompt",
    "generation_prompt",
    "output_text",
]

EXPECTED_OUTPUT_COLUMNS = [
    "signal",
    "signal_start",
    "signal_end",
    "encoder_prompt",
    "decoder_prompt",
    "output",
]

SPLIT_NAMES = ("train", "validation", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare SignCLIP TSV splits from YouTube-ASL metadata.tsv."
    )
    parser.add_argument(
        "--input-tsv",
        type=Path,
        required=True,
        help="Path to the original YouTube-ASL metadata.tsv file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where train/validation/test TSV files will be written.",
    )
    parser.add_argument(
        "--old-prefix",
        type=str,
        default="/home/gsantm/common/YouTube-ASL/downloads",
        help="Signal path prefix to replace.",
    )
    parser.add_argument(
        "--new-prefix",
        type=str,
        default="/shares/iict-sp2.ebling.cl.uzh/common/YouTube-ASL/downloads",
        help="Replacement signal path prefix.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.98,
        help="Fraction of signal-path groups assigned to train.",
    )
    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=0.01,
        help="Fraction of signal-path groups assigned to validation.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.01,
        help="Fraction of signal-path groups assigned to test.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for grouped split assignment.",
    )
    parser.add_argument(
        "--min-output-chars",
        type=int,
        default=1,
        help="Drop samples whose stripped output text is shorter than this many characters.",
    )
    parser.add_argument(
        "--max-duration-ms",
        type=float,
        default=None,
        help="Optional maximum segment duration in milliseconds. Longer samples are dropped.",
    )
    return parser.parse_args()


def detect_schema(fieldnames: Sequence[str]) -> str:
    fields = set(fieldnames)
    if set(ROOT_METADATA_COLUMNS).issubset(fields):
        return "root"
    if set(DOWNLOADS_METADATA_COLUMNS).issubset(fields):
        return "downloads"
    raise ValueError(
        "Input TSV does not match a supported YouTube-ASL schema. "
        f"Found columns: {list(fieldnames)}"
    )


def load_rows(path: Path) -> tuple[List[dict], str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fieldnames = reader.fieldnames or []
        schema = detect_schema(fieldnames)
        return list(reader), schema


def remap_signal_path(signal: str, old_prefix: str, new_prefix: str) -> str:
    signal = (signal or "").strip()
    if signal.startswith(old_prefix):
        signal = new_prefix + signal[len(old_prefix):]
    if signal.endswith(".mp4"):
        signal = signal[:-4] + ".pose"
    return signal


def compute_end_from_start_and_duration(start_value: str, duration_value: str) -> int:
    start = int(float(start_value or 0))
    duration = int(float(duration_value or 0))
    return start + duration


def to_signclip_row(row: dict, old_prefix: str, new_prefix: str, schema: str) -> dict:
    if schema == "root":
        signal = remap_signal_path(row["file"], old_prefix, new_prefix)
        signal_start = row["offset"]
        signal_end = compute_end_from_start_and_duration(row["offset"], row["duration"])
        encoder_prompt = ""
        decoder_prompt = ""
        output = row.get("utf8", "") or ""
    elif schema == "downloads":
        signal = remap_signal_path(row["source_signal"], old_prefix, new_prefix)
        signal_start = row["source_start"]
        signal_end = compute_end_from_start_and_duration(row["source_start"], row["source_end"])
        encoder_prompt = row.get("source_prompt", "") or ""
        decoder_prompt = row.get("generation_prompt", "") or ""
        output = row.get("output_text", "") or ""
    else:
        raise ValueError(f"Unsupported schema: {schema}")

    return {
        "signal": signal,
        "signal_start": signal_start,
        "signal_end": signal_end,
        "encoder_prompt": encoder_prompt,
        "decoder_prompt": decoder_prompt,
        "output": output,
    }


def should_keep_row(
    row: dict,
    *,
    schema: str,
    min_output_chars: int,
    max_duration_ms: float | None,
) -> bool:
    if schema == "root":
        output = (row.get("utf8", "") or "").strip()
        duration = float(row["duration"])
    elif schema == "downloads":
        output = (row.get("output_text", "") or "").strip()
        duration = float(row["source_end"])
    else:
        raise ValueError(f"Unsupported schema: {schema}")

    if len(output) < min_output_chars:
        return False
    if max_duration_ms is not None and duration > max_duration_ms:
        return False
    return True


def grouped_split_keys(
    keys: Sequence[str],
    *,
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
    seed: int,
) -> Dict[str, List[str]]:
    ratio_sum = train_ratio + validation_ratio + test_ratio
    if not math.isclose(ratio_sum, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError(f"Split ratios must sum to 1.0, got {ratio_sum}.")

    shuffled = list(keys)
    random.Random(seed).shuffle(shuffled)

    num_groups = len(shuffled)
    if num_groups == 0:
        return {split: [] for split in SPLIT_NAMES}

    num_train = int(num_groups * train_ratio)
    num_validation = int(num_groups * validation_ratio)
    num_test = num_groups - num_train - num_validation

    if num_groups >= 3:
        if num_validation == 0:
            num_validation = 1
            num_train -= 1
        if num_test == 0:
            num_test = 1
            num_train -= 1

    if num_train < 0:
        raise ValueError(
            f"Invalid split counts after enforcing non-empty validation/test: "
            f"train={num_train}, validation={num_validation}, test={num_test}"
        )

    train_keys = shuffled[:num_train]
    validation_keys = shuffled[num_train : num_train + num_validation]
    test_keys = shuffled[num_train + num_validation :]

    return {
        "train": train_keys,
        "validation": validation_keys,
        "test": test_keys,
    }


def write_tsv(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=EXPECTED_OUTPUT_COLUMNS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    input_tsv = args.input_tsv.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    raw_rows, schema = load_rows(input_tsv)
    filtered_rows = [
        row
        for row in raw_rows
        if should_keep_row(
            row,
            schema=schema,
            min_output_chars=args.min_output_chars,
            max_duration_ms=args.max_duration_ms,
        )
    ]

    groups: "OrderedDict[str, List[dict]]" = OrderedDict()
    for row in filtered_rows:
        if schema == "root":
            raw_signal = row["file"]
        else:
            raw_signal = row["source_signal"]
        signal = remap_signal_path(raw_signal, args.old_prefix, args.new_prefix)
        groups.setdefault(signal, []).append(row)

    split_keys = grouped_split_keys(
        list(groups.keys()),
        train_ratio=args.train_ratio,
        validation_ratio=args.validation_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    split_rows: Dict[str, List[dict]] = defaultdict(list)
    for split_name, keys in split_keys.items():
        for key in keys:
            split_rows[split_name].extend(
                to_signclip_row(row, args.old_prefix, args.new_prefix, schema) for row in groups[key]
            )

    for split_name in SPLIT_NAMES:
        rows = split_rows[split_name]
        write_tsv(output_dir / f"{split_name}.tsv", rows)
        print(f"[{split_name}] groups={len(split_keys[split_name])} rows={len(rows)}")
    print(
        f"[summary] input_rows={len(raw_rows)} kept_rows={len(filtered_rows)} "
        f"dropped_rows={len(raw_rows) - len(filtered_rows)}"
    )


if __name__ == "__main__":
    main()
