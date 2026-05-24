#!/usr/bin/env python3
"""
Inspect SignCLIP pose sequence lengths under the current processor settings.

This script is intended to answer questions like:
  - How many samples would exceed `max_frames=256`?
  - How many samples sit right at the frame cap?
  - What does the post-processor frame-length distribution look like?

Important note:
  In the current Pose2Text pipeline, train/validation samples whose pose length
  exceeds `max_frames` are filtered out during dataset preparation rather than
  truncated at collation time. This script therefore reports:

  1. Samples with `num_frames > max_frames`  -> would be filtered out
  2. Samples with `num_frames == max_frames` -> kept, but the frame cap is tight
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import statistics
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from multimodalhugs.processors.pose_modality_processor import PoseModalityProcessor


SPLITS = ("train", "validation", "test")
EXPECTED_COLUMNS = [
    "signal",
    "signal_start",
    "signal_end",
    "encoder_prompt",
    "decoder_prompt",
    "output",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze SignCLIP pose-frame lengths for TSV splits."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing train.tsv / validation.tsv / test.tsv.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=256,
        help="Frame cap used by the dataset/model setup.",
    )
    parser.add_argument(
        "--sign-max-position-embeddings",
        type=int,
        default=258,
        help="Position embedding cap used by the sign encoder.",
    )
    parser.add_argument(
        "--special-tokens",
        type=int,
        default=2,
        help="Number of extra sign tokens added by the model (CLS + SEP = 2).",
    )
    parser.add_argument(
        "--skip-frames-stride",
        type=int,
        default=None,
        help="Optional frame skipping stride to mirror processor behavior.",
    )
    parser.add_argument(
        "--no-reduce-holistic-poses",
        action="store_true",
        help="Disable reduce_holistic processing when computing lengths.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Optional number of rows to sample per split for a faster estimate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Sampling seed used when --sample-size is provided.",
    )
    parser.add_argument(
        "--max-failure-examples",
        type=int,
        default=10,
        help="Maximum number of failure examples to print per split.",
    )
    return parser.parse_args()


def percentile(values: Sequence[float], p: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    pos = (len(values) - 1) * p
    lower = math.floor(pos)
    upper = math.ceil(pos)
    if lower == upper:
        return float(values[lower])
    weight = pos - lower
    return float(values[lower] * (1 - weight) + values[upper] * weight)


def load_rows(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fieldnames = reader.fieldnames or []
        missing = [column for column in EXPECTED_COLUMNS if column not in fieldnames]
        if missing:
            raise ValueError(f"{path} is missing expected columns: {missing}")
        return list(reader)


def maybe_sample_rows(rows: List[dict], sample_size: int | None, seed: int) -> List[dict]:
    if sample_size is None or sample_size >= len(rows):
        return rows
    rng = random.Random(seed)
    return rng.sample(rows, sample_size)


def summarize_lengths(
    rows: Iterable[dict],
    *,
    processor: PoseModalityProcessor,
    max_frames: int,
    sign_max_position_embeddings: int,
    special_tokens: int,
    max_failure_examples: int,
) -> Dict[str, object]:
    frame_lengths: List[int] = []
    failed_rows = 0
    failure_examples: List[dict] = []
    failure_counter: Counter[str] = Counter()

    for row in rows:
        try:
            tensor = processor.process_sample(
                {
                    "signal": row["signal"],
                    "signal_start": int(float(row.get("signal_start") or 0)),
                    "signal_end": int(float(row.get("signal_end") or 0)),
                }
            )
            frame_lengths.append(int(tensor.size(0)))
        except Exception as exc:
            failed_rows += 1
            error_type = type(exc).__name__
            failure_counter[error_type] += 1
            if len(failure_examples) < max_failure_examples:
                failure_examples.append(
                    {
                        "signal": row["signal"],
                        "signal_start": row.get("signal_start"),
                        "signal_end": row.get("signal_end"),
                        "error_type": error_type,
                        "error_message": str(exc),
                    }
                )

    frame_lengths_sorted = sorted(frame_lengths)
    num_rows = len(frame_lengths)
    over_max = sum(length > max_frames for length in frame_lengths)
    at_max = sum(length == max_frames for length in frame_lengths)
    at_or_over_max = sum(length >= max_frames for length in frame_lengths)
    over_position_limit = sum(
        (length + special_tokens) > sign_max_position_embeddings for length in frame_lengths
    )
    at_position_limit = sum(
        (length + special_tokens) == sign_max_position_embeddings for length in frame_lengths
    )

    return {
        "num_rows": num_rows,
        "failed_rows": failed_rows,
        "frame_lengths": {
            "min": min(frame_lengths_sorted) if frame_lengths_sorted else 0,
            "p50": percentile(frame_lengths_sorted, 0.5),
            "p90": percentile(frame_lengths_sorted, 0.9),
            "p95": percentile(frame_lengths_sorted, 0.95),
            "max": max(frame_lengths_sorted) if frame_lengths_sorted else 0,
            "mean": statistics.mean(frame_lengths_sorted) if frame_lengths_sorted else 0.0,
        },
        "over_max_frames": over_max,
        "at_max_frames": at_max,
        "at_or_over_max_frames": at_or_over_max,
        "over_position_limit": over_position_limit,
        "at_position_limit": at_position_limit,
        "failure_types": failure_counter,
        "failure_examples": failure_examples,
    }


def ratio_string(count: int, total: int) -> str:
    if total == 0:
        return "0/0 (0.00%)"
    return f"{count}/{total} ({100.0 * count / total:.2f}%)"


def format_summary(
    split: str,
    *,
    total_rows_before_sampling: int,
    sampled_rows: int,
    summary: Dict[str, object],
    max_frames: int,
    sign_max_position_embeddings: int,
    special_tokens: int,
) -> str:
    lengths = summary["frame_lengths"]
    num_rows = summary["num_rows"]
    lines = [
        f"[{split}]",
        f"total_rows_before_sampling={total_rows_before_sampling}",
        f"sampled_rows={sampled_rows}",
        f"analyzed_rows={num_rows}",
        f"failed_rows={summary['failed_rows']}",
        (
            "frame_lengths="
            f"min:{lengths['min']} p50:{lengths['p50']:.1f} p90:{lengths['p90']:.1f} "
            f"p95:{lengths['p95']:.1f} max:{lengths['max']} mean:{lengths['mean']:.1f}"
        ),
        (
            f"would_be_filtered_by_max_frames_{max_frames}="
            f"{ratio_string(summary['over_max_frames'], num_rows)}"
        ),
        (
            f"exactly_at_max_frames_{max_frames}="
            f"{ratio_string(summary['at_max_frames'], num_rows)}"
        ),
        (
            f"at_or_over_max_frames_{max_frames}="
            f"{ratio_string(summary['at_or_over_max_frames'], num_rows)}"
        ),
        (
            f"would_exceed_position_embeddings_{sign_max_position_embeddings}"
            f"_with_{special_tokens}_special_tokens="
            f"{ratio_string(summary['over_position_limit'], num_rows)}"
        ),
        (
            f"exactly_at_position_embeddings_{sign_max_position_embeddings}"
            f"_with_{special_tokens}_special_tokens="
            f"{ratio_string(summary['at_position_limit'], num_rows)}"
        ),
    ]
    failure_types = dict(summary["failure_types"])
    if failure_types:
        lines.append(f"failure_types={failure_types}")
    failure_examples = summary["failure_examples"]
    if failure_examples:
        lines.append("failure_examples=")
        for example in failure_examples:
            lines.append(
                "  - "
                f"{example['error_type']}: signal={example['signal']} "
                f"start={example['signal_start']} end={example['signal_end']} "
                f"message={example['error_message']}"
            )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()

    processor = PoseModalityProcessor(
        reduce_holistic_poses=not args.no_reduce_holistic_poses,
        skip_frames_stride=args.skip_frames_stride,
    )

    for split in SPLITS:
        path = input_dir / f"{split}.tsv"
        rows = load_rows(path)
        sampled_rows = maybe_sample_rows(rows, args.sample_size, args.seed)
        summary = summarize_lengths(
            sampled_rows,
            processor=processor,
            max_frames=args.max_frames,
            sign_max_position_embeddings=args.sign_max_position_embeddings,
            special_tokens=args.special_tokens,
            max_failure_examples=args.max_failure_examples,
        )
        print(
            format_summary(
                split,
                total_rows_before_sampling=len(rows),
                sampled_rows=len(sampled_rows),
                summary=summary,
                max_frames=args.max_frames,
                sign_max_position_embeddings=args.sign_max_position_embeddings,
                special_tokens=args.special_tokens,
            )
        )
        print()


if __name__ == "__main__":
    main()
