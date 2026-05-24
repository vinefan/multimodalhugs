#!/usr/bin/env python3
"""
Lightweight descriptive statistics for SignCLIP-style TSV splits.

Expected columns:
    signal  signal_start  signal_end  encoder_prompt  decoder_prompt  output
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List


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
    parser = argparse.ArgumentParser(description="Analyze SignCLIP TSV split files.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing train.tsv / validation.tsv / test.tsv.",
    )
    return parser.parse_args()


def percentile(values: List[float], p: float) -> float:
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
        missing = [column for column in EXPECTED_COLUMNS if column not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path} is missing expected columns: {missing}")
        return list(reader)


def summarize_rows(rows: Iterable[dict]) -> Dict[str, object]:
    rows = list(rows)
    signals = [row["signal"] for row in rows]
    outputs = [row["output"] for row in rows]
    durations = [float(row["signal_end"]) for row in rows]
    output_chars = [len(text) for text in outputs]
    output_words = [len(text.split()) for text in outputs]

    signal_counter = Counter(signals)
    output_counter = Counter(outputs)

    durations_sorted = sorted(durations)
    output_chars_sorted = sorted(output_chars)
    output_words_sorted = sorted(output_words)

    return {
        "num_rows": len(rows),
        "num_unique_signals": len(signal_counter),
        "num_unique_outputs": len(output_counter),
        "avg_segments_per_signal": len(rows) / len(signal_counter) if signal_counter else 0.0,
        "duration_ms": {
            "min": min(durations_sorted) if durations_sorted else 0.0,
            "p50": percentile(durations_sorted, 0.5),
            "p90": percentile(durations_sorted, 0.9),
            "p95": percentile(durations_sorted, 0.95),
            "max": max(durations_sorted) if durations_sorted else 0.0,
            "mean": statistics.mean(durations_sorted) if durations_sorted else 0.0,
        },
        "output_chars": {
            "min": min(output_chars_sorted) if output_chars_sorted else 0,
            "p50": percentile(output_chars_sorted, 0.5),
            "p90": percentile(output_chars_sorted, 0.9),
            "p95": percentile(output_chars_sorted, 0.95),
            "max": max(output_chars_sorted) if output_chars_sorted else 0,
            "mean": statistics.mean(output_chars_sorted) if output_chars_sorted else 0.0,
        },
        "output_words": {
            "min": min(output_words_sorted) if output_words_sorted else 0,
            "p50": percentile(output_words_sorted, 0.5),
            "p90": percentile(output_words_sorted, 0.9),
            "p95": percentile(output_words_sorted, 0.95),
            "max": max(output_words_sorted) if output_words_sorted else 0,
            "mean": statistics.mean(output_words_sorted) if output_words_sorted else 0.0,
        },
        "top_signals": signal_counter.most_common(10),
        "top_outputs": output_counter.most_common(10),
    }


def format_summary(name: str, summary: Dict[str, object]) -> str:
    d = summary["duration_ms"]
    c = summary["output_chars"]
    w = summary["output_words"]
    lines = [
        f"[{name}]",
        f"rows={summary['num_rows']}",
        f"unique_signals={summary['num_unique_signals']}",
        f"unique_outputs={summary['num_unique_outputs']}",
        f"avg_segments_per_signal={summary['avg_segments_per_signal']:.2f}",
        (
            "duration_ms="
            f"min:{d['min']:.1f} p50:{d['p50']:.1f} p90:{d['p90']:.1f} "
            f"p95:{d['p95']:.1f} max:{d['max']:.1f} mean:{d['mean']:.1f}"
        ),
        (
            "output_chars="
            f"min:{c['min']} p50:{c['p50']:.1f} p90:{c['p90']:.1f} "
            f"p95:{c['p95']:.1f} max:{c['max']} mean:{c['mean']:.1f}"
        ),
        (
            "output_words="
            f"min:{w['min']} p50:{w['p50']:.1f} p90:{w['p90']:.1f} "
            f"p95:{w['p95']:.1f} max:{w['max']} mean:{w['mean']:.1f}"
        ),
        f"top_signals={summary['top_signals']}",
        f"top_outputs={summary['top_outputs']}",
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    for split in SPLITS:
        path = input_dir / f"{split}.tsv"
        rows = load_rows(path)
        summary = summarize_rows(rows)
        print(format_summary(split, summary))
        print()


if __name__ == "__main__":
    main()
