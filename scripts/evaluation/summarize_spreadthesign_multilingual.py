#!/usr/bin/env python3
"""Aggregate pair-level SpreadTheSign retrieval metrics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


DEFAULT_PAIRS = ("en_ase", "en_ins", "pl_pso", "de_gsg", "en_bfi", "it_ise", "ja_jsl")
METRICS = (
    "eval_loss",
    "v2t_r@1",
    "v2t_r@5",
    "v2t_r@10",
    "v2t_median_r",
    "v2t_mean_r",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--pair", action="append", dest="pairs")
    parser.add_argument("--result-name", default="eval_results.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pairs = args.pairs or list(DEFAULT_PAIRS)
    rows = []
    for pair in pairs:
        result_path = args.results_root / pair / args.result_name
        if not result_path.is_file():
            raise FileNotFoundError(f"Missing result for {pair}: {result_path}")
        result = json.loads(result_path.read_text())
        row = {"pair": pair, "result_path": str(result_path), **result}
        rows.append(row)

    total_samples = sum(int(row["eval_samples"]) for row in rows)
    macro = {
        metric: sum(float(row[metric]) for row in rows) / len(rows)
        for metric in METRICS
    }
    weighted = {
        metric: sum(float(row[metric]) * int(row["eval_samples"]) for row in rows)
        / total_samples
        for metric in METRICS
    }
    summary = {
        "pairs": rows,
        "pair_count": len(rows),
        "total_eval_samples": total_samples,
        "macro_average": macro,
        "sample_weighted_average": weighted,
    }

    json_path = args.results_root / "summary.json"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    fieldnames = (
        "pair",
        "eval_samples",
        "text_candidates",
        *METRICS,
        "result_path",
    )
    tsv_path = args.results_root / "summary.tsv"
    with tsv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(json_path)
    print(tsv_path)


if __name__ == "__main__":
    main()
