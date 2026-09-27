#!/usr/bin/env python3
"""Prepare pair-specific SpreadTheSign metadata for SignCLIP evaluation.

The source CSV is streamed once. Only selected language pairs are retained, so
pose validation touches selected files rather than scanning the full pose tree.
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_SOURCE_CSV = Path(
    "/shares/iict-sp2.ebling.cl.uzh/common/spreadthesign/SperadTheSign.csv"
)
DEFAULT_POSE_ROOT = Path(
    "/shares/iict-sp2.ebling.cl.uzh/common/spreadthesign/sign-mt-poses"
)
DEFAULT_OUTPUT_DIR = Path(
    "/home/faxu/scratch/signclip/metadata/"
    "spreadthesign_multilingual_youtube_sl25"
)
DEFAULT_PAIRS = (
    "en:ase",
    "en:ins",
    "pl:pso",
    "de:gsg",
    "en:bfi",
    "it:ise",
    "ja:jsl",
)

EVAL_COLUMNS = (
    "signal",
    "signal_start",
    "signal_end",
    "encoder_prompt",
    "decoder_prompt",
    "output",
)
MASTER_COLUMNS = (
    "source_row",
    "text_language",
    "sign_language",
    "pose_name",
    "frame_count",
    *EVAL_COLUMNS,
)


@dataclass(frozen=True, order=True)
class LanguagePair:
    text_language: str
    sign_language: str

    @property
    def key(self) -> str:
        return f"{self.text_language}_{self.sign_language}"

    @property
    def prompt(self) -> str:
        return f"<{self.text_language}> <{self.sign_language}>"


@dataclass
class SelectedRow:
    source_row: int
    pair: LanguagePair
    pose_name: str
    pose_path: str
    text: str


def parse_pair(value: str) -> LanguagePair:
    parts = [part.strip() for part in value.split(":", maxsplit=1)]
    if len(parts) != 2 or not all(parts):
        raise argparse.ArgumentTypeError(
            f"Invalid pair {value!r}; expected TEXT_LANGUAGE:SIGN_LANGUAGE"
        )
    return LanguagePair(parts[0], parts[1])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-csv", type=Path, default=DEFAULT_SOURCE_CSV)
    parser.add_argument("--pose-root", type=Path, default=DEFAULT_POSE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--pair",
        action="append",
        type=parse_pair,
        dest="pairs",
        help=(
            "Language pair as TEXT_LANGUAGE:SIGN_LANGUAGE. Repeat for multiple "
            "pairs. Defaults to the seven core evaluation pairs."
        ),
    )
    parser.add_argument("--max-frames", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--validate-poses",
        action="store_true",
        help="Read selected pose files and keep only readable poses within max-frames.",
    )
    parser.add_argument(
        "--limit-per-pair",
        type=int,
        default=None,
        help="Optional deterministic row limit for smoke tests only.",
    )
    return parser.parse_args()


def resolve_pairs(values: list[LanguagePair] | None) -> tuple[LanguagePair, ...]:
    pairs = tuple(values) if values else tuple(parse_pair(value) for value in DEFAULT_PAIRS)
    if len(set(pairs)) != len(pairs):
        raise ValueError("Duplicate --pair values are not allowed")
    return pairs


def stream_selected_rows(
    source_csv: Path,
    pose_root: Path,
    pairs: tuple[LanguagePair, ...],
    limit_per_pair: int | None,
) -> list[SelectedRow]:
    selected_pairs = {(pair.text_language, pair.sign_language): pair for pair in pairs}
    pair_counts: Counter[LanguagePair] = Counter()
    selected: list[SelectedRow] = []

    with source_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"pose", "videoLanguage", "language", "text"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Source CSV is missing columns: {sorted(missing)}")

        for source_row, row in enumerate(reader, start=2):
            text_language = (row.get("language") or "").strip()
            sign_language = (row.get("videoLanguage") or "").strip()
            pair = selected_pairs.get((text_language, sign_language))
            if pair is None:
                continue
            if limit_per_pair is not None and pair_counts[pair] >= limit_per_pair:
                continue

            pose_name = (row.get("pose") or "").strip()
            text = (row.get("text") or "").strip()
            if not pose_name or not text:
                continue

            source_path = Path(pose_name)
            pose_path = source_path if source_path.is_absolute() else pose_root / source_path
            selected.append(
                SelectedRow(
                    source_row=source_row,
                    pair=pair,
                    pose_name=pose_name,
                    pose_path=str(pose_path),
                    text=text,
                )
            )
            pair_counts[pair] += 1

    return selected


def inspect_pose(task: tuple[str, int]) -> tuple[str, str, int | None, str]:
    pose_path, max_frames = task
    path = Path(pose_path)
    if not path.is_file():
        return pose_path, "missing", None, "file does not exist"

    try:
        from pose_format import Pose

        with path.open("rb") as handle:
            # Parse the complete body so truncated tensor payloads are rejected here,
            # before an evaluation job waits for and occupies a GPU.
            pose = Pose.read(handle)
        frame_count = int(pose.body.duration_in_frames())
    except Exception as error:  # The report must retain malformed-file failures.
        message = " ".join(str(error).split())[:500]
        return pose_path, "unreadable", None, message

    if frame_count > max_frames:
        return pose_path, "too_long", frame_count, f"{frame_count} > {max_frames}"
    return pose_path, "ok", frame_count, ""


def validate_selected_poses(
    rows: Iterable[SelectedRow], max_frames: int, num_workers: int
) -> dict[str, tuple[str, int | None, str]]:
    unique_paths = list(dict.fromkeys(row.pose_path for row in rows))
    tasks = ((path, max_frames) for path in unique_paths)
    results: dict[str, tuple[str, int | None, str]] = {}

    if num_workers <= 1:
        iterator = map(inspect_pose, tasks)
        pool = None
    else:
        pool = mp.Pool(processes=num_workers)
        iterator = pool.imap_unordered(inspect_pose, tasks, chunksize=32)

    try:
        for index, (path, status, frames, message) in enumerate(iterator, start=1):
            results[path] = (status, frames, message)
            if index % 10000 == 0 or index == len(unique_paths):
                print(f"Validated {index}/{len(unique_paths)} unique poses", flush=True)
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    return results


def write_tsv(path: Path, columns: tuple[str, ...], rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def eval_record(row: SelectedRow) -> dict[str, str | int]:
    return {
        "signal": row.pose_path,
        "signal_start": 0,
        "signal_end": 0,
        "encoder_prompt": row.pair.prompt,
        "decoder_prompt": "",
        "output": row.text,
    }


def write_outputs(
    output_dir: Path,
    pairs: tuple[LanguagePair, ...],
    rows: list[SelectedRow],
    validation: dict[str, tuple[str, int | None, str]],
    validate_poses: bool,
    max_frames: int,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    grouped: dict[LanguagePair, list[SelectedRow]] = defaultdict(list)
    rejected: list[dict] = []
    status_counts: Counter[str] = Counter()
    kept_rows: list[tuple[SelectedRow, int | None]] = []

    for row in rows:
        status, frame_count, message = validation.get(
            row.pose_path, ("unchecked", None, "pose validation not requested")
        )
        status_counts[status] += 1
        if validate_poses and status != "ok":
            rejected.append(
                {
                    "source_row": row.source_row,
                    "text_language": row.pair.text_language,
                    "sign_language": row.pair.sign_language,
                    "pose_name": row.pose_name,
                    "signal": row.pose_path,
                    "status": status,
                    "frame_count": "" if frame_count is None else frame_count,
                    "error": message,
                }
            )
            continue
        grouped[row.pair].append(row)
        kept_rows.append((row, frame_count))

    master_rows = []
    for row, frame_count in kept_rows:
        record = {
            "source_row": row.source_row,
            "text_language": row.pair.text_language,
            "sign_language": row.pair.sign_language,
            "pose_name": row.pose_name,
            "frame_count": "" if frame_count is None else frame_count,
        }
        record.update(eval_record(row))
        master_rows.append(record)
    write_tsv(output_dir / "master.tsv", MASTER_COLUMNS, master_rows)

    failure_columns = (
        "source_row",
        "text_language",
        "sign_language",
        "pose_name",
        "signal",
        "status",
        "frame_count",
        "error",
    )
    write_tsv(output_dir / "rejected.tsv", failure_columns, rejected)

    pair_summaries = {}
    for pair in pairs:
        pair_rows = grouped[pair]
        pair_path = output_dir / "pairs" / pair.key / "test.tsv"
        write_tsv(pair_path, EVAL_COLUMNS, (eval_record(row) for row in pair_rows))
        pair_summaries[pair.key] = {
            "text_language": pair.text_language,
            "sign_language": pair.sign_language,
            "prompt": pair.prompt,
            "selected_rows": sum(1 for row in rows if row.pair == pair),
            "kept_rows": len(pair_rows),
            "unique_poses": len({row.pose_path for row in pair_rows}),
            "unique_texts": len({row.text for row in pair_rows}),
            "metadata_tsv": str(pair_path),
        }

    summary = {
        "pose_validation_enabled": validate_poses,
        "max_frames": max_frames,
        "selected_rows": len(rows),
        "kept_rows": len(kept_rows),
        "rejected_rows": len(rejected),
        "status_counts": dict(sorted(status_counts.items())),
        "pairs": pair_summaries,
    }
    return summary


def main() -> None:
    args = parse_args()
    pairs = resolve_pairs(args.pairs)
    if args.max_frames <= 0:
        raise ValueError("--max-frames must be positive")
    if args.num_workers <= 0:
        raise ValueError("--num-workers must be positive")
    if args.limit_per_pair is not None and args.limit_per_pair <= 0:
        raise ValueError("--limit-per-pair must be positive")

    print(f"Source CSV: {args.source_csv}")
    print(f"Pose root: {args.pose_root}")
    print(f"Output directory: {args.output_dir}")
    print("Pairs: " + ", ".join(pair.key for pair in pairs))
    rows = stream_selected_rows(
        args.source_csv, args.pose_root, pairs, args.limit_per_pair
    )
    print(f"Selected rows: {len(rows)}")

    validation = {}
    if args.validate_poses:
        validation = validate_selected_poses(rows, args.max_frames, args.num_workers)

    summary = write_outputs(
        args.output_dir,
        pairs,
        rows,
        validation,
        args.validate_poses,
        args.max_frames,
    )
    summary["source_csv"] = str(args.source_csv)
    summary["pose_root"] = str(args.pose_root)
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(summary_path)


if __name__ == "__main__":
    main()
