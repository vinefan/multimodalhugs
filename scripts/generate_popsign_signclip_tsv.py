#!/usr/bin/env python3
"""
Generate SignCLIP-compatible TSV metadata from a PopSign-style directory tree.

Supported layouts:

1. Pre-split dataset root:
   root/
     train/
       airplane/
         sample_a.mp4
         sample_a.pose
     validation/
       ...
     test/
       ...

2. Single split root:
   root/
     airplane/
       sample_a.mp4
       sample_a.pose
     alligator/
       ...

The generated TSV files follow the Pose2Text / SignCLIP metadata format:
    signal  signal_start  signal_end  encoder_prompt  decoder_prompt  output
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


EXPECTED_COLUMNS = [
    "signal",
    "signal_start",
    "signal_end",
    "encoder_prompt",
    "decoder_prompt",
    "output",
]

SPLIT_NAMES = ("train", "validation", "test")


@dataclass(frozen=True)
class SampleRecord:
    split: str
    label: str
    pose_path: Path
    video_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate SignCLIP TSV metadata from PopSign directory layouts."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        required=True,
        help="Root directory of the PopSign dataset or a single split directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where generated TSV files will be written.",
    )
    parser.add_argument(
        "--layout",
        choices=("auto", "pre-split", "single-split"),
        default="auto",
        help="How to interpret --input-root. 'auto' detects train/validation/test if present.",
    )
    parser.add_argument(
        "--single-split-name",
        type=str,
        default="train",
        help="Split name to use when --layout=single-split or auto detects no split folders.",
    )
    parser.add_argument(
        "--signal-type",
        choices=("pose", "video"),
        default="pose",
        help="Which paired file path should be written into the TSV 'signal' column.",
    )
    parser.add_argument(
        "--output-template",
        type=str,
        default="<en> <ase> {label}",
        help=(
            "Template for the TSV output text. Available fields: {label}, {raw_label}. "
            "Defaults to a SignCLIP-style pretraining prompt."
        ),
    )
    parser.add_argument(
        "--encoder-prompt",
        type=str,
        default="",
        help="Static encoder_prompt value for every TSV row.",
    )
    parser.add_argument(
        "--decoder-prompt",
        type=str,
        default="",
        help="Static decoder_prompt value for every TSV row.",
    )
    parser.add_argument(
        "--keep-label-format",
        action="store_true",
        help="Keep raw directory names instead of replacing '_' and '-' with spaces.",
    )
    parser.add_argument(
        "--use-relative-paths",
        action="store_true",
        help="Write paths relative to --input-root instead of absolute paths.",
    )
    return parser.parse_args()


def iter_sign_dirs(root: Path) -> Iterable[Path]:
    for child in sorted(root.iterdir()):
        if child.is_dir():
            yield child


def normalize_label(raw_label: str, keep_label_format: bool) -> str:
    if keep_label_format:
        return raw_label
    return raw_label.replace("_", " ").replace("-", " ")


def detect_layout(input_root: Path, requested_layout: str) -> tuple[str, Sequence[str]]:
    if requested_layout == "pre-split":
        return "pre-split", SPLIT_NAMES
    if requested_layout == "single-split":
        return "single-split", ()

    child_names = {child.name for child in iter_sign_dirs(input_root)}
    if set(SPLIT_NAMES).issubset(child_names):
        return "pre-split", SPLIT_NAMES
    return "single-split", ()


def collect_split_records(split_root: Path, split_name: str) -> List[SampleRecord]:
    records: List[SampleRecord] = []
    for sign_dir in iter_sign_dirs(split_root):
        label = sign_dir.name
        mp4_by_stem = {path.stem: path for path in sign_dir.glob("*.mp4")}
        pose_by_stem = {path.stem: path for path in sign_dir.glob("*.pose")}

        if not mp4_by_stem and not pose_by_stem:
            continue

        stems = sorted(set(mp4_by_stem) | set(pose_by_stem))
        missing_mp4 = sorted(set(pose_by_stem) - set(mp4_by_stem))
        missing_pose = sorted(set(mp4_by_stem) - set(pose_by_stem))
        if missing_mp4 or missing_pose:
            raise ValueError(
                f"Unpaired files found in {sign_dir}: "
                f"missing mp4 for {missing_mp4[:5]}, missing pose for {missing_pose[:5]}"
            )

        for stem in stems:
            records.append(
                SampleRecord(
                    split=split_name,
                    label=label,
                    pose_path=pose_by_stem[stem],
                    video_path=mp4_by_stem[stem],
                )
            )
    return records


def collect_records(
    input_root: Path,
    layout: str,
    single_split_name: str,
) -> Dict[str, List[SampleRecord]]:
    split_to_records: Dict[str, List[SampleRecord]] = {}
    if layout == "pre-split":
        for split_name in SPLIT_NAMES:
            split_root = input_root / split_name
            if not split_root.exists():
                raise ValueError(f"Expected split directory does not exist: {split_root}")
            split_to_records[split_name] = collect_split_records(split_root, split_name)
    else:
        split_to_records[single_split_name] = collect_split_records(input_root, single_split_name)
    return split_to_records


def render_signal_path(path: Path, input_root: Path, use_relative_paths: bool) -> str:
    return str(path.relative_to(input_root) if use_relative_paths else path.resolve())


def build_tsv_rows(
    records: Sequence[SampleRecord],
    *,
    input_root: Path,
    signal_type: str,
    output_template: str,
    encoder_prompt: str,
    decoder_prompt: str,
    keep_label_format: bool,
    use_relative_paths: bool,
) -> List[dict]:
    rows: List[dict] = []
    for record in records:
        raw_label = record.label
        label = normalize_label(raw_label, keep_label_format)
        signal_path = record.pose_path if signal_type == "pose" else record.video_path
        rows.append(
            {
                "signal": render_signal_path(signal_path, input_root, use_relative_paths),
                "signal_start": 0,
                "signal_end": 0,
                "encoder_prompt": encoder_prompt,
                "decoder_prompt": decoder_prompt,
                "output": output_template.format(label=label, raw_label=raw_label),
            }
        )
    return rows


def write_tsv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=EXPECTED_COLUMNS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    input_root = args.input_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    layout, _ = detect_layout(input_root, args.layout)

    split_to_records = collect_records(
        input_root=input_root,
        layout=layout,
        single_split_name=args.single_split_name,
    )

    for split_name, records in split_to_records.items():
        rows = build_tsv_rows(
            records,
            input_root=input_root,
            signal_type=args.signal_type,
            output_template=args.output_template,
            encoder_prompt=args.encoder_prompt,
            decoder_prompt=args.decoder_prompt,
            keep_label_format=args.keep_label_format,
            use_relative_paths=args.use_relative_paths,
        )
        if not rows:
            raise ValueError(f"No rows generated for split '{split_name}' from {input_root}")
        write_tsv(output_dir / f"{split_name}.tsv", rows)
        print(f"[{split_name}] wrote {len(rows)} rows to {output_dir / f'{split_name}.tsv'}")


if __name__ == "__main__":
    main()
