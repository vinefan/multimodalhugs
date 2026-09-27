import csv
import json
import sys

import pytest

from scripts.prepare_spreadthesign_multilingual_eval import (
    LanguagePair,
    inspect_pose,
    stream_selected_rows,
    write_outputs,
)
from scripts.evaluation.summarize_spreadthesign_multilingual import (
    main as summarize_main,
)


def write_source_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("pose", "videoLanguage", "language", "text"),
        )
        writer.writeheader()
        writer.writerows(rows)


def test_prepare_pair_specific_metadata(tmp_path):
    source_csv = tmp_path / "SpreadTheSign.csv"
    pose_root = tmp_path / "poses"
    output_dir = tmp_path / "output"
    write_source_csv(
        source_csv,
        [
            {
                "pose": "ase.pose",
                "videoLanguage": "ase",
                "language": "en",
                "text": "hello",
            },
            {
                "pose": "gsg.pose",
                "videoLanguage": "gsg",
                "language": "de",
                "text": "hallo",
            },
            {
                "pose": "ignored.pose",
                "videoLanguage": "fsl",
                "language": "fr",
                "text": "bonjour",
            },
        ],
    )
    pairs = (LanguagePair("en", "ase"), LanguagePair("de", "gsg"))

    rows = stream_selected_rows(source_csv, pose_root, pairs, limit_per_pair=None)
    summary = write_outputs(
        output_dir,
        pairs,
        rows,
        validation={},
        validate_poses=False,
        max_frames=256,
    )

    assert len(rows) == 2
    assert summary["pairs"]["en_ase"]["kept_rows"] == 1
    assert summary["pairs"]["de_gsg"]["kept_rows"] == 1
    with (output_dir / "pairs" / "en_ase" / "test.tsv").open(
        encoding="utf-8", newline=""
    ) as handle:
        records = list(csv.DictReader(handle, delimiter="\t"))
    assert records[0]["encoder_prompt"] == "<en> <ase>"
    assert records[0]["output"] == "hello"


def test_pose_inspection_reads_frame_count_and_rejects_missing_file():
    valid = inspect_pose(("tests/assets/pose/sample_01.pose", 256))
    missing = inspect_pose(("tests/assets/pose/not-present.pose", 256))

    assert valid[1:3] == ("ok", 22)
    assert missing[1] == "missing"


def test_multilingual_summary_reports_macro_and_weighted_averages(
    tmp_path, monkeypatch
):
    metrics = {
        "eval_loss": 1.0,
        "v2t_r@1": 0.2,
        "v2t_r@5": 0.4,
        "v2t_r@10": 0.6,
        "v2t_median_r": 3.0,
        "v2t_mean_r": 4.0,
        "text_candidates": 8,
    }
    for pair, samples, multiplier in (("en_ase", 10, 1.0), ("de_gsg", 30, 2.0)):
        pair_dir = tmp_path / pair
        pair_dir.mkdir()
        result = {
            **metrics,
            "eval_samples": samples,
            "v2t_r@1": metrics["v2t_r@1"] * multiplier,
        }
        (pair_dir / "eval_results.json").write_text(json.dumps(result))

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "summarize_spreadthesign_multilingual.py",
            "--results-root",
            str(tmp_path),
            "--pair",
            "en_ase",
            "--pair",
            "de_gsg",
        ],
    )
    summarize_main()
    summary = json.loads((tmp_path / "summary.json").read_text())

    assert summary["macro_average"]["v2t_r@1"] == pytest.approx(0.3)
    assert summary["sample_weighted_average"]["v2t_r@1"] == pytest.approx(0.35)
    assert summary["total_eval_samples"] == 40
