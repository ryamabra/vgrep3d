import pytest

from vgrep3d.evaluation import load_jsonl, summarize


def test_summarize_query_records() -> None:
    result = summarize(
        [
            {"prompt": "car", "found": True, "num_gaussians": 10},
            {"prompt": "sign", "found": True, "num_gaussians": 30},
            {"prompt": "chair", "found": False},
        ]
    )

    assert result == {
        "queries": 3,
        "found": 2,
        "found_rate": pytest.approx(2 / 3),
        "mean_support": 20.0,
        "min_support": 10,
        "max_support": 30,
    }


def test_load_jsonl_reports_bad_line(tmp_path) -> None:
    path = tmp_path / "results.jsonl"
    path.write_text('{"found": true}\nnot-json\n', encoding="utf-8")

    with pytest.raises(ValueError, match="line 2"):
        load_jsonl(path)
