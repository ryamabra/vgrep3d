"""Small, dependency-free helpers for summarizing 3D query runs."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path


def load_jsonl(path: str | Path) -> list[dict]:
    """Load query records from JSON Lines with useful line-numbered errors."""
    records = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc.msg}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Line {line_number} must contain a JSON object")
            records.append(record)
    return records


def summarize(records: Iterable[dict]) -> dict:
    """Compute stable aggregate metrics from serialized query results."""
    rows = list(records)
    found = [row for row in rows if bool(row.get("found"))]
    supports = [int(row["num_gaussians"]) for row in found if "num_gaussians" in row]
    return {
        "queries": len(rows),
        "found": len(found),
        "found_rate": len(found) / len(rows) if rows else 0.0,
        "mean_support": sum(supports) / len(supports) if supports else None,
        "min_support": min(supports) if supports else None,
        "max_support": max(supports) if supports else None,
    }
