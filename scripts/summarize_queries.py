#!/usr/bin/env python3
"""Summarize JSONL records emitted by repeated vgrep3d queries."""

from __future__ import annotations

import argparse
import json

from vgrep3d.evaluation import load_jsonl, summarize


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize found rate and Gaussian support counts from query JSONL."
    )
    parser.add_argument("results", help="Path to a JSONL file containing one query per line")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    args = parser.parse_args()

    summary = summarize(load_jsonl(args.results))
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return

    print(f"Queries:      {summary['queries']}")
    print(f"Found:        {summary['found']} ({summary['found_rate']:.1%})")
    if summary["mean_support"] is not None:
        print(f"Mean support: {summary['mean_support']:,.1f}")
        print(f"Support span: {summary['min_support']:,} - {summary['max_support']:,}")
    else:
        print("Mean support: n/a")


if __name__ == "__main__":
    main()
