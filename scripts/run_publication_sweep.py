#!/usr/bin/env python3
"""Run concurrent COW publication and snapshot-validation sweeps."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import pathlib
import subprocess
import sys


BINARY_FIELDS = [
    "benchmark", "segments", "capacity", "epochs", "query_blocks",
    "repetitions", "publish_ms", "publications_mps",
    "accepted_snapshots", "query_retries", "snapshot_mismatches",
    "cas_failures", "correct",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, type=pathlib.Path)
    parser.add_argument("--config", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["case", *BINARY_FIELDS]

    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for case in config["cases"]:
            points = itertools.product(
                case["segments"], case["capacities"], case["epochs"]
            )
            for segments, capacity, epochs in points:
                command = [
                    str(args.binary),
                    "--segments", str(segments),
                    "--capacity", str(capacity),
                    "--epochs", str(epochs),
                    "--query-blocks", str(case.get("query_blocks", 0)),
                    "--repetitions", str(config["repetitions"]),
                ]
                print(" ".join(command), file=sys.stderr, flush=True)
                completed = subprocess.run(
                    command, check=True, text=True, capture_output=True
                )
                rows = list(csv.DictReader(completed.stdout.splitlines()))
                if len(rows) != 1 or set(rows[0]) != set(BINARY_FIELDS):
                    raise RuntimeError(f"unexpected output schema: {command}")
                row = rows[0]
                row["case"] = case["name"]
                writer.writerow(row)
                stream.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
