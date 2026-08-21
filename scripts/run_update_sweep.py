#!/usr/bin/env python3
"""Run reproducible one-level update correctness and throughput matrices."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import pathlib
import subprocess
import sys


BINARY_FIELDS = [
    "benchmark",
    "mode",
    "parallel_threshold",
    "serial_segments",
    "parallel_segments",
    "vertices",
    "segment_capacity",
    "density",
    "input_updates",
    "unique_updates",
    "affected_segments",
    "distribution",
    "insert_ratio",
    "duplicate_ratio",
    "preprocess_ms",
    "update_ms",
    "update_mups",
    "approx_bytes_per_update",
    "overflow_segments",
    "correct",
]


def values(case: dict, name: str) -> list:
    value = case[name]
    return value if isinstance(value, list) else [value]


def case_points(case: dict):
    axes = [
        values(case, "vertices"),
        values(case, "segment_capacity"),
        values(case, "densities"),
        values(case, "batch_sizes"),
        values(case, "insert_ratios"),
        values(case, "duplicate_ratios"),
        values(case, "distributions"),
    ]
    yield from itertools.product(*axes)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, type=pathlib.Path)
    parser.add_argument("--config", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["case", "repetition", "seed", *BINARY_FIELDS]

    with args.output.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for repetition in range(config["repetitions"]):
            for case in config["cases"]:
                for point in case_points(case):
                    (
                        vertices,
                        capacity,
                        density,
                        batch_size,
                        insert_ratio,
                        duplicate_ratio,
                        distribution,
                    ) = point
                    seed = config["base_seed"] + repetition
                    command = [
                        str(args.binary),
                        "--vertices", str(vertices),
                        "--segment-capacity", str(capacity),
                        "--density", str(density),
                        "--batch-size", str(batch_size),
                        "--insert-ratio", str(insert_ratio),
                        "--duplicate-ratio", str(duplicate_ratio),
                        "--distribution", distribution,
                        "--mode", case.get("mode", "adaptive"),
                        "--parallel-threshold",
                        str(case.get("parallel_threshold", 1024)),
                        "--zipf-skew", str(case.get("zipf_skew", 1.1)),
                        "--seed", str(seed),
                        "--warmup", str(config["warmup"]),
                        "--iterations", str(config["iterations"]),
                    ]
                    print(" ".join(command), file=sys.stderr, flush=True)
                    completed = subprocess.run(
                        command, check=True, text=True, capture_output=True
                    )
                    rows = list(csv.DictReader(completed.stdout.splitlines()))
                    if len(rows) != 1:
                        raise RuntimeError(
                            f"expected one measurement, got {len(rows)}: {command}"
                        )
                    row = rows[0]
                    if set(row) != set(BINARY_FIELDS):
                        raise RuntimeError(f"unexpected schema: {sorted(row)}")
                    row.update(
                        case=case["name"], repetition=repetition, seed=seed
                    )
                    writer.writerow(row)
                    output_file.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
