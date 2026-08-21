#!/usr/bin/env python3
"""Run atomic versus warp-aggregated delta append experiments."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import pathlib
import subprocess
import sys


BINARY_FIELDS = [
    "benchmark", "mode", "input_order", "distribution", "vertices",
    "block_capacity", "batch_size", "blocks_used",
    "reservation_atomics", "rollover_attempts", "cas_retries",
    "wasted_blocks", "pool_exhausted", "append_ms", "append_mups",
    "reservation_atomics_per_update", "ready_records", "correct",
]


def axis(case: dict, name: str) -> list:
    value = case[name]
    return value if isinstance(value, list) else [value]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, type=pathlib.Path)
    parser.add_argument("--config", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["case", "repetition", "seed", *BINARY_FIELDS]

    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for repetition in range(config["repetitions"]):
            for case in config["cases"]:
                points = itertools.product(
                    axis(case, "vertices"),
                    axis(case, "block_capacities"),
                    axis(case, "batch_sizes"),
                    axis(case, "input_orders"),
                    axis(case, "distributions"),
                )
                for vertices, capacity, batch, order, distribution in points:
                    seed = config["base_seed"] + repetition
                    command = [
                        str(args.binary),
                        "--vertices", str(vertices),
                        "--block-capacity", str(capacity),
                        "--batch-size", str(batch),
                        "--input-order", order,
                        "--distribution", distribution,
                        "--mode", "all",
                        "--insert-ratio", str(case.get("insert_ratio", 0.5)),
                        "--zipf-skew", str(case.get("zipf_skew", 1.1)),
                        "--pool-factor", str(case.get("pool_factor", 8)),
                        "--seed", str(seed),
                        "--warmup", str(config["warmup"]),
                        "--iterations", str(config["iterations"]),
                    ]
                    print(" ".join(command), file=sys.stderr, flush=True)
                    completed = subprocess.run(
                        command, check=True, text=True, capture_output=True
                    )
                    rows = list(csv.DictReader(completed.stdout.splitlines()))
                    modes = {row["mode"] for row in rows}
                    if not {"atomic", "warp"}.issubset(modes):
                        raise RuntimeError(f"missing comparison rows: {command}")
                    for row in rows:
                        if set(row) != set(BINARY_FIELDS):
                            raise RuntimeError(
                                f"unexpected schema: {sorted(row)}"
                            )
                        row.update(
                            case=case["name"],
                            repetition=repetition,
                            seed=seed,
                        )
                        writer.writerow(row)
                    stream.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
