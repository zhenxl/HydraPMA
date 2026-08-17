#!/usr/bin/env python3
"""Run the segment redistribution matrix and preserve every raw measurement."""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import subprocess
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, type=pathlib.Path)
    parser.add_argument("--config", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "repetition",
        "device",
        "cc",
        "mode",
        "layout",
        "segment_bytes",
        "density",
        "segments",
        "live_entries",
        "latency_ms",
        "effective_gbps",
        "valid",
    ]

    with args.output.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for repetition in range(config["repetitions"]):
            for segment_bytes in config["segment_bytes"]:
                for density in config["densities"]:
                    for layout in config.get("layouts", ["prefix"]):
                        command = [
                            str(args.binary),
                            "--segment-bytes",
                            str(segment_bytes),
                            "--density",
                            str(density),
                            "--working-set-mb",
                            str(config["working_set_mb"]),
                            "--warmup",
                            str(config["warmup"]),
                            "--iterations",
                            str(config["iterations"]),
                            "--layout",
                            layout,
                            "--mode",
                            config.get("mode", "all"),
                        ]
                        print(" ".join(command), file=sys.stderr, flush=True)
                        completed = subprocess.run(
                            command, check=True, text=True, capture_output=True
                        )
                        rows = list(csv.DictReader(completed.stdout.splitlines()))
                        if not rows:
                            raise RuntimeError(f"no measurements from: {command}")
                        for row in rows:
                            row["repetition"] = repetition
                            writer.writerow(row)
                        output_file.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
