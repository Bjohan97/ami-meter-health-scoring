"""Command line: ``ami-health score | from-ami-dq | synthetic``.

Outputs (deterministic CSV): health.csv (every meter, every score and feature), field_list.csv
(meters needing action, in priority order), group_summary.csv, run_manifest.json.
Exit code 2 on configuration or input errors.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

from . import __version__
from .config import ConfigError, load_settings
from .inputs import from_ami_data_quality, load_inputs
from .pipeline import group_summary, score_fleet


def _sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_outputs(health: pd.DataFrame, out: str | Path, inputs: dict, config_path) -> None:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    kw = dict(index=False, float_format="%.6g", lineterminator="\n")
    health.to_csv(out / "health.csv", **kw)
    health[health["diagnosis"] != "OK"].to_csv(out / "field_list.csv", **kw)
    group_summary(health).to_csv(out / "group_summary.csv", **kw)
    manifest = {"tool": "ami-meter-health-scoring", "version": __version__,
                "config": Path(config_path).name if config_path else "defaults",
                "inputs": {k: {"file": Path(v).name, "sha256": _sha(v)} for k, v in inputs.items() if v},
                "meters": int(len(health)), "diagnoses": health["diagnosis"].value_counts().to_dict()}
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="ami-health", description="Health scoring of AMI meters.")
    sub = ap.add_subparsers(dest="command", required=True)
    sc = sub.add_parser("score", help="score from the minimal input tables")
    sc.add_argument("--meter-days", required=True)
    sc.add_argument("--slots")
    sc.add_argument("--voltage")
    sc.add_argument("--events")
    sc.add_argument("--exclude-days", nargs="*", default=[], help="local dates (YYYY-MM-DD) left out, e.g. truncated days")
    fa = sub.add_parser("from-ami-dq", help="score from an ami-data-quality output folder")
    fa.add_argument("--dir", required=True)
    fa.add_argument("--events")
    for p in (sc, fa):
        p.add_argument("--groups")
        p.add_argument("--group-col")
        p.add_argument("--config")
        p.add_argument("--out", required=True)
    sy = sub.add_parser("synthetic", help="write a SYNTHETIC example fleet")
    sy.add_argument("--out", required=True)
    sy.add_argument("--days", type=int, default=14)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "synthetic":
            from .synthetic import generate

            print(f"synthetic fleet written to {args.out}: {generate(args.out, days=args.days)}")
            return 0
        settings = load_settings(args.config)
        if args.command == "score":
            inp = load_inputs(args.meter_days, args.slots, args.voltage, args.groups, args.events, args.exclude_days)
            files = {"meter_days": args.meter_days, "slots": args.slots, "voltage": args.voltage,
                     "groups": args.groups, "events": args.events}
        else:
            inp = from_ami_data_quality(args.dir, args.groups, args.events)
            files = {"meter_days": Path(args.dir) / "meter_days_interval.csv", "groups": args.groups,
                     "events": args.events}
        if args.group_col and (inp.groups is None or args.group_col not in inp.groups.columns):
            raise ConfigError(f"--group-col {args.group_col!r} not found in --groups")
        health = score_fleet(inp, settings, args.group_col)
        write_outputs(health, args.out, files, args.config)
        print(f"{len(health)} meters scored; diagnoses: {health['diagnosis'].value_counts().to_dict()}")
        return 0
    except (ConfigError, FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
