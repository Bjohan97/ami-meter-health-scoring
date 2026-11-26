"""Build the local identifier-fingerprint file used by check_confidential.py.

Reads one or more local CSV exports and stores the SHA-256 fingerprint of every distinct
value found in the given columns. Only fingerprints are written; no identifier is stored in
plain text. The output file ``.confidential_ids.sha256`` is git-ignored.

Note: fingerprints of short numeric identifiers can be brute-forced; the file is a local
safety net, not a way to share identifiers.

Usage:
    python tools/build_confidential_ids.py --columns <col> [<col> ...] --files <csv> [<csv> ...]
        [--sep ";"] [--encoding latin-1]
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_confidential import LOCAL_ID_HASHES, fingerprint  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--columns", nargs="+", required=True, help="column names holding identifiers")
    ap.add_argument("--sep", default=";")
    ap.add_argument("--encoding", default="utf-8-sig")
    ap.add_argument("--min-length", type=int, default=5, help="ignore identifiers shorter than this")
    args = ap.parse_args(argv)

    hashes: set[str] = set()
    if LOCAL_ID_HASHES.exists():
        hashes |= {h.strip() for h in LOCAL_ID_HASHES.read_text(encoding="ascii").splitlines() if h.strip()}
    before = len(hashes)

    for f in args.files:
        with open(f, newline="", encoding=args.encoding, errors="replace") as fh:
            reader = csv.DictReader(fh, delimiter=args.sep)
            header = [h.lstrip("﻿") for h in (reader.fieldnames or [])]
            reader.fieldnames = header
            missing = [c for c in args.columns if c not in header]
            if missing:
                print(f"{Path(f).name}: columns not found: {missing}", file=sys.stderr)
                return 2
            for row in reader:
                for c in args.columns:
                    v = (row.get(c) or "").strip()
                    if len(v.lstrip("0")) >= args.min_length:
                        hashes.add(fingerprint(v))

    LOCAL_ID_HASHES.write_text("\n".join(sorted(hashes)) + "\n", encoding="ascii")
    print(f"fingerprints: {len(hashes)} total ({len(hashes) - before} new) -> {LOCAL_ID_HASHES.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
