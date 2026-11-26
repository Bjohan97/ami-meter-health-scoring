"""Block commits that contain confidential material.

Three layers of checks (standard library only):

1. Generic built-in patterns (safe to publish): IPv4 addresses, Windows absolute paths,
   e-mail addresses.
2. Local regex patterns from ``.confidential_patterns.local`` (git-ignored): one regex per
   line, ``#`` starts a comment. Use it for customer names, platform names, serial prefixes...
3. Local identifier fingerprints from ``.confidential_ids.sha256`` (git-ignored): SHA-256
   hashes of real meter / supply identifiers, built with ``build_confidential_ids.py``.
   Tokens found in the files are normalised and hashed the same way, so real identifiers
   are detected without keeping a plain-text copy of them.

Usage:
    python tools/check_confidential.py            # all tracked files
    python tools/check_confidential.py --staged   # files staged for commit (pre-commit hook)
    python tools/check_confidential.py --paths a.py b.toml

Exit code 0 = clean, 1 = findings, 2 = usage / environment error.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOCAL_PATTERNS = ROOT / ".confidential_patterns.local"
LOCAL_ID_HASHES = ROOT / ".confidential_ids.sha256"

BUILTIN_PATTERNS: dict[str, re.Pattern[str]] = {
    "ipv4_address": re.compile(r"(?<![\d.])(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)(?![\d.])"),
    "windows_abs_path": re.compile(r"(?<![A-Za-z])[A-Za-z]:[\\/](?![\\/])"),
    "email_address": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
}

# Tokens that may be real identifiers: alphanumeric serials or long digit runs.
ID_TOKEN = re.compile(r"[A-Za-z]{0,6}\d{5,}")

# Files that legitimately contain generic patterns as *definitions* (this checker itself).
SELF_EXEMPT = {"tools/check_confidential.py"}


def normalise_id(token: str) -> str:
    """Same normalisation as build_confidential_ids.py: strip, upper-case, drop leading zeros
    of purely numeric identifiers."""
    t = token.strip().upper()
    if t.isdigit():
        return t.lstrip("0") or "0"
    return t


def fingerprint(token: str) -> str:
    return hashlib.sha256(normalise_id(token).encode("utf-8")).hexdigest()


def load_local_patterns() -> dict[str, re.Pattern[str]]:
    patterns: dict[str, re.Pattern[str]] = {}
    if LOCAL_PATTERNS.exists():
        for n, line in enumerate(LOCAL_PATTERNS.read_text(encoding="utf-8").splitlines(), 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                patterns[f"local_pattern_{n}"] = re.compile(line)
            except re.error as exc:
                print(f"[check_confidential] invalid regex on line {n} of {LOCAL_PATTERNS.name}: {exc}", file=sys.stderr)
                sys.exit(2)
    return patterns


def load_id_hashes() -> set[str]:
    if not LOCAL_ID_HASHES.exists():
        return set()
    return {h.strip() for h in LOCAL_ID_HASHES.read_text(encoding="ascii").splitlines() if h.strip()}


def git_files(staged: bool) -> list[str]:
    cmd = ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"] if staged else ["git", "ls-files"]
    try:
        out = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"[check_confidential] cannot list git files: {exc}", file=sys.stderr)
        sys.exit(2)
    return [line for line in out.splitlines() if line.strip()]


def mask(text: str) -> str:
    """Never echo a confidential value in full."""
    return text if len(text) <= 2 else text[0] + "*" * (len(text) - 2) + text[-1]


def scan_file(path: Path, rel: str, patterns: dict[str, re.Pattern[str]], id_hashes: set[str]) -> list[str]:
    raw = path.read_bytes()
    if b"\x00" in raw[:8192]:
        return [f"{rel}: binary file - cannot be scanned; review it manually or keep it out of the repository"]
    text = raw.decode("utf-8", errors="replace")
    findings = []
    active = {k: v for k, v in patterns.items() if not (rel in SELF_EXEMPT and k in BUILTIN_PATTERNS)}
    for lineno, line in enumerate(text.splitlines(), 1):
        for name, rx in active.items():
            for m in rx.finditer(line):
                findings.append(f"{rel}:{lineno}: {name}: '{mask(m.group(0))}'")
        if id_hashes:
            for m in ID_TOKEN.finditer(line):
                token = m.group(0)
                digits = token.lstrip("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
                if fingerprint(token) in id_hashes or fingerprint(digits) in id_hashes:
                    findings.append(f"{rel}:{lineno}: real_identifier_fingerprint: '{mask(token)}'")
    return findings


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--staged", action="store_true", help="scan files staged for commit")
    group.add_argument("--paths", nargs="+", help="scan these files (relative to repo root or absolute)")
    args = ap.parse_args(argv)

    patterns = {**BUILTIN_PATTERNS, **load_local_patterns()}
    id_hashes = load_id_hashes()

    if args.paths:
        targets = [(Path(p) if Path(p).is_absolute() else ROOT / p) for p in args.paths]
    else:
        targets = [ROOT / f for f in git_files(args.staged)]

    findings: list[str] = []
    for path in targets:
        if not path.is_file():
            continue
        try:
            rel = path.resolve().relative_to(ROOT).as_posix()
        except ValueError:
            rel = path.name
        findings.extend(scan_file(path, rel, patterns, id_hashes))

    n_local = len(patterns) - len(BUILTIN_PATTERNS)
    print(f"[check_confidential] {len(targets)} file(s) scanned | built-in patterns: {len(BUILTIN_PATTERNS)} "
          f"| local patterns: {n_local} | identifier fingerprints: {len(id_hashes)}")
    if not LOCAL_PATTERNS.exists() or not id_hashes:
        print("[check_confidential] WARNING: local pattern file or identifier fingerprints missing - "
              "only generic checks were applied.")
    if findings:
        print("[check_confidential] BLOCKED - possible confidential content:")
        for f in findings:
            print("  " + f)
        return 1
    print("[check_confidential] OK - no confidential patterns found.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
