#!/usr/bin/env python3
"""Check the two core-Lean modules of formal/ (no mathlib needed) with the pinned toolchain.

formal/FormalLean/CuspJacobian.lean (factorization of the cusp Jacobian) and
formal/FormalLean/CuspCertificateDecimals.lean (exact checks of the decimal bounds of the two
half-line cusp certificates) import nothing, so `lean <file>` checks them without building mathlib.
The script runs `lean` from formal/ (so that elan selects the toolchain of formal/lean-toolchain),
requires an empty output for each file, and scans both files for sorry/admit/native_decide.
Run by reproduce_all.py --full when `lean` is on the PATH.
Writes outputs/tables/formal_cusp_check.txt.
"""
from __future__ import annotations

import re
import shutil
import subprocess

from paths import ROOT, TABLES

FORMAL = ROOT / "formal"
FILES = ("FormalLean/CuspJacobian.lean", "FormalLean/CuspCertificateDecimals.lean")


def main():
    lines = []
    lean = shutil.which("lean")
    if lean is None:
        raise SystemExit("lean not found on the PATH")
    ok = True
    for name in FILES:
        source = (FORMAL / name).read_text()
        theorems = len(re.findall(r"^theorem ", source, re.M))
        forbidden = re.findall(r"\b(sorry|admit|native_decide)\b", re.sub(r"--.*|/-.*?-/", "", source, flags=re.S))
        proc = subprocess.run([lean, name], cwd=FORMAL, capture_output=True, text=True)
        out = (proc.stdout + proc.stderr).strip()
        good = proc.returncode == 0 and not out and not forbidden
        ok &= good
        lines.append(f"{name}: {theorems} theorems; lean exit code {proc.returncode}; "
                     f"output {'empty' if not out else repr(out[:300])}; "
                     f"sorry/admit/native_decide: {len(forbidden)} -> {'ok' if good else 'FAIL'}")
    lines.append(f"result: {'PASS' if ok else 'FAIL'}")
    print("\n".join(lines))
    (TABLES / "formal_cusp_check.txt").write_text("\n".join(lines) + "\n")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
