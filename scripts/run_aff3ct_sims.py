"""Drive the AFF3CT C++ BER binary and write caches in the Python format.

The C++ chain (aff3ct/) is the source of results; this script is only the
bridge, so that everything downstream (make_ber_overview.py,
make_ber_convergence.py, render_testbed_ber.py, the notebooks) keeps reading
one cache layout regardless of which implementation produced it.

The two implementations are validated equivalent, not merely similar: the
modulator, both BCJRs and the encoder are diffed bit-for-bit against nsm/ on
identical input via the binary's modtest / demodtest / tdemodtest / enctest /
dectest modes. So caches from either carry the same METRIC_CONVENTION and may
share a figure; `implementation` records which one ran.

Usage (from the repo root, after building aff3ct/):

    python scripts/run_aff3ct_sims.py --L0 3 4 5 6 --family unbalanced balanced
    python scripts/run_aff3ct_sims.py --L0 3 --family balanced --mode uncoded-msprs
    python scripts/run_aff3ct_sims.py --L0 3 --family unbalanced --jobs 1 --dry-run
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nsm.bounds import clopper_pearson
from nsm.provenance import stamp

BINARY   = ROOT / "aff3ct" / "build" / "bin" / "msprs"
TAPS     = ROOT / "nsm" / "modem" / "filters"
OUT_ROOT = ROOT / "results" / "ber"
# Absolute, because the binary is invoked with cwd = bin/ and the C++ default
# for this path is written relative to build/. Everything the script passes to
# the binary is absolute for the same reason.
LDPC_H   = (ROOT / "aff3ct" / ".aff3ct" / "install" / "share" / "aff3ct-4.1.2"
            / "conf" / "dec" / "LDPC" / "MACKAY_4000_8000.alist")

# "     3.00 |      3.00 |        300 |      12927 |  8.621e-03 | 18.5"
_ROW = re.compile(
    r"^\s*(-?[\d.]+)\s*\|\s*(-?[\d.]+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*([\d.eE+-]+)\s*\|\s*([\d.]+)\s*$"
)
# "# per-iter 3.0 149940 26596 15674 ..."
_ITER = re.compile(r"^#\s*per-iter\s+(-?[\d.]+)\s+(\d+)\s+(.*)$")


def scheme_dir(mode: str, L0: int, family: str, iters: int) -> Path:
    if mode == "uncoded-msprs":
        return OUT_ROOT / f"nsm_L{L0}_{family}_uncoded"
    if mode == "coded-msprs":
        return OUT_ROOT / f"nsm_L{L0}_{family}_conv_K3_{iters}iters"
    if mode == "coded-bpsk":
        return OUT_ROOT / "ask2_conv_K3"
    if mode == "coded-ftn":
        return OUT_ROOT / f"ftn_tau0p5_conv_K3_{iters}iters"
    if mode == "ldpc-msprs":
        return OUT_ROOT / f"ldpc_msprs_L{L0}_{family}_turbo{iters}"
    raise ValueError(f"unknown mode {mode!r}")


def parse(text: str) -> dict[float, dict]:
    """Table rows keyed by Eb/N0, merged with their per-iteration sidecars."""
    points: dict[float, dict] = {}
    for line in text.splitlines():
        m = _ROW.match(line)
        if m:
            ebn0 = float(m.group(1))
            points.setdefault(ebn0, {}).update(
                eb_no_db=ebn0,
                es_no_db=float(m.group(2)),
                n_frames=int(m.group(3)),
                ers_cnt=int(m.group(4)),
                ber=float(m.group(5)),
                duration_s=float(m.group(6)),
            )
            continue
        m = _ITER.match(line)
        if m:
            ebn0 = float(m.group(1))
            points.setdefault(ebn0, {}).update(
                bits_cnt=int(m.group(2)),
                ers_per_iter=[int(v) for v in m.group(3).split()],
            )
    return points


def run_one(mode: str, L0: int, family: str, args) -> tuple[str, int, str]:
    cmd = [
        str(BINARY), "--mode", mode, "--L0", str(L0), "--family", family,
        "--taps", str(TAPS), "--iters", str(args.iters),
        "--ebn0-min", str(args.ebn0_min), "--ebn0-max", str(args.ebn0_max),
        "--ebn0-step", str(args.ebn0_step), "--min-fra", str(args.min_fra),
        "--be", str(args.be), "--max-fra", str(args.max_fra),
        "--threads", str(args.threads),
    ]
    if mode == "ldpc-msprs":
        cmd += ["--ldpc-h", str(LDPC_H)]
    label = f"{mode} L0={L0} {family}"
    if args.dry_run:
        return label, 0, " ".join(cmd)

    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(BINARY.parent))
    if proc.returncode != 0:
        return label, proc.returncode, proc.stderr[-2000:]

    points = parse(proc.stdout)
    if not points:
        return label, 1, "no BER rows parsed:\n" + proc.stdout[-2000:]

    out = scheme_dir(mode, L0, family, args.iters)
    out.mkdir(parents=True, exist_ok=True)
    skipped = []
    for ebn0, rec in sorted(points.items()):
        # bits_cnt only arrives on the per-iter sidecar, which uncoded runs
        # do not emit; derive it from the frame count in that case.
        src_bits = 9998 if mode == "uncoded-msprs" else (4000 if mode == "ldpc-msprs" else 4998)
        rec.setdefault("bits_cnt", rec["n_frames"] * src_bits)
        # Every point carries its exact binomial interval. Without it a
        # zero-error point serialises as ber = 0.0 and reads downstream as a
        # measurement rather than the upper bound it actually is, which is how
        # the pre-2026-08 caches ended up quoting BER 0.0e+00 and 1e-8 from two
        # error events.
        lo, hi = clopper_pearson(rec["ers_cnt"], rec["bits_cnt"])
        rec["ber_lo95"], rec["ber_ub95"] = float(lo), float(hi)
        rec["is_upper_bound"] = rec["ers_cnt"] == 0
        rec["timestamp"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        rec["config"] = stamp({
            "implementation": "aff3ct-4.1.2",
            "scheme": mode, "L0": L0, "filter_type": family,
            **({"tau": 0.5, "rolloff": 0.3, "L_isi": 5} if mode == "coded-ftn" else {}),
            "source_bits": src_bits,
            **({"outer_code": "LDPC MACKAY_4000_8000, rate 1/2", "bp_ite": 20}
               if mode == "ldpc-msprs" else {"K": 3, "octal_code": [5, 7]}),
            "max_iters": args.iters, "interleaver_seed": 42,
            "be_target": args.be, "min_fra": args.min_fra, "threads": args.threads,
        })
        # Never replace a point with a worse-sampled one. Below an LDPC
        # threshold the errors arrive as rare catastrophic frame failures, so a
        # short run legitimately returns zero errors where a long one measured a
        # real BER: a 4000-frame pass reported 0 events at 5.0 dB where a
        # 60000-frame pass had measured 7.6e-6 from 1832. Both are honest, but
        # overwriting the long one with the short one destroys the curve, and it
        # does so silently because the file simply gets smaller numbers.
        dst = out / f"snr_{ebn0:.1f}dB.json"
        if dst.exists() and not args.force:
            prev = json.loads(dst.read_text())
            if prev.get("bits_cnt", 0) > rec["bits_cnt"] and prev.get("ers_cnt", 0) > 0:
                skipped.append(f"{ebn0:.1f}")
                continue
        dst.write_text(json.dumps(rec, indent=1))
    msg = f"{len(points) - len(skipped)} points -> {out.relative_to(ROOT)}"
    if skipped:
        msg += f"   (kept {len(skipped)} better-sampled: {', '.join(skipped)} dB)"
    return label, 0, msg


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", default="coded-msprs",
                    choices=["coded-msprs", "uncoded-msprs", "coded-bpsk", "coded-ftn",
                             "ldpc-msprs"])
    ap.add_argument("--L0", type=int, nargs="+", default=[3, 4, 5, 6])
    ap.add_argument("--family", nargs="+", default=["unbalanced", "balanced"])
    ap.add_argument("--iters", type=int, default=7)
    ap.add_argument("--ebn0-min", type=float, default=0.0)
    ap.add_argument("--ebn0-max", type=float, default=6.01)
    ap.add_argument("--ebn0-step", type=float, default=0.5)
    ap.add_argument("--be", type=int, default=3000)
    ap.add_argument("--min-fra", type=int, default=300)
    ap.add_argument("--max-fra", type=int, default=2000000)
    ap.add_argument("--jobs", type=int, default=8, help="curves in flight")
    ap.add_argument("--threads", type=int, default=1,
                    help="threads inside each C++ run; jobs * threads should be <= cores")
    ap.add_argument("--force", action="store_true",
                    help="overwrite existing points even when they are better sampled")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not BINARY.exists():
        print(f"error: {BINARY} not found. Build aff3ct/ first.", file=sys.stderr)
        return 1

    jobs = [(args.mode, L0, fam) for L0 in args.L0 for fam in args.family]
    print(f"{len(jobs)} curve(s), {args.jobs} at a time")

    rc = 0
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        futs = {ex.submit(run_one, m, L, f, args): (m, L, f) for m, L, f in jobs}
        for fut in as_completed(futs):
            label, code, msg = fut.result()
            print(f"[{'ok ' if code == 0 else 'FAIL'}] {label}: {msg}", flush=True)
            rc |= code
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
