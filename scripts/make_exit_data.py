"""Generate modem EXIT-curve data for the §IV.A EXIT figure.

Generates the modem EXIT caches read by make_exit_figure.py, so the
output JSONs share the schema the figure reader expects:

    results/exit/nsm_L{L0}_{filter}.json
        {"IA": [N_IA], "eb_no_db": [N_SNR],
         "results": {"<eb>": {"IE_avg": [...], "IE_hist": [...], "IE_mag": [...]}},
         "meta": {...}}

Idempotent: a config whose JSON already exists is skipped unless ``--force``.
The 4-ASK and convolutional-decoder EXIT JSONs are produced by the sibling
EXIT sweeps and are not regenerated here.

Run from repo root in the venv:
    python scripts/make_exit_data.py                 # all 8 MS-PRS configs
    python scripts/make_exit_data.py --L0 5 6        # subset
    python scripts/make_exit_data.py --force         # recompute existing
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nsm.modem.msprs import precompute, modulate, exit_curve_nsm
from nsm.provenance import stamp
from nsm.channel.awgn import setup as awgn_setup

# ── EXIT data-generation parameters ──
# These reproduce the recipe that produced the original nsm_L3_balanced.json so
# all eight panels of the EXIT figure share one provenance and one Eb/N0 axis.
#
# RESOLVED 2026-08-17, but NOT by this script.
#
# Citation correction: the note that stood here said "Siala review comments
# 62/63". That is wrong. In Siala-comments-extracted.md, 62 concerns the +-1
# equivalence of the bipolar streams and 63 is the single word "Modulations?".
# Neither mentions EXIT charts. The comments about the balanced-vs-unbalanced
# ordering are 171, 172, 173 and 180, and the review file flags 171 vs 172 and
# 172/173 vs 180 as internal contradictions, so data alone cannot close them:
# what the fix below buys is a trustworthy measurement to argue from.
# Use scripts/run_aff3ct_exit.py instead; this one is kept as the reference
# implementation the C++ was validated against.
#
# The crossovers were diagnosed rather than out-sampled. `_make_rx` below draws
# ONE (bits, rx) pair per Eb/N0 and reuses it across the entire IA grid and
# every trial, so averaging over a-priori draws cannot remove that
# realisation's error: every trial sees the same channel and the bias lands on
# the whole curve. Balanced and unbalanced draw independently, so their biases
# are independent, and their ordering near I_A = 1 flips between runs.
#
# It was not cosmetic. In the caches this script produced, the balanced-minus-
# unbalanced sign near I_A -> 1 was wrong at 5 of 21 Eb/N0 points >= 6 dB for
# L0 = 5, and at 21 of 21 for L0 = 6, i.e. inverted everywhere.
#
# Redrawing the channel per trial fixes it: 0 of 21 wrong at every L0, for 4.5 %
# more compute. The plan recorded here, N_TRIALS >= 200 and SOURCE_BITS >= 20000
# at ~30 h, would have reduced the variance without removing the bias. The
# replacement regenerates all eight caches in 64 min.
N_TRIALS = 50           # a-priori realisations averaged per IA point
N_IA = 100              # IA grid points in [0, 1]
SOURCE_BITS = 4998      # info bits per realisation (even → MS-PRS pairs)
EB_NO_DB = np.linspace(0.0, 10.0, 51)            # 0..10 dB, 0.2-dB steps
IA_GRID = 0.999 * np.linspace(1e-3, 1.0, N_IA)

# avg_bit_energy passed to awgn.setup. Use the SAME (avg_bit_energy=0.5, rate=0.5)
# convention as the BER simulation (scripts/run_offline_sims.py:92) and the 4-ASK
# EXIT caches, so the Eb/N0 axis here is the per-INFORMATION-bit Eb/N0 of Figs 5/8
# (avg_bit_energy=0.5 is the per-coded-bit energy; rate=0.5 converts it to the
# per-info-bit abscissa, N0=avg_bit_energy/(rate*gamma) => Eb_info/N0=gamma), and the
# shared colorbar is consistent across the 4-ASK and MS-PRS panels. (The earlier
# rate=1.0 setting gave the MS-PRS panels half the noise of the 4-ASK panels at
# the same dB, making the comparison unreliable.)
EB_PER_BIT = 0.5

OUT = ROOT / "results" / "exit"


def _make_rx(L0: int, filt: str, eb_no_db: float, seed: int = 0xE117):
    """Modulate a fixed random payload and add AWGN at the given Eb/N0."""
    nsm = precompute(L0, SOURCE_BITS, filt)
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, SOURCE_BITS).astype(np.int64)
    sym = modulate(bits, L0, nsm["h0"], nsm["h1"])
    ch = awgn_setup((eb_no_db, eb_no_db, 1), EB_PER_BIT, rate=0.5)
    noise_std = float(ch["noise_std"][0])
    rng_n = np.random.default_rng(seed ^ 0xABCD)
    rx = sym + noise_std * rng_n.standard_normal(sym.shape)
    return nsm, bits, rx, float(ch["noise_var"][0])


def generate_exit_data(L0: int, filt: str) -> Path:
    """Sweep Eb/N0; for each, compute the NSM modem EXIT curve over IA_GRID."""
    nsm, _, _, _ = _make_rx(L0, filt, 0.0)  # warm precompute / JIT
    results = {}
    t0 = time.time()
    for eb in EB_NO_DB:
        nsm, bits, rx, nvar = _make_rx(L0, filt, float(eb))
        ie_avg, ie_hist, ie_mag, ia_meas = exit_curve_nsm(
            IA_GRID, bits, rx, nvar, nsm["modulation_length"],
            nsm["branch_labels"], len(bits), nsm["memory"],
            nsm["total_states"], nsm["next_states"],
            nsm["branch_indices"], N_TRIALS,
        )
        results[f"{eb:.1f}"] = {
            "IE_avg": ie_avg.tolist(),
            "IE_hist": ie_hist.tolist(),
            "IE_mag": ie_mag.tolist(),
            "IA_measured": ia_meas.tolist(),
        }
    dur = time.time() - t0
    payload = {
        "IA": IA_GRID.tolist(),
        "eb_no_db": [float(x) for x in EB_NO_DB],
        "results": results,
        "meta": stamp({"L0": L0, "filter": filt, "N_TRIALS": N_TRIALS,
                       "source_bits": SOURCE_BITS, "duration_s": round(dur, 1)}),
    }
    out = OUT / f"nsm_L{L0}_{filt}.json"
    out.write_text(json.dumps(payload))
    print(f"  wrote {out.relative_to(ROOT)}  ({dur:.0f} s)", flush=True)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--L0", nargs="+", type=int, default=[3, 4, 5, 6])
    p.add_argument("--filter", nargs="+", default=["balanced", "unbalanced"],
                   choices=["balanced", "unbalanced"], dest="filters")
    p.add_argument("--force", action="store_true",
                   help="Recompute even if the output JSON already exists")
    args = p.parse_args(argv)

    OUT.mkdir(parents=True, exist_ok=True)
    grand = time.time()
    for L0 in args.L0:
        for filt in args.filters:
            out = OUT / f"nsm_L{L0}_{filt}.json"
            tag = f"nsm_L{L0}_{filt}"
            if out.exists() and not args.force:
                print(f"  skip {tag}  (exists)", flush=True)
                continue
            print(f"=== generating {tag} ===", flush=True)
            generate_exit_data(L0, filt)
    print(f"\nAll EXIT data done in {(time.time()-grand):.0f} s", flush=True)


if __name__ == "__main__":
    main()
