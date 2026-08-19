"""EXIT characteristic of the MacKay LDPC decoder used by the ldpc-msprs chain.

Section IV-C claims that the outer code decides which MS-PRS energy family
wins, because a steep-transfer decoder is governed by the bootstrap end of the
modem characteristic and a gradual one by the terminal end. Fig.~4 overlays the
K=3 convolutional decoder on the modem curves; this script produces the second
decoder curve so the contrast is visible rather than asserted.

The code is the SAME one the BER curves use: the regular (3,6) MacKay matrix
`MACKAY_4000_8000.alist` shipped with AFF3CT, N=8000, M=4000, rate 1/2, decoded
by 20 flooding-SPA iterations (`bp_ite` default in aff3ct/src/main.cpp, which
scripts/run_aff3ct_sims.py does not override). Using a different matrix, or a
different iteration count, would make the figure describe a decoder that
produced none of the plotted BER points.

Output matches the layout of `coder_K3.json` so make_exit_figure.py can consume
either interchangeably.

Run from the repo root in the venv:
    python scripts/make_ldpc_exit_data.py
    python scripts/make_ldpc_exit_data.py --n-pts 60 --trials 4     # quick
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import scipy.sparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nsm.codec.ldpc import _bitsandnodes, exit_curve_ldpc

ALIST = (ROOT / "aff3ct" / ".aff3ct" / "install" / "share" / "aff3ct-4.1.2"
         / "conf" / "dec" / "LDPC" / "MACKAY_4000_8000.alist")
OUT = ROOT / "results" / "exit" / "ldpc_mackay.json"

#: BP iterations per turbo pass, from aff3ct/src/main.cpp `bp_ite`. Changing
#: this without regenerating the BER caches would desynchronise figure and data.
BP_ITE = 20


def read_alist(path: Path):
    """Parse a MacKay .alist into a scipy CSR parity-check matrix.

    Layout after any leading '#' comment lines:
        N M
        max_col_weight max_row_weight
        column weights (N ints)
        row weights    (M ints)
        N lines: 1-based check indices incident on each variable
        M lines: 1-based variable indices incident on each check

    Only the variable-incidence block is used; the check block is redundant and
    is left unread. Zero padding to the maximum degree is stripped.
    """
    lines = [ln for ln in path.read_text().splitlines() if not ln.startswith("#")]
    n_var, n_chk = (int(v) for v in lines[0].split())

    rows, cols = [], []
    for v, ln in enumerate(lines[4:4 + n_var]):
        for c in (int(t) for t in ln.split()):
            if c > 0:                      # 0 is padding to the max column weight
                rows.append(c - 1)         # alist indices are 1-based
                cols.append(v)

    H = scipy.sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.int64), (rows, cols)),
        shape=(n_chk, n_var),
    )
    return H, n_var, n_chk


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-pts", type=int, default=200,
                    help="I_A grid points (coder_K3.json uses 200)")
    ap.add_argument("--ia-max", type=float, default=0.99,
                    help="upper end of the I_A grid; see the note below on why "
                         "this stops short of 1")
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--bp-ite", type=int, default=BP_ITE)
    args = ap.parse_args()

    if not ALIST.exists():
        print(f"error: {ALIST} not found. Build/install aff3ct first.", file=sys.stderr)
        return 1

    H, n_var, n_chk = read_alist(ALIST)
    deg_v = np.diff(H.tocsc().indptr)
    deg_c = np.diff(H.indptr)
    print(f"H: {n_chk} x {n_var}, rate {1 - n_chk / n_var:.3f}, "
          f"nnz {H.nnz}, var degrees {deg_v.min()}-{deg_v.max()}, "
          f"check degrees {deg_c.min()}-{deg_c.max()}")

    bits_hist, bits_values, nodes_hist, nodes_values = _bitsandnodes(H)
    # _bitsandnodes returns int32 index arrays on its sparse branch, because
    # scipy.sparse.find does, while its dense branch returns int64 from
    # np.where. _logbp_numba_n1 is compiled for int64 only, so the sparse
    # branch fails to type-check without this cast. Value-preserving.
    bits_values = np.ascontiguousarray(bits_values, dtype=np.int64)
    nodes_values = np.ascontiguousarray(nodes_values, dtype=np.int64)
    bits_hist = np.ascontiguousarray(bits_hist, dtype=np.int64)
    nodes_hist = np.ascontiguousarray(nodes_hist, dtype=np.int64)

    # The all-zero codeword is a valid choice for a linear code on a symmetric
    # channel: the decoder's extrinsic MI does not depend on which codeword was
    # sent, so this is the standard EXIT convention rather than a shortcut.
    coded = np.zeros(n_var, dtype=np.int64)

    # The grid stops at ia_max (0.99 by default) rather than 1. As I_A -> 1 the
    # a priori LLRs grow without bound, and belief propagation does not clip its
    # output the way codec/conv.py clips at +-50, so the extrinsic LLRs saturate
    # and both MI estimators break down: _mi_hist_jit returns exactly 0 (every
    # sample lands in one bin) and _mi_avg_jit dips non-physically, reading
    # 0.9389 at I_A = 0.9999 after sitting at 1.0000 just below it. Those dips
    # are numerical, not a decoder that gets worse with better a priori input.
    IA = np.linspace(1e-4, args.ia_max, args.n_pts)
    print(f"sweeping {args.n_pts} I_A points, {args.trials} trials, "
          f"{args.bp_ite} BP iterations ...")

    ie_avg, ie_hist, ia_meas = exit_curve_ldpc(
        IA, coded, bits_hist, bits_values, nodes_hist, nodes_values,
        n_chk, args.bp_ite, args.trials,
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "IA": IA.tolist(),
        "IE_avg": ie_avg.tolist(),
        "IE_hist": ie_hist.tolist(),
        "IA_measured": ia_meas.tolist(),
        "coder": {
            "family": "LDPC",
            "matrix": ALIST.name,
            "N": n_var, "M": n_chk, "K": n_var - n_chk,
            "rate": 1 - n_chk / n_var,
            "bp_iterations": args.bp_ite,
            "trials": args.trials,
            "note": "same matrix and BP iteration count as the ldpc-msprs BER caches",
        },
    }, indent=1))
    print(f"wrote {OUT}")
    print(f"  I_E at I_A=0    : {ie_avg[0]:.4f}")
    print(f"  I_E at I_A=1    : {ie_avg[-1]:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
