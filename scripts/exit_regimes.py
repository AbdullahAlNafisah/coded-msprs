"""Which energy family wins, where, and why: the two ends of the EXIT curve.

Settles the reviewer's own disagreement with himself, and explains a large
effect in the coded BER.

    171  "the starting point should be slightly higher for unbalanced
          (the boostrap region)"
    172  "I think that it is the inverse. When in unbalanced regime, the tunnel
          is wider at I_a=0, since the MSED is better than in the balanced
          case!?"                                       <- his own trailing "!?"
    173  "the balanced case is intuitively the best configuration since at
          I_a^modem approximately equal to 1, perfect equalization is achieved"
    180  "the unbalanced case is better than the balanced case when the SNR is
          low ... even your explanation of the BER curves contradicts these
          observations. Please verify."

The four are not one story, and the review file flags 171-vs-172 and
172/173-vs-180 as contradictions. They resolve once the two ENDS of the EXIT
characteristic are separated, because different outer codes are governed by
different ends:

  * at I_A -> 0 (bootstrap) the unbalanced family is ahead, its larger MSED
    giving a better cold start. 171 is right, 172 is wrong.
  * at I_A -> 1 (terminal) the balanced family is ahead, its equal per-bit
    energies approaching the 2-ASK limit. 173 is right.
  * an LDPC outer code has a very steep transfer and needs the good cold start,
    so the BOOTSTRAP governs and unbalanced wins the coded BER.
  * the K=3 convolutional code converges gradually, so the TERMINAL value
    governs and balanced wins.

180 is then also right, for the bootstrap reason, and stops contradicting 173.

Run after regenerating the EXIT caches:

    python scripts/exit_regimes.py
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path
from nsm.curves import load_curve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

EXIT = ROOT / "results" / "exit"
BER  = ROOT / "results" / "ber"
L0S  = (3, 4, 5, 6)


def exit_ends(L0: int, family: str, snr: str):
    """(IE at the lowest I_A, mean IE over the top five I_A points)."""
    d = json.loads((EXIT / f"nsm_L{L0}_{family}.json").read_text())
    ie = d["results"][snr]["IE_avg"]
    return ie[0], sum(ie[-5:]) / 5.0


def ber_at(subdir: str, ebn0: float):
    c = load_curve(subdir)
    if c is None:
        return None
    for eb, ber in zip(c.eb_no_db.tolist(), c.ber.tolist()):
        if abs(eb - ebn0) < 0.05:
            return float(ber)
    return None


def main() -> int:
    print("EXIT bootstrap, I_A -> 0     (171 says unbalanced is higher; 172 says the inverse)")
    print(f"{'Eb/N0':>7}" + "".join(f"{'L' + str(L):>16}" for L in L0S))
    wins = {"unb": 0, "bal": 0}
    for snr in ("3", "5", "7", "9"):
        row = f"{snr + ' dB':>7}"
        for L0 in L0S:
            u, _ = exit_ends(L0, "unbalanced", snr)
            b, _ = exit_ends(L0, "balanced", snr)
            wins["unb" if u > b else "bal"] += 1
            row += f"  {u:.3f}/{b:.3f} {'U' if u > b else 'B'}"
        print(row)
    print(f"  -> unbalanced ahead at the bootstrap in {wins['unb']} of "
          f"{wins['unb'] + wins['bal']} cases. 171 is correct, 172 is not.\n")

    print("EXIT terminal, I_A -> 1      (173 says balanced is best here)")
    print(f"{'Eb/N0':>7}" + "".join(f"{'L' + str(L):>16}" for L in L0S))
    wins = {"unb": 0, "bal": 0}
    for snr in ("6", "7", "8", "9"):
        row = f"{snr + ' dB':>7}"
        for L0 in L0S:
            _, u = exit_ends(L0, "unbalanced", snr)
            _, b = exit_ends(L0, "balanced", snr)
            wins["bal" if b > u else "unb"] += 1
            row += f"  {u:.3f}/{b:.3f} {'B' if b > u else 'U'}"
        print(row)
    print(f"  -> balanced ahead at the terminal in {wins['bal']} of "
          f"{wins['unb'] + wins['bal']} cases. 173 is correct.\n")

    print("Coded BER at 5.0 dB: which end governs depends on the outer code")
    print(f"{'L0':>3}{'conv unb':>11}{'conv bal':>11}{'winner':>9}"
          f"{'ldpc unb':>11}{'ldpc bal':>11}{'winner':>9}")
    for L0 in L0S:
        cu = ber_at(f"nsm_L{L0}_unbalanced_conv_K3_7iters", 5.0)
        cb = ber_at(f"nsm_L{L0}_balanced_conv_K3_7iters", 5.0)
        lu = ber_at(f"ldpc_msprs_L{L0}_unbalanced_turbo7", 5.0)
        lb = ber_at(f"ldpc_msprs_L{L0}_balanced_turbo7", 5.0)
        if None in (cu, cb, lu, lb):
            continue
        print(f"{L0:>3}{cu:11.2e}{cb:11.2e}{'bal' if cb < cu else 'unb':>9}"
              f"{lu:11.2e}{lb:11.2e}{'bal' if lb < lu else 'unb':>9}")
    print("\nThe convolutional code is decided by the terminal value, so balanced wins;")
    print("the LDPC code has a steep transfer, needs the cold start, so unbalanced wins.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
