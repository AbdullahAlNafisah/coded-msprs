# coded-msprs

Turbo-equalized **Multi-Stream Partial Response Signaling (MS-PRS)** at rate 2:
a C++ simulation chain built on [AFF3CT](https://github.com/aff3ct/aff3ct), the
result caches it produced, and the scripts that render them into figures.

MS-PRS is the one-dimensional, convolutional sub-family of Nyquist Signaling
Modulations. It signals at the Nyquist rate and introduces controlled
inter-symbol interference digitally, through short FIR filters across two
bipolar sub-streams, rather than by compressing the symbol period as
faster-than-Nyquist signaling does. The rate-2 case carries two bits per channel
use, the spectral efficiency of 4-ASK on one quadrature.

The MS-PRS scheme itself, its filter tap families and their MSED optimisation
are due to Siala, Al-Nafisah and Al-Naffouri, *Nyquist Signaling Modulation
(NSM): An FTN-Inspired Paradigm Shift in Modulation Design for 6G and Beyond*,
[arXiv:2511.08553](https://arxiv.org/abs/2511.08553). This repository
accompanies a paper contributing the coded turbo-equalized receiver, its EXIT
convergence analysis and the simulated coded BER. Please cite the paper rather
than the code; `CITATION.cff` carries both.

It is a **reproducibility artifact**, not a library. Issues reporting a
reproduction failure are welcome. It is not accepting feature contributions,
and no API stability is promised.

The receiver is a turbo loop: a BCJR equalizer on the `2^(L0-1)`-state MS-PRS
trellis exchanges extrinsic log-likelihood ratios with an outer SISO decoder,
either a rate-1/2 convolutional code or a rate-1/2 LDPC code.

## Layout

```
cpp/          C++ simulation chain (AFF3CT v4.1.2)
  include/MSPRS/   Modem_MSPRS, Modem_FTN, NSC encoder/decoder, EXIT kernels
  src/main.cpp     multi-mode driver
results/
  ber/        399 simulated BER points, one JSON per scheme per Eb/N0
  exit/       12 EXIT characteristic caches
nsm/          Python reference implementation and the filter tap tables
scripts/      figure rendering, cache generation, analysis
figures/      the rendered figures, so results are visible without a run
tutorials/    eight teaching notebooks, from interleavers to filter families
```

The published package is simulation-only. The hardware and over-the-air layer
of the wider project is deliberately not part of this artifact.

## Building the simulator

Requires a C++14 compiler, CMake, and AFF3CT v4.1.2 (commit `60e113d`) built as
a static library.

```
cd aff3ct && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DAFF3CT_COMPILE_EXE=OFF -DAFF3CT_COMPILE_STATIC_LIB=ON
cmake --build . -j && cmake --install . --prefix ../install

cd /path/to/coded-msprs/cpp && mkdir build && cd build
cmake .. && cmake --build . -j
```

The driver exposes several modes:

```
./bin/msprs --mode coded-msprs  --L0 3 --family balanced --iters 7
./bin/msprs --mode ldpc-msprs   --L0 3 --family unbalanced --ldpc-h <matrix.alist>
./bin/msprs --mode uncoded-msprs --L0 4 --family unbalanced
./bin/msprs --mode exit         --L0 3 --family balanced
```

Parity harnesses (`--mode modtest`, `demodtest`, `tdemodtest`, `enctest`,
`dectest`, `mitest`) diff the C++ against the Python on identical input. The
modulator, both BCJRs and the encoder agree to better than 1e-6.

## Reproducing the figures

```
pip install -e .
python scripts/make_exit_figure.py       # modem and decoder EXIT characteristics
python scripts/make_ber_overview.py      # coded BER, both energy families
python scripts/make_ber_convergence.py   # BER against turbo iteration index
```

Each reads only `results/`, so the figures regenerate without rerunning any
simulation.

## Result cache format

One JSON per `(scheme, Eb/N0)` under `results/ber/<scheme>/snr_<x>dB.json`:

| Field | Meaning |
|---|---|
| `eb_no_db` | per-information-bit Eb/N0, the common abscissa across all schemes |
| `ber`, `ers_cnt`, `bits_cnt`, `n_frames` | error rate and the sample supporting it |
| `ber_lo95`, `ber_ub95`, `is_upper_bound` | Clopper-Pearson interval. A zero-error point is stored as a bound, never as `ber = 0` |
| `config.metric_convention` | provenance stamp; the figure scripts refuse to mix conventions in one plot |
| `implementation` | which of the two validated implementations produced the point |

**The stored intervals are not valid for the LDPC caches.** They are binomial
intervals on bit errors and assume independent trials. That holds for the
convolutional and ASK schemes, whose errors are scattered bits. Near an LDPC
threshold the chain fails in whole frames of a few hundred bits, so a point
carrying a few thousand bit errors rests on of order ten independent frame
failures and the interval understates its spread by roughly an order of
magnitude. Treat LDPC margins as good to about 0.1 dB.

## Energy and rate conventions

Every curve shares one per-information-bit Eb/N0 axis, with `Es = m*Rc*Eb`.
Rate-2 MS-PRS and 4-ASK carry `m = 2` coded bits per symbol, 2-ASK carries
`m = 1`, and all use `Rc = 1/2`. Getting this wrong shifts a scheme by 3 dB
while each piece still looks individually correct.

The JSON `eta` field is the energy on the FIR sub-stream. The loader applies
`sqrt(eta/5)*h0` and `sqrt((5-eta)/5)*h1`, with total symbol energy `Es = 5`.

## A note on the FTN module

`nsm/modem/ftn.py` and `cpp/include/MSPRS/Modem_FTN.hpp` are **not a valid FTN
model** and are retained only as an idealised ISI reference. They normalise the
one-sided pulse autocorrelation to unit energy and drive it with white noise,
which pins the isolated-error squared distance to 4, the ISI-free 2-ASK value,
independently of the packing factor. The true minimum squared Euclidean distance
at `tau = 0.5` is 2.03, a 2.95 dB deficit; `scripts/ftn_msed.py` measures both
and validates itself by reproducing the Mazo limit. A correct receiver needs the
Ungerboeck observation model with noise covariance `sigma^2 * G`.

## Analysis scripts

- `verify_paper_claims.py` recomputes the reported margins and distance tables
  from the caches and exits non-zero on disagreement
- `ftn_msed.py` minimum squared Euclidean distance of the FTN benchmark channel
- `ftn_whiten.py` shows no stable whitening filter exists at `tau = 0.5`
- `msed_multiplicity.py` multiplicity of minimum-distance error events
- `exit_regimes.py` bootstrap against terminal EXIT ordering for the two families

## Author

Abdullah Al-Nafisah, King Abdullah University of Science and Technology.

The modulation scheme is due to Mohamed Siala, Abdullah Al-Nafisah and Tareq
Al-Naffouri; see the preferred citation in `CITATION.cff`.

Licensed under Apache 2.0. AFF3CT is a separate project under the MIT licence.
