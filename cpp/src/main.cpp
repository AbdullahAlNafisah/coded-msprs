/*!
 * \file main.cpp
 * \brief BER driver for the AFF3CT MS-PRS chain.
 *
 * Three modes, each with a Python cache under notebooks/BER/outputs/ to match:
 *
 *   uncoded-msprs  Source -> MSPRS -> AWGN -> MSPRS -> hard
 *                  ref: nsm_L{L0}_{family}_uncoded
 *   coded-bpsk     Source -> NSC -> BPSK -> AWGN -> NSC BCJR
 *                  ref: ask2_conv_K3          (validates the outer code alone)
 *   coded-msprs    Source -> NSC -> interleave -> MSPRS -> AWGN
 *                  -> [tdemodulate <-> decode_siso] x iters -> hard
 *                  ref: nsm_L{L0}_{family}_conv_K3_7iters
 *
 * Every mode reports the SAME per-information-bit Eb/N0 abscissa the Python
 * uses, so curves from the two implementations are directly comparable. The
 * conversion is Es = m * Rc * Eb, i.e. Es/N0[dB] = Eb/N0[dB] + 10log10(m * Rc),
 * with m the bits per symbol and Rc the code rate:
 *
 *      uncoded-msprs   m = 2, Rc = 1     -> Es/N0 = Eb/N0 + 3.01 dB
 *      coded-bpsk      m = 1, Rc = 1/2   -> Es/N0 = Eb/N0 - 3.01 dB
 *      coded-msprs     m = 2, Rc = 1/2   -> Es/N0 = Eb/N0
 *
 * Leaving m at AFF3CT's BPSK default of 1 for an MS-PRS run shifts the whole
 * curve by 3.01 dB while every individual stage still looks correct.
 */
#include <array>
#include <chrono>
#include <memory>
#include <thread>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <aff3ct.hpp>

#include "MSPRS/Exit.hpp"
#include "MSPRS/Modem_FTN.hpp"
#include "MSPRS/Modem_MSPRS.hpp"
#include "MSPRS/NSC.hpp"
#include "MSPRS/Taps.hpp"

using namespace aff3ct;

namespace
{
struct params
{
    std::string mode      = "uncoded-msprs";
    int         L0        = 3;
    std::string family    = "balanced";
    int         K         = 0;      //!< 0 -> per-mode default
    int         iters     = 7;      //!< turbo iterations; the loop runs iters+1 passes
    int         be        = 500;    //!< bit errors before a point is done
    int         min_fra   = 20;
    int         max_fra   = 200000;
    int         seed      = 0;
    int         itl_seed  = 42;
    float       ebn0_min  = 0.00f;
    float       ebn0_max  = 12.01f;
    float       ebn0_step = 1.00f;
    std::string taps_dir  = "../../nsm/modem/filters";
    double      dt_sigma  = 1.0;   //!< sigma for the demodtest harness
    int         threads   = 1;     //!< independent chains run in parallel
    int         chunk     = 25;    //!< frames per thread between stop-rule checks
    int         L_isi     = 5;     //!< one-sided FTN ISI truncation
    int         n_trials  = 50;    //!< a-priori draws averaged per IA point
    int         n_ia      = 100;   //!< IA grid size
    int         exit_bits = 4998;  //!< info bits per EXIT realisation
    bool        exit_fresh_rx = false; //!< redraw the channel per trial
    int         bp_ite    = 20;    //!< BP iterations inside each turbo pass
    std::string ldpc_h    = "../.aff3ct/install/share/aff3ct-4.1.2/conf/dec/LDPC/MACKAY_4000_8000.alist";
};

void header(const params& p, const int N_in, const int N_mod, const double m, const double Rc)
{
    const std::string v = "v" + std::to_string(tools::version_major()) + "." +
                          std::to_string(tools::version_minor()) + "." +
                          std::to_string(tools::version_release());
    std::cout << "# MS-PRS on AFF3CT " << v << "   mode = " << p.mode << "\n"
              << "#   L0 / family   = " << p.L0 << " / " << p.family << "\n"
              << "#   info bits (K) = " << p.K << "\n"
              << "#   modem in  (N) = " << N_in << "\n"
              << "#   symbols       = " << N_mod << "\n"
              << "#   m, Rc         = " << m << ", " << Rc << "\n"
              << "#   turbo iters   = " << (p.mode == "coded-msprs" ? std::to_string(p.iters) : "-") << "\n#\n"
              << "#     Eb/N0 |     Es/N0 |        FRA |         BE |       BER |    s\n";
}

void row(const float ebn0, const float esn0, module::Monitor_BFER<>& mon, const double et)
{
    std::cout << std::fixed << std::setprecision(2) << std::setw(11) << ebn0 << " |" << std::setw(10) << esn0
              << " |" << std::setw(11) << mon.get_n_analyzed_fra() << " |" << std::setw(11) << mon.get_n_be()
              << " |" << std::scientific << std::setprecision(3) << std::setw(11) << mon.get_ber() << " |"
              << std::fixed << std::setprecision(1) << std::setw(5) << et << std::endl;
}
} // namespace

int main(int argc, char** argv)
{
    params p;
    for (int i = 1; i < argc; i++)
    {
        const std::string a   = argv[i];
        auto              nxt = [&]() { return argv[++i]; };
        if      (a == "--mode"      && i + 1 < argc) p.mode      = nxt();
        else if (a == "--L0"        && i + 1 < argc) p.L0        = std::atoi(nxt());
        else if (a == "--family"    && i + 1 < argc) p.family    = nxt();
        else if (a == "--K"         && i + 1 < argc) p.K         = std::atoi(nxt());
        else if (a == "--iters"     && i + 1 < argc) p.iters     = std::atoi(nxt());
        else if (a == "--be"        && i + 1 < argc) p.be        = std::atoi(nxt());
        else if (a == "--min-fra"   && i + 1 < argc) p.min_fra   = std::atoi(nxt());
        else if (a == "--max-fra"   && i + 1 < argc) p.max_fra   = std::atoi(nxt());
        else if (a == "--seed"      && i + 1 < argc) p.seed      = std::atoi(nxt());
        else if (a == "--taps"      && i + 1 < argc) p.taps_dir  = nxt();
        else if (a == "--sigma"     && i + 1 < argc) p.dt_sigma  = std::atof(nxt());
        else if (a == "--threads"   && i + 1 < argc) p.threads   = std::atoi(nxt());
        else if (a == "--chunk"     && i + 1 < argc) p.chunk     = std::atoi(nxt());
        else if (a == "--L-isi"     && i + 1 < argc) p.L_isi     = std::atoi(nxt());
        else if (a == "--n-trials"  && i + 1 < argc) p.n_trials  = std::atoi(nxt());
        else if (a == "--n-ia"      && i + 1 < argc) p.n_ia      = std::atoi(nxt());
        else if (a == "--exit-bits" && i + 1 < argc) p.exit_bits = std::atoi(nxt());
        else if (a == "--exit-fresh-rx") p.exit_fresh_rx = true;
        else if (a == "--bp-ite"    && i + 1 < argc) p.bp_ite    = std::atoi(nxt());
        else if (a == "--ldpc-h"    && i + 1 < argc) p.ldpc_h    = nxt();
        else if (a == "--ebn0-min"  && i + 1 < argc) p.ebn0_min  = (float)std::atof(nxt());
        else if (a == "--ebn0-max"  && i + 1 < argc) p.ebn0_max  = (float)std::atof(nxt());
        else if (a == "--ebn0-step" && i + 1 < argc) p.ebn0_step = (float)std::atof(nxt());
        else { std::cerr << "unknown argument: " << a << std::endl; return 1; }
    }

    // Parity harnesses, all reading stdin and printing one value per line. They
    // let the C++ be diffed against the Python on identical input, which is a
    // far sharper test than comparing two Monte-Carlo BER curves. Keep them:
    // every numerical claim about this implementation rests on them.
    //   modtest   bits          -> MS-PRS symbols        (vs nsm.modem.msprs.modulate)
    //   demodtest symbols       -> MS-PRS extrinsic LLRs (vs ...demodulate)
    //   enctest   bits          -> NSC codeword          (vs nsm.codec.conv.encode)
    //   dectest   coded-bit LLR -> NSC extrinsic LLRs    (vs ...decode)
    // tdemodtest: stdin is `n_sym`, then n_sym symbols, then N a-priori LLRs.
    // Exercises the path the turbo loop actually uses, which demodtest does not:
    // non-zero a-priori and the extrinsic subtraction.
    if (p.mode == "tdemodtest")
    {
        const auto t = msprs::load_taps(p.taps_dir, p.L0, p.family);
        int        n_sym = 0;
        std::cin >> n_sym;
        std::vector<float> y(n_sym);
        for (int i = 0; i < n_sym; i++) std::cin >> y[i];
        std::vector<float> la;
        double             v;
        while (std::cin >> v) la.push_back((float)v);

        msprs::Modem_MSPRS<>     mdm((int)la.size(), t);
        std::vector<float>       ext(la.size());
        const std::vector<float> CP = { (float)p.dt_sigma };
        mdm.tdemodulate(CP, y, la, ext);
        std::cout << std::setprecision(12);
        for (auto z : ext) std::cout << z << "\n";
        return 0;
    }

    // mitest: stdin is n, then n LLRs (PYTHON sign), then n bits. Prints the
    // three MI estimates, so the estimators can be diffed against _exit_jit.py
    // on identical input, independently of any RNG difference.
    /* MS-PRS with an LDPC outer code, turbo-equalised.
     *
     * Section VI names turbo-LDPC equalisation as future work, and the Python
     * could not do it: nsm/codec/ldpc.py runs a single BCJR-then-BP pass, which
     * cannot work for MS-PRS because the inner trellis needs the extrinsic
     * exchange. AFF3CT's LDPC BP decoders derive from Decoder_SISO and their
     * _decode_siso returns post - Y_N1, i.e. the extrinsic on the CODED bits,
     * which is exactly the interface the loop needs.
     *
     * Rate is K/N of the matrix; with the shipped MACKAY_4000_8000 that is 1/2,
     * matching the convolutional benchmark, so Es/N0 = Eb/N0 as before.
     */
    if (p.mode == "ldpc-msprs")
    {
        const auto taps = msprs::load_taps(p.taps_dir, p.L0, p.family);
        const auto H    = tools::LDPC_matrix_handler::read(p.ldpc_h);
        // Take the dimensions the way AFF3CT's own factory does
        // (Decoder_LDPC.cpp:149): N_cw is the width of H and K = N_cw - M.
        // Reading them off the Sparse_matrix instead gets the orientation
        // wrong and yields a negative K.
        int M = 0, N = 0;
        tools::LDPC_matrix_handler::read_matrix_size(p.ldpc_h, M, N);
        const int Kl = N - M;
        const int  Nm   = msprs::Modem_MSPRS<>::size_mod(N, p.L0);
        const int  nth  = std::max(1, p.threads);

        std::cout << "# MS-PRS + LDPC turbo equalisation\n"
                  << "#   H          = " << p.ldpc_h << "\n"
                  << "#   K, N, rate = " << Kl << ", " << N << ", " << (double)Kl / N << "\n"
                  << "#   L0/family  = " << p.L0 << " / " << p.family << "\n"
                  << "#   BP ite     = " << p.bp_ite << ",  turbo ite = " << p.iters << "\n#\n"
                  << "#     Eb/N0 |     Es/N0 |        FRA |         BE |       BER |    s\n";

        module::Encoder_LDPC_from_H<> enc0(Kl, N, H, "IDENTITY");
        const auto                    ibp = enc0.get_info_bits_pos();

        struct LChain
        {
            spu::module::Source_random<>              src;
            module::Encoder_LDPC_from_H<>             enc;
            module::Decoder_LDPC_BP_flooding_SPA<>    dec;
            msprs::Modem_MSPRS<>                      mdm;
            module::Channel_AWGN_LLR<>                chn;
            module::Interleaver<int>                  ib;
            module::Interleaver<float>                il;
            std::vector<int>      u, cw, cwi, hard;
            std::vector<uint32_t> ng;
            std::vector<float>    sym, rx, Lei, Lai, Len, Den;
            LChain(int K, int N, int Nm, int bp, const tools::Sparse_matrix& H,
                   const std::vector<uint32_t>& ibp, const msprs::Taps& t,
                   const tools::Interleaver_core_random<>& core, int seed, int stream)
              : src(K), enc(K, N, H, "IDENTITY"), dec(K, N, bp, H, ibp, false)
              , mdm(N, t), chn(Nm), ib(core), il(core)
              , u(K), cw(N), cwi(N), hard(K), ng(1)
              , sym(Nm), rx(Nm), Lei(N), Lai(N), Len(N), Den(N)
            { chn.set_seed(seed + 7919 * stream); src.set_seed(seed + 6271 * stream); }
        };

        tools::Interleaver_core_random<> core(N, p.itl_seed, false);
        std::vector<std::unique_ptr<LChain>> ch;
        for (int t = 0; t < nth; t++)
            ch.emplace_back(new LChain(Kl, N, Nm, p.bp_ite, H, ibp, taps, core, p.seed, t + 1));

        for (float ebn0 = p.ebn0_min; ebn0 < p.ebn0_max; ebn0 += p.ebn0_step)
        {
            const double Rl   = (double)Kl / (double)N;
            const float  esn0 = ebn0 + 10.0f * (float)std::log10(2.0 * Rl);
            const float  sig  = (float)std::sqrt(1.0 / (2.0 * std::pow(10.0, esn0 / 10.0)));
            const std::vector<float> CP = { sig };

            long long frames = 0, errs = 0;
            const auto t0 = std::chrono::steady_clock::now();
            while (frames < p.max_fra && (frames < p.min_fra || errs < p.be) &&
                   !spu::tools::Signal_handler::is_sigint())
            {
                std::vector<long long> pe(nth, 0), pf(nth, 0);
                std::vector<std::thread> pool;
                for (int t = 0; t < nth; t++)
                    pool.emplace_back([&, t] {
                        auto& c = *ch[t];
                        for (int f = 0; f < p.chunk; f++)
                        {
                            c.src.generate(c.u, c.ng);
                            c.enc.encode(c.u, c.cw);
                            c.ib .interleave(c.cw, c.cwi);
                            c.mdm.modulate(c.cwi, c.sym);
                            c.chn.add_noise(CP, c.sym, c.rx);
                            std::fill(c.Lai.begin(), c.Lai.end(), 0.f);
                            for (int it = 0; it <= p.iters; it++)
                            {
                                c.mdm.tdemodulate(CP, c.rx, c.Lai, c.Lei);
                                c.il .deinterleave(c.Lei, c.Len);
                                c.dec.decode_siso(c.Len, c.Den);
                                c.il .interleave(c.Den, c.Lai);
                            }
                            c.dec.decode_siho(c.Len, c.hard);
                            long long e = 0;
                            for (int i = 0; i < Kl; i++) e += (c.hard[i] != c.u[i]);
                            pe[t] += e; pf[t] += 1;
                        }
                    });
                for (auto& th : pool) th.join();
                for (int t = 0; t < nth; t++) { errs += pe[t]; frames += pf[t]; }
            }
            const double et = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            std::cout << std::fixed << std::setprecision(2) << std::setw(11) << ebn0 << " |"
                      << std::setw(10) << esn0 << " |" << std::setw(11) << frames << " |"
                      << std::setw(11) << errs << " |" << std::scientific << std::setprecision(3)
                      << std::setw(11) << (double)errs / (double)(frames * Kl) << " |" << std::fixed
                      << std::setprecision(1) << std::setw(5) << et << std::endl;
        }
        std::cout << "# done" << std::endl;
        return 0;
    }

    // EXIT sweep. Mirrors scripts/make_exit_data.py: one (bits, rx) realisation
    // per Eb/N0, reused across the whole IA grid, with N_TRIALS a-priori draws
    // averaged per IA point. Parallel over IA points, as the Python is via
    // prange. Emits one line per (snr, ia) for scripts/run_aff3ct_exit.py.
    if (p.mode == "exit")
    {
        const auto taps = msprs::load_taps(p.taps_dir, p.L0, p.family);
        const int  K    = p.exit_bits;
        const int  Ns   = msprs::Modem_MSPRS<>::size_mod(K, p.L0);
        const int  nth  = std::max(1, p.threads);

        std::vector<double> IA(p.n_ia);
        for (int i = 0; i < p.n_ia; i++)
            IA[i] = 0.999 * (1e-3 + (1.0 - 1e-3) * (double)i / (double)(p.n_ia - 1));

        std::cout << "# exit L0=" << p.L0 << " " << p.family << " trials=" << p.n_trials
                  << " n_ia=" << p.n_ia << " bits=" << K << "\n";

        std::vector<std::unique_ptr<msprs::Modem_MSPRS<>>> mdm;
        for (int t = 0; t < nth; t++) mdm.emplace_back(new msprs::Modem_MSPRS<>(K, taps));

        for (float ebn0 = p.ebn0_min; ebn0 < p.ebn0_max; ebn0 += p.ebn0_step)
        {
            // Es = 1 and m = 2 bits/symbol at Rc = 1/2, so Es/N0 = Eb/N0, the
            // same convention the BER sweep uses.
            const float sigma = (float)std::sqrt(1.0 / (2.0 * std::pow(10.0, ebn0 / 10.0)));
            const std::vector<float> CP = { sigma };

            std::mt19937_64 gen(0xE117u ^ (uint64_t)std::lround(ebn0 * 1000.0));
            std::vector<int> bits(K);
            for (int i = 0; i < K; i++) bits[i] = (int)(gen() & 1ull);
            std::vector<float> sym(Ns), rx(Ns);
            mdm[0]->modulate(bits, sym);
            std::normal_distribution<double> nd(0.0, 1.0);
            for (int i = 0; i < Ns; i++) rx[i] = sym[i] + (float)(sigma * nd(gen));

            std::vector<std::array<double, 4>> out((size_t)p.n_ia);
            std::vector<std::thread>           pool;
            for (int t = 0; t < nth; t++)
                pool.emplace_back([&, t] {
                    std::mt19937_64     rng(0xA5A5u + 7919ull * (uint64_t)t
                                            + 104729ull * (uint64_t)std::lround(ebn0 * 1000.0));
                    std::vector<double> la(K), ext(K);
                    std::vector<float>  laf(K), extf(K);
                    std::vector<int>    lb(K);
                    std::vector<float>  lsym(Ns), lrx(Ns);
                    std::normal_distribution<double> lnd(0.0, 1.0);
                    for (int k = t; k < p.n_ia; k += nth)
                    {
                        const double sa = msprs::i_inv(IA[k]);
                        double a = 0, h = 0, m = 0, iam = 0;
                        for (int tr = 0; tr < p.n_trials; tr++)
                        {
                            // A single channel realisation per Eb/N0, as the
                            // Python does, leaves its own noise in the estimate:
                            // averaging over a-priori draws cannot remove it,
                            // because every trial sees the same rx. Redrawing it
                            // per trial costs one modulate and averages that
                            // component away too.
                            const std::vector<int>*   pb = &bits;
                            const std::vector<float>* pr = &rx;
                            if (p.exit_fresh_rx)
                            {
                                for (int i = 0; i < K; i++) lb[i] = (int)(rng() & 1ull);
                                mdm[t]->modulate(lb, lsym);
                                for (int i = 0; i < Ns; i++) lrx[i] = lsym[i] + (float)(sigma * lnd(rng));
                                pb = &lb; pr = &lrx;
                            }
                            msprs::gen_llrs(*pb, sa, rng, la);            // Python sign
                            if (tr == 0) iam = msprs::mi_avg(la, *pb);
                            for (int i = 0; i < K; i++) laf[i] = (float)(-la[i]);  // -> AFF3CT sign
                            mdm[t]->tdemodulate(CP, *pr, laf, extf);
                            for (int i = 0; i < K; i++) ext[i] = -(double)extf[i]; // -> Python sign
                            a += msprs::mi_avg(ext, *pb);
                            h += msprs::mi_hist(ext, *pb);
                            m += msprs::mi_mag(ext);
                        }
                        out[(size_t)k] = { a / p.n_trials, h / p.n_trials, m / p.n_trials, iam };
                    }
                });
            for (auto& th : pool) th.join();

            std::cout << std::setprecision(10);
            for (int k = 0; k < p.n_ia; k++)
                std::cout << "E " << ebn0 << " " << IA[k] << " " << out[(size_t)k][0] << " "
                          << out[(size_t)k][1] << " " << out[(size_t)k][2] << " "
                          << out[(size_t)k][3] << "\n";
            std::cout << std::flush;
        }
        std::cout << "# done" << std::endl;
        return 0;
    }

    if (p.mode == "mitest")
    {
        int n = 0; std::cin >> n;
        std::vector<double> llr(n); std::vector<int> bits(n);
        for (int i = 0; i < n; i++) std::cin >> llr[i];
        for (int i = 0; i < n; i++) std::cin >> bits[i];
        std::cout << std::setprecision(15)
                  << msprs::mi_avg(llr, bits) << "\n"
                  << msprs::mi_hist(llr, bits) << "\n"
                  << msprs::mi_mag(llr) << "\n";
        return 0;
    }

    if (p.mode == "iinv")
    {
        double v; std::cout << std::setprecision(15);
        while (std::cin >> v) std::cout << msprs::i_inv(v) << "\n";
        return 0;
    }

    if (p.mode == "ftn-taps" || p.mode == "ftn-modtest" || p.mode == "ftn-demodtest")
    {
        std::cout << std::setprecision(12);
        // mitest: stdin is n, then n LLRs (PYTHON sign), then n bits. Prints the
    // three MI estimates, so the estimators can be diffed against _exit_jit.py
    // on identical input, independently of any RNG difference.
    /* MS-PRS with an LDPC outer code, turbo-equalised.
     *
     * Section VI names turbo-LDPC equalisation as future work, and the Python
     * could not do it: nsm/codec/ldpc.py runs a single BCJR-then-BP pass, which
     * cannot work for MS-PRS because the inner trellis needs the extrinsic
     * exchange. AFF3CT's LDPC BP decoders derive from Decoder_SISO and their
     * _decode_siso returns post - Y_N1, i.e. the extrinsic on the CODED bits,
     * which is exactly the interface the loop needs.
     *
     * Rate is K/N of the matrix; with the shipped MACKAY_4000_8000 that is 1/2,
     * matching the convolutional benchmark, so Es/N0 = Eb/N0 as before.
     */
    if (p.mode == "ldpc-msprs")
    {
        const auto taps = msprs::load_taps(p.taps_dir, p.L0, p.family);
        const auto H    = tools::LDPC_matrix_handler::read(p.ldpc_h);
        // Take the dimensions the way AFF3CT's own factory does
        // (Decoder_LDPC.cpp:149): N_cw is the width of H and K = N_cw - M.
        // Reading them off the Sparse_matrix instead gets the orientation
        // wrong and yields a negative K.
        int M = 0, N = 0;
        tools::LDPC_matrix_handler::read_matrix_size(p.ldpc_h, M, N);
        const int Kl = N - M;
        const int  Nm   = msprs::Modem_MSPRS<>::size_mod(N, p.L0);
        const int  nth  = std::max(1, p.threads);

        std::cout << "# MS-PRS + LDPC turbo equalisation\n"
                  << "#   H          = " << p.ldpc_h << "\n"
                  << "#   K, N, rate = " << Kl << ", " << N << ", " << (double)Kl / N << "\n"
                  << "#   L0/family  = " << p.L0 << " / " << p.family << "\n"
                  << "#   BP ite     = " << p.bp_ite << ",  turbo ite = " << p.iters << "\n#\n"
                  << "#     Eb/N0 |     Es/N0 |        FRA |         BE |       BER |    s\n";

        module::Encoder_LDPC_from_H<> enc0(Kl, N, H, "IDENTITY");
        const auto                    ibp = enc0.get_info_bits_pos();

        struct LChain
        {
            spu::module::Source_random<>              src;
            module::Encoder_LDPC_from_H<>             enc;
            module::Decoder_LDPC_BP_flooding_SPA<>    dec;
            msprs::Modem_MSPRS<>                      mdm;
            module::Channel_AWGN_LLR<>                chn;
            module::Interleaver<int>                  ib;
            module::Interleaver<float>                il;
            std::vector<int>      u, cw, cwi, hard;
            std::vector<uint32_t> ng;
            std::vector<float>    sym, rx, Lei, Lai, Len, Den;
            LChain(int K, int N, int Nm, int bp, const tools::Sparse_matrix& H,
                   const std::vector<uint32_t>& ibp, const msprs::Taps& t,
                   const tools::Interleaver_core_random<>& core, int seed, int stream)
              : src(K), enc(K, N, H, "IDENTITY"), dec(K, N, bp, H, ibp, false)
              , mdm(N, t), chn(Nm), ib(core), il(core)
              , u(K), cw(N), cwi(N), hard(K), ng(1)
              , sym(Nm), rx(Nm), Lei(N), Lai(N), Len(N), Den(N)
            { chn.set_seed(seed + 7919 * stream); src.set_seed(seed + 6271 * stream); }
        };

        tools::Interleaver_core_random<> core(N, p.itl_seed, false);
        std::vector<std::unique_ptr<LChain>> ch;
        for (int t = 0; t < nth; t++)
            ch.emplace_back(new LChain(Kl, N, Nm, p.bp_ite, H, ibp, taps, core, p.seed, t + 1));

        for (float ebn0 = p.ebn0_min; ebn0 < p.ebn0_max; ebn0 += p.ebn0_step)
        {
            const double Rl   = (double)Kl / (double)N;
            const float  esn0 = ebn0 + 10.0f * (float)std::log10(2.0 * Rl);
            const float  sig  = (float)std::sqrt(1.0 / (2.0 * std::pow(10.0, esn0 / 10.0)));
            const std::vector<float> CP = { sig };

            long long frames = 0, errs = 0;
            const auto t0 = std::chrono::steady_clock::now();
            while (frames < p.max_fra && (frames < p.min_fra || errs < p.be) &&
                   !spu::tools::Signal_handler::is_sigint())
            {
                std::vector<long long> pe(nth, 0), pf(nth, 0);
                std::vector<std::thread> pool;
                for (int t = 0; t < nth; t++)
                    pool.emplace_back([&, t] {
                        auto& c = *ch[t];
                        for (int f = 0; f < p.chunk; f++)
                        {
                            c.src.generate(c.u, c.ng);
                            c.enc.encode(c.u, c.cw);
                            c.ib .interleave(c.cw, c.cwi);
                            c.mdm.modulate(c.cwi, c.sym);
                            c.chn.add_noise(CP, c.sym, c.rx);
                            std::fill(c.Lai.begin(), c.Lai.end(), 0.f);
                            for (int it = 0; it <= p.iters; it++)
                            {
                                c.mdm.tdemodulate(CP, c.rx, c.Lai, c.Lei);
                                c.il .deinterleave(c.Lei, c.Len);
                                c.dec.decode_siso(c.Len, c.Den);
                                c.il .interleave(c.Den, c.Lai);
                            }
                            c.dec.decode_siho(c.Len, c.hard);
                            long long e = 0;
                            for (int i = 0; i < Kl; i++) e += (c.hard[i] != c.u[i]);
                            pe[t] += e; pf[t] += 1;
                        }
                    });
                for (auto& th : pool) th.join();
                for (int t = 0; t < nth; t++) { errs += pe[t]; frames += pf[t]; }
            }
            const double et = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            std::cout << std::fixed << std::setprecision(2) << std::setw(11) << ebn0 << " |"
                      << std::setw(10) << esn0 << " |" << std::setw(11) << frames << " |"
                      << std::setw(11) << errs << " |" << std::scientific << std::setprecision(3)
                      << std::setw(11) << (double)errs / (double)(frames * Kl) << " |" << std::fixed
                      << std::setprecision(1) << std::setw(5) << et << std::endl;
        }
        std::cout << "# done" << std::endl;
        return 0;
    }

    // EXIT sweep. Mirrors scripts/make_exit_data.py: one (bits, rx) realisation
    // per Eb/N0, reused across the whole IA grid, with N_TRIALS a-priori draws
    // averaged per IA point. Parallel over IA points, as the Python is via
    // prange. Emits one line per (snr, ia) for scripts/run_aff3ct_exit.py.
    if (p.mode == "exit")
    {
        const auto taps = msprs::load_taps(p.taps_dir, p.L0, p.family);
        const int  K    = p.exit_bits;
        const int  Ns   = msprs::Modem_MSPRS<>::size_mod(K, p.L0);
        const int  nth  = std::max(1, p.threads);

        std::vector<double> IA(p.n_ia);
        for (int i = 0; i < p.n_ia; i++)
            IA[i] = 0.999 * (1e-3 + (1.0 - 1e-3) * (double)i / (double)(p.n_ia - 1));

        std::cout << "# exit L0=" << p.L0 << " " << p.family << " trials=" << p.n_trials
                  << " n_ia=" << p.n_ia << " bits=" << K << "\n";

        std::vector<std::unique_ptr<msprs::Modem_MSPRS<>>> mdm;
        for (int t = 0; t < nth; t++) mdm.emplace_back(new msprs::Modem_MSPRS<>(K, taps));

        for (float ebn0 = p.ebn0_min; ebn0 < p.ebn0_max; ebn0 += p.ebn0_step)
        {
            // Es = 1 and m = 2 bits/symbol at Rc = 1/2, so Es/N0 = Eb/N0, the
            // same convention the BER sweep uses.
            const float sigma = (float)std::sqrt(1.0 / (2.0 * std::pow(10.0, ebn0 / 10.0)));
            const std::vector<float> CP = { sigma };

            std::mt19937_64 gen(0xE117u ^ (uint64_t)std::lround(ebn0 * 1000.0));
            std::vector<int> bits(K);
            for (int i = 0; i < K; i++) bits[i] = (int)(gen() & 1ull);
            std::vector<float> sym(Ns), rx(Ns);
            mdm[0]->modulate(bits, sym);
            std::normal_distribution<double> nd(0.0, 1.0);
            for (int i = 0; i < Ns; i++) rx[i] = sym[i] + (float)(sigma * nd(gen));

            std::vector<std::array<double, 4>> out((size_t)p.n_ia);
            std::vector<std::thread>           pool;
            for (int t = 0; t < nth; t++)
                pool.emplace_back([&, t] {
                    std::mt19937_64     rng(0xA5A5u + 7919ull * (uint64_t)t
                                            + 104729ull * (uint64_t)std::lround(ebn0 * 1000.0));
                    std::vector<double> la(K), ext(K);
                    std::vector<float>  laf(K), extf(K);
                    std::vector<int>    lb(K);
                    std::vector<float>  lsym(Ns), lrx(Ns);
                    std::normal_distribution<double> lnd(0.0, 1.0);
                    for (int k = t; k < p.n_ia; k += nth)
                    {
                        const double sa = msprs::i_inv(IA[k]);
                        double a = 0, h = 0, m = 0, iam = 0;
                        for (int tr = 0; tr < p.n_trials; tr++)
                        {
                            // A single channel realisation per Eb/N0, as the
                            // Python does, leaves its own noise in the estimate:
                            // averaging over a-priori draws cannot remove it,
                            // because every trial sees the same rx. Redrawing it
                            // per trial costs one modulate and averages that
                            // component away too.
                            const std::vector<int>*   pb = &bits;
                            const std::vector<float>* pr = &rx;
                            if (p.exit_fresh_rx)
                            {
                                for (int i = 0; i < K; i++) lb[i] = (int)(rng() & 1ull);
                                mdm[t]->modulate(lb, lsym);
                                for (int i = 0; i < Ns; i++) lrx[i] = lsym[i] + (float)(sigma * lnd(rng));
                                pb = &lb; pr = &lrx;
                            }
                            msprs::gen_llrs(*pb, sa, rng, la);            // Python sign
                            if (tr == 0) iam = msprs::mi_avg(la, *pb);
                            for (int i = 0; i < K; i++) laf[i] = (float)(-la[i]);  // -> AFF3CT sign
                            mdm[t]->tdemodulate(CP, *pr, laf, extf);
                            for (int i = 0; i < K; i++) ext[i] = -(double)extf[i]; // -> Python sign
                            a += msprs::mi_avg(ext, *pb);
                            h += msprs::mi_hist(ext, *pb);
                            m += msprs::mi_mag(ext);
                        }
                        out[(size_t)k] = { a / p.n_trials, h / p.n_trials, m / p.n_trials, iam };
                    }
                });
            for (auto& th : pool) th.join();

            std::cout << std::setprecision(10);
            for (int k = 0; k < p.n_ia; k++)
                std::cout << "E " << ebn0 << " " << IA[k] << " " << out[(size_t)k][0] << " "
                          << out[(size_t)k][1] << " " << out[(size_t)k][2] << " "
                          << out[(size_t)k][3] << "\n";
            std::cout << std::flush;
        }
        std::cout << "# done" << std::endl;
        return 0;
    }

    if (p.mode == "mitest")
    {
        int n = 0; std::cin >> n;
        std::vector<double> llr(n); std::vector<int> bits(n);
        for (int i = 0; i < n; i++) std::cin >> llr[i];
        for (int i = 0; i < n; i++) std::cin >> bits[i];
        std::cout << std::setprecision(15)
                  << msprs::mi_avg(llr, bits) << "\n"
                  << msprs::mi_hist(llr, bits) << "\n"
                  << msprs::mi_mag(llr) << "\n";
        return 0;
    }

    if (p.mode == "iinv")
    {
        double v; std::cout << std::setprecision(15);
        while (std::cin >> v) std::cout << msprs::i_inv(v) << "\n";
        return 0;
    }

    if (p.mode == "ftn-taps")
        {
            for (auto v : msprs::ftn_isi(0.5, 0.3, p.L_isi)) std::cout << v << "\n";
            return 0;
        }
        if (p.mode == "ftn-modtest")
        {
            std::vector<int> bits; int b;
            while (std::cin >> b) bits.push_back(b);
            msprs::Modem_FTN<> m((int)bits.size(), 0.5, 0.3, p.L_isi);
            std::vector<float> out(msprs::Modem_FTN<>::size_mod((int)bits.size(), p.L_isi));
            m.modulate(bits, out);
            for (auto v : out) std::cout << v << "\n";
            return 0;
        }
        std::vector<float> y; double v;
        while (std::cin >> v) y.push_back((float)v);
        const int Nb = (int)y.size() - p.L_isi + 1;
        msprs::Modem_FTN<> m(Nb, 0.5, 0.3, p.L_isi);
        std::vector<float> llr(Nb);
        const std::vector<float> CP = { (float)p.dt_sigma };
        m.demodulate(CP, y, llr);
        for (auto z : llr) std::cout << z << "\n";
        return 0;
    }

    if (p.mode == "modtest" || p.mode == "demodtest")
    {
        const auto t = msprs::load_taps(p.taps_dir, p.L0, p.family);
        std::cout << std::setprecision(12);
        if (p.mode == "modtest")
        {
            std::vector<int> bits;
            int              b;
            while (std::cin >> b) bits.push_back(b);
            msprs::Modem_MSPRS<> mdm((int)bits.size(), t);
            std::vector<float>   out(msprs::Modem_MSPRS<>::size_mod((int)bits.size(), p.L0));
            mdm.modulate(bits, out);
            for (auto v : out) std::cout << v << "\n";
        }
        else
        {
            std::vector<float> y;
            double             v;
            while (std::cin >> v) y.push_back((float)v);
            const int Nb = 2 * (int)y.size() - (p.L0 - 1) - (p.L0 % 2 == 0 ? 1 : 0);
            msprs::Modem_MSPRS<>     mdm(Nb, t);
            std::vector<float>       llr(Nb);
            const std::vector<float> CP = { (float)p.dt_sigma };
            mdm.demodulate(CP, y, llr);
            for (auto z : llr) std::cout << z << "\n";
        }
        return 0;
    }

    if (p.mode == "enctest" || p.mode == "dectest")
    {
        const msprs::NSC_Trellis tr(3, { 5, 7 });
        std::cout << std::setprecision(12);
        if (p.mode == "enctest")
        {
            std::vector<int> u;
            int              b;
            while (std::cin >> b) u.push_back(b);
            msprs::Encoder_NSC<> enc((int)u.size(), tr);
            std::vector<int>     x(tr.codeword_length((int)u.size()));
            enc.encode(u, x);
            for (auto v : x) std::cout << v << "\n";
        }
        else
        {
            std::vector<float> lin;
            double             v;
            while (std::cin >> v) lin.push_back((float)v);
            const int K = (int)lin.size() / tr.n_out - tr.memory;
            msprs::Decoder_NSC_SISO<> d(K, tr);
            std::vector<float>        ext(lin.size());
            std::vector<int>          hard(K);
            d.decode_both(lin.data(), ext.data(), hard.data());
            for (auto z : ext) std::cout << z << "\n";
        }
        return 0;
    }

    const bool uncoded = (p.mode == "uncoded-msprs");
    const bool bpsk    = (p.mode == "coded-bpsk");
    const bool ftn     = (p.mode == "coded-ftn");
    const bool coded   = (p.mode == "coded-msprs") || ftn;
    if (!uncoded && !bpsk && !coded) { std::cerr << "unknown mode: " << p.mode << std::endl; return 1; }
    if (p.K == 0) p.K = uncoded ? 9998 : 4998;

    const msprs::NSC_Trellis trellis(3, { 5, 7 });
    const int  N_in  = uncoded ? p.K : trellis.codeword_length(p.K); // bits into the modem
    const auto taps  = msprs::load_taps(p.taps_dir, p.L0, p.family);
    const int  N_mod = bpsk ? N_in
                     : ftn  ? msprs::Modem_FTN<>::size_mod(N_in, p.L_isi)
                            : msprs::Modem_MSPRS<>::size_mod(N_in, p.L0);

    // The Python passes rate = 0.5 rather than the exact K/N = 0.4998, so the
    // abscissae line up only if we do the same.
    // FTN carries ONE bit per symbol with ||h||^2 = 1, so m = 1 like BPSK, not
    // the 2 of rate-2 MS-PRS. Using 2 here would hand FTN a spurious 3 dB and
    // let the uncoded curve beat the ISI-free bound.
    const double m  = (bpsk || ftn) ? 1.0 : 2.0;
    const double Rc = uncoded ? 1.0 : 0.5;
    header(p, N_in, N_mod, m, Rc);

    // One Chain per thread. Every module here holds mutable scratch (the BCJR
    // alpha/beta/gamma buffers above all), so sharing one across threads would
    // race silently and produce plausible-looking garbage. Frames are
    // independent Monte-Carlo trials, so this is embarrassingly parallel and,
    // crucially, numerically identical to the serial loop: the bit-exact
    // harnesses stay valid because no arithmetic changes, only how many frames
    // run at once. Each chain gets its own RNG stream.
    struct Chain
    {
        spu::module::Source_random<>     source;
        msprs::Encoder_NSC<>             encoder;
        msprs::Decoder_NSC_SISO<>        decoder;
        msprs::Modem_MSPRS<>             msprs_m;
        module::Modem_BPSK<>             bpsk_m;
        msprs::Modem_FTN<>               ftn_m;
        module::Channel_AWGN_LLR<>       channel;
        module::Interleaver<int>         itl_b;
        module::Interleaver<float>       itl_l;
        std::vector<int>                 ref, dec, cw, cw_i;
        std::vector<uint32_t>            n_gen;
        std::vector<float>               sym, rx, Le_i, La_i, Le_n, De_n;

        Chain(const params& p, const msprs::NSC_Trellis& tr, const msprs::Taps& taps,
              int N_in, int N_mod, const tools::Interleaver_core_random<>& core, int stream)
          : source(p.K), encoder(p.K, tr), decoder(p.K, tr), msprs_m(N_in, taps)
          , bpsk_m(N_in), ftn_m(N_in, 0.5, 0.3, p.L_isi), channel(N_mod), itl_b(core), itl_l(core)
          , ref(p.K), dec(p.K), cw(N_in), cw_i(N_in), n_gen(1)
          , sym(N_mod), rx(N_mod), Le_i(N_in), La_i(N_in), Le_n(N_in), De_n(N_in)
        {
            channel.set_seed(p.seed + 7919 * stream);
            source .set_seed(p.seed + 6271 * stream);
        }
    };

    struct Stats
    {
        long long              frames = 0, bits = 0, errors = 0, frame_errors = 0;
        std::vector<long long> per_iter;
    };

    const int n_threads = std::max(1, p.threads);
    tools::Interleaver_core_random<> itl_core(N_in, p.itl_seed, false);

    std::vector<std::unique_ptr<Chain>> chains;
    for (int t = 0; t < n_threads; t++)
        chains.emplace_back(new Chain(p, trellis, taps, N_in, N_mod, itl_core, t + 1));

    auto run_frames = [&](Chain& c, int n, const std::vector<float>& CP, Stats& st)
    {
        for (int f = 0; f < n; f++)
        {
            c.source.generate(c.ref, c.n_gen);

            if (uncoded)
            {
                c.msprs_m.modulate  (c.ref, c.sym);
                c.channel.add_noise (CP, c.sym, c.rx);
                c.msprs_m.demodulate(CP, c.rx, c.Le_i);
                for (int i = 0; i < p.K; i++) c.dec[i] = (c.Le_i[i] < 0.f) ? 1 : 0;
            }
            else if (bpsk)
            {
                c.encoder.encode     (c.ref, c.cw);
                c.bpsk_m .modulate   (c.cw, c.sym);
                c.channel.add_noise  (CP, c.sym, c.rx);
                c.bpsk_m .demodulate (CP, c.rx, c.Le_n);
                c.decoder.decode_both(c.Le_n.data(), nullptr, c.dec.data());
            }
            else
            {
                c.encoder.encode    (c.ref, c.cw);
                c.itl_b  .interleave(c.cw, c.cw_i);
                if (ftn) c.ftn_m  .modulate(c.cw_i, c.sym);
                else     c.msprs_m.modulate(c.cw_i, c.sym);
                c.channel.add_noise (CP, c.sym, c.rx);

                std::fill(c.La_i.begin(), c.La_i.end(), 0.f);
                for (int it = 0; it <= p.iters; it++)
                {
                    if (ftn) c.ftn_m  .tdemodulate(CP, c.rx, c.La_i, c.Le_i);
                    else     c.msprs_m.tdemodulate(CP, c.rx, c.La_i, c.Le_i);
                    c.itl_l  .deinterleave(c.Le_i, c.Le_n);
                    c.decoder.decode_both (c.Le_n.data(), c.De_n.data(), c.dec.data());
                    c.itl_l  .interleave  (c.De_n, c.La_i);

                    // Fig 6 plots BER against turbo iteration, so the
                    // intermediate hard decisions have to be counted too.
                    long long e = 0;
                    for (int i = 0; i < p.K; i++) e += (c.dec[i] != c.ref[i]);
                    st.per_iter[it] += e;
                }
            }

            long long e = 0;
            for (int i = 0; i < p.K; i++) e += (c.dec[i] != c.ref[i]);
            st.errors += e;
            st.frame_errors += (e > 0);
            st.frames += 1;
            st.bits   += p.K;
        }
    };

    for (float ebn0 = p.ebn0_min; ebn0 < p.ebn0_max; ebn0 += p.ebn0_step)
    {
        const float              esn0  = ebn0 + 10.0f * (float)std::log10(m * Rc);
        const float              sigma = (float)std::sqrt(1.0 / (2.0 * std::pow(10.0, esn0 / 10.0)));
        const std::vector<float> CP    = { sigma };

        Stats total; total.per_iter.assign((size_t)p.iters + 1, 0);
        const auto t0 = std::chrono::steady_clock::now();

        // Chunked rounds rather than one long parallel region: the stopping
        // rule depends on the accumulated error count, so it has to be
        // re-evaluated between rounds. The chunk is per thread.
        while (total.frames < p.max_fra &&
               (total.frames < p.min_fra || total.errors < p.be) &&
               !spu::tools::Signal_handler::is_sigint())
        {
            const int chunk = std::max(1, std::min(p.chunk,
                                  (int)((p.max_fra - total.frames + n_threads - 1) / n_threads)));
            std::vector<Stats>       part(n_threads);
            std::vector<std::thread> pool;
            for (int t = 0; t < n_threads; t++)
            {
                part[t].per_iter.assign((size_t)p.iters + 1, 0);
                pool.emplace_back([&, t] { run_frames(*chains[t], chunk, CP, part[t]); });
            }
            for (auto& th : pool) th.join();
            for (auto& st : part)
            {
                total.frames += st.frames; total.bits += st.bits;
                total.errors += st.errors; total.frame_errors += st.frame_errors;
                for (size_t i = 0; i < st.per_iter.size(); i++) total.per_iter[i] += st.per_iter[i];
            }
        }

        const double et  = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        const double ber = total.bits ? (double)total.errors / (double)total.bits : 0.0;
        std::cout << std::fixed << std::setprecision(2) << std::setw(11) << ebn0 << " |" << std::setw(10)
                  << esn0 << " |" << std::setw(11) << total.frames << " |" << std::setw(11) << total.errors
                  << " |" << std::scientific << std::setprecision(3) << std::setw(11) << ber << " |"
                  << std::fixed << std::setprecision(1) << std::setw(5) << et << std::endl;
        if (coded)
        {
            std::cout << "# per-iter " << ebn0 << " " << total.bits;
            for (auto e : total.per_iter) std::cout << " " << e;
            std::cout << std::endl;
        }
    }
    std::cout << "# done" << std::endl;
    return 0;
}
