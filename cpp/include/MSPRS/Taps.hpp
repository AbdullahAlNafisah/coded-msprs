/*!
 * \file Taps.hpp
 * \brief MS-PRS filter taps, read from the same JSON the Python package uses.
 *
 * There is deliberately no second copy of the tap tables. `nsm/modem/filters/`
 * is the single source of truth for both implementations, so a re-export via
 * `scripts/reexport_filter_jsons.py` reaches the C++ without a manual edit.
 */
#ifndef MSPRS_TAPS_HPP
#define MSPRS_TAPS_HPP

#include <cmath>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace msprs
{

struct Taps
{
    int                 L0;
    std::vector<double> h0;   //!< energy-scaled FIR taps of the memory stream
    double              h1;   //!< energy-scaled scalar tap of the memoryless stream
};

/*!
 * \brief Load and energy-normalise the taps for one (L0, family).
 *
 * The JSON stores unit-norm `h0` and `h1` plus `eta`, where **eta is eta_0, the
 * energy on the FIR stream**, on an Es=5 scale. The loader applies
 * `sqrt(eta/5)*h0` and `sqrt((5-eta)/5)*h1`, which normalises the pair to
 * `||h0||^2 + h1^2 = 1`. Swapping the two was a real bug in the Python once;
 * the postcondition below is what catches it.
 *
 * \param dir    directory holding unbalanced.json / balanced.json
 * \param L0     filter length, the key in the JSON object
 * \param family "unbalanced" or "balanced"
 */
inline Taps load_taps(const std::string& dir, const int L0, const std::string& family)
{
    const std::string path = dir + "/" + family + ".json";
    std::ifstream     ifs(path);
    if (!ifs.good()) throw std::runtime_error("MS-PRS: cannot open tap file '" + path + "'");

    nlohmann::json j;
    ifs >> j;

    const std::string key = std::to_string(L0);
    if (!j.contains(key))
        throw std::runtime_error("MS-PRS: no L0=" + key + " in '" + path +
                                 "'. Higher unbalanced keys are intentionally absent; "
                                 "recompute from the companion preprint App. E before adding one.");

    Taps t;
    t.L0                              = L0;
    const std::vector<double> h0_unit = j[key]["h0"].get<std::vector<double>>();
    const std::vector<double> h1_unit = j[key]["h1"].get<std::vector<double>>();
    const double              eta     = j[key]["eta"].get<double>();

    if ((int)h0_unit.size() != L0)
        throw std::runtime_error("MS-PRS: h0 length " + std::to_string(h0_unit.size()) +
                                 " != L0 " + std::to_string(L0));

    double n0 = 0.0;
    for (auto v : h0_unit) n0 += v * v;
    double n1 = 0.0;
    for (auto v : h1_unit) n1 += v * v;
    if (std::abs(n0 - 1.0) > 1e-6 || std::abs(n1 - 1.0) > 1e-6)
        throw std::runtime_error("MS-PRS: taps for L0=" + key + " in '" + path + "' are not unit-norm");

    const double s0 = std::sqrt(eta / 5.0);
    const double s1 = std::sqrt((5.0 - eta) / 5.0);

    t.h0.resize(L0);
    for (int i = 0; i < L0; i++) t.h0[i] = s0 * h0_unit[i];
    t.h1 = s1 * h1_unit[0];

    double e = t.h1 * t.h1;
    for (auto v : t.h0) e += v * v;
    if (std::abs(e - 1.0) > 1e-9)
        throw std::runtime_error("MS-PRS: energy normalisation failed for L0=" + key +
                                 ", ||h0||^2 + h1^2 = " + std::to_string(e));

    return t;
}

} // namespace msprs

#endif
