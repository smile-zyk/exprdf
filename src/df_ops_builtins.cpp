// df_ops_builtins.cpp -- non-registry DataFrame operations.
//
// Functions with extra parameters (beyond const DataFrame&) that cannot be
// stored in the UnaryFn registry. Declared in include/exprdf/df_ops_builtins.hpp.
//
// HOW TO ADD A NEW NON-REGISTRY FUNCTION:
//   1. Declare it in include/exprdf/df_ops_builtins.hpp.
//   2. Implement it here.

#include <exprdf/df_ops_builtins.hpp>
#include <complex>
#include <stdexcept>

namespace exprdf {

// ============================================================
// zin: input impedance from reflection coefficient S11.
// Zin = Z0 * (1 + S11) / (1 - S11) element-wise on the last column.
// ============================================================
std::shared_ptr<DataFrame> zin(const DataFrame& df, std::complex<double> z0)
{
    using C = std::complex<double>;
    if (df.num_columns() == 0)
        throw std::invalid_argument("zin: DataFrame has no columns");
    const std::string& ln = df.column_name(df.num_columns() - 1);
    const Column& cc = df.get_column(ln);
    if (cc.tag == DType::String)
        throw std::invalid_argument("zin: string columns not supported");
    std::vector<C> src = detail::to_complex_vec(cc);
    std::vector<C> out; out.reserve(src.size());
    for (std::size_t i = 0; i < src.size(); ++i) {
        C denom = C(1.0, 0.0) - src[i];
        if (std::abs(denom) == 0.0)
            throw std::invalid_argument("zin: S11 = 1 leads to division by zero");
        out.push_back(z0 * (C(1.0, 0.0) + src[i]) / denom);
    }
    auto r = df.copy();
    Column nc = make_column<C>(out); nc.quantity = cc.quantity;
    r->get_column(ln) = std::move(nc);
    return r;
}

std::shared_ptr<DataFrame> zin(std::shared_ptr<const DataFrame> df, std::complex<double> z0) {
    return zin(*df, z0);
}

} // namespace exprdf
