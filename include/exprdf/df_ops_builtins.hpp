#ifndef EXPRDF_DF_OPS_BUILTINS_HPP
#define EXPRDF_DF_OPS_BUILTINS_HPP

// df_ops_builtins.hpp -- non-registry DataFrame operations.
//
// These functions have extra parameters beyond (const DataFrame&) and therefore
// cannot be stored in the UnaryFn registry. Each is declared here and defined
// in src/df_ops_builtins.cpp.
//
// HOW TO ADD A NEW NON-REGISTRY FUNCTION:
//   1. Declare it here.
//   2. Implement it in src/df_ops_builtins.cpp.

#include <exprdf/df_ops.hpp>
#include <complex>
#include <memory>

namespace exprdf {

// ============================================================
// zin: input impedance from reflection coefficient S11.
// Zin = Z0 * (1 + S11) / (1 - S11) element-wise on the last column.
// Default Z0 = 50 Ohm.
// ============================================================
std::shared_ptr<DataFrame> zin(
    const DataFrame& df,
    std::complex<double> z0 = std::complex<double>(50.0, 0.0));

std::shared_ptr<DataFrame> zin(
    std::shared_ptr<const DataFrame> df,
    std::complex<double> z0 = std::complex<double>(50.0, 0.0));

} // namespace exprdf

#endif // EXPRDF_DF_OPS_BUILTINS_HPP
