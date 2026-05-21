// df_ops.cpp -- built-in DataFrame operation registrations for exprdf.
//
// ============================================================
// HOW TO ADD A NEW BUILT-IN OPERATION (no header needs to change):
//
//   1. Open this file (src/df_ops.cpp).
//   2. Add a register_fn("name", ...) call inside DFOpsRegistrar_() below.
//      Use detail::unary_to_double / detail::unary_promote as helpers.
//   3. Done.
//      - C++  : exprdf::apply_fn("name", df)
//      - Python: exprdf.name(df)   (auto-discovered via registry loop in bindings)
// ============================================================

#include <exprdf/df_ops.hpp>
#include <cmath>
#include <complex>
#include <stdexcept>

namespace exprdf {

namespace {

struct DFOpsRegistrar_ {
    DFOpsRegistrar_() {
        using C = std::complex<double>;

        register_fn("abs", [](const DataFrame& df) {
            if (df.num_columns() == 0) throw std::invalid_argument("DataFrame has no columns");
            const std::string& ln = df.column_name(df.num_columns() - 1);
            const Column& cc = df.get_column(ln);
            auto r = df.copy();
            switch (cc.tag) {
                case DType::Int:    for (auto& x : r->get_column(ln).as<int>())    x = std::abs(x); break;
                case DType::Double: for (auto& x : r->get_column(ln).as<double>()) x = std::abs(x); break;
                case DType::Complex: {
                    const auto& src = cc.as<C>();
                    std::vector<double> out(src.size());
                    for (std::size_t i = 0; i < src.size(); ++i) out[i] = std::abs(src[i]);
                    Column nc = make_column<double>(out); nc.quantity = cc.quantity;
                    r->get_column(ln) = std::move(nc); break;
                }
                case DType::String: throw std::invalid_argument("abs: string columns not supported");
            }
            return r;
        });

        register_fn("mag", [](const DataFrame& df) { return apply_fn("abs", df); });

        register_fn("real", [](const DataFrame& df) {
            return detail::unary_to_double(df,
                [](double x)   { return x; },
                [](const C& z) { return z.real(); });
        });

        register_fn("imag", [](const DataFrame& df) {
            return detail::unary_to_double(df,
                [](double)     { return 0.0; },
                [](const C& z) { return z.imag(); });
        });

        register_fn("phase", [](const DataFrame& df) {
            return detail::unary_to_double(df,
                [](double x)   { return std::atan2(0.0, x); },
                [](const C& z) { return std::arg(z); });
        });

        register_fn("dB", [](const DataFrame& df) {
            return detail::unary_to_double(df,
                [](double x)   { return 20.0 * std::log10(std::abs(x)); },
                [](const C& z) { return 20.0 * std::log10(std::abs(z)); });
        });

        register_fn("dBm", [](const DataFrame& df) {
            return detail::unary_to_double(df,
                [](double x)   { return 20.0 * std::log10(std::abs(x)) + 10.0; },
                [](const C& z) { return 20.0 * std::log10(std::abs(z)) + 10.0; });
        });

        register_fn("wtodBm", [](const DataFrame& df) {
            if (df.num_columns() == 0) throw std::invalid_argument("DataFrame has no columns");
            const std::string& ln = df.column_name(df.num_columns() - 1);
            const Column& cc = df.get_column(ln);
            std::vector<double> out; out.reserve(cc.size());
            switch (cc.tag) {
                case DType::Int:
                    for (auto x : cc.as<int>()) {
                        out.push_back(10.0 * std::log10(double(x) * 1000.0));
                    }
                    break;
                case DType::Double:
                    for (auto x : cc.as<double>()) {
                        out.push_back(10.0 * std::log10(x * 1000.0));
                    }
                    break;
                case DType::Complex:
                    throw std::invalid_argument(
                        "wtodBm: complex input not supported (use dBm for complex magnitude)");
                case DType::String:
                    throw std::invalid_argument("wtodBm: string columns not supported");
            }
            auto r = df.copy();
            Column nc = make_column<double>(out); nc.quantity = cc.quantity;
            r->get_column(ln) = std::move(nc);
            return r;
        });

        register_fn("sqr", [](const DataFrame& df) {
            if (df.num_columns() == 0) throw std::invalid_argument("DataFrame has no columns");
            const std::string& ln = df.column_name(df.num_columns() - 1);
            auto r = df.copy();
            switch (df.get_column(ln).tag) {
                case DType::Int:     for (auto& x : r->get_column(ln).as<int>())    x = x * x; break;
                case DType::Double:  for (auto& x : r->get_column(ln).as<double>()) x = x * x; break;
                case DType::Complex: for (auto& z : r->get_column(ln).as<C>())      z = z * z; break;
                case DType::String:  throw std::invalid_argument("sqr: string columns not supported");
            }
            return r;
        });

        register_fn("sqrt", [](const DataFrame& df) {
            return detail::unary_promote(df,
                [](double x)   { return std::sqrt(x); },
                [](const C& z) { return std::sqrt(z); });
        });

        register_fn("exp", [](const DataFrame& df) {
            return detail::unary_promote(df,
                [](double x)   { return std::exp(x); },
                [](const C& z) { return std::exp(z); });
        });

        register_fn("ln", [](const DataFrame& df) {
            return detail::unary_promote(df,
                [](double x)   { return std::log(x); },
                [](const C& z) { return std::log(z); });
        });

        register_fn("log10", [](const DataFrame& df) {
            return detail::unary_promote(df,
                [](double x)   { return std::log10(x); },
                [](const C& z) { return std::log10(z); });
        });

        register_fn("conj", [](const DataFrame& df) {
            if (df.num_columns() == 0) throw std::invalid_argument("DataFrame has no columns");
            const std::string& ln = df.column_name(df.num_columns() - 1);
            const Column& cc = df.get_column(ln);
            auto r = df.copy();
            switch (cc.tag) {
                case DType::Int:    break; // real: conj is identity
                case DType::Double: break; // real: conj is identity
                case DType::Complex:
                    for (auto& z : r->get_column(ln).as<C>()) z = std::conj(z);
                    break;
                case DType::String:
                    throw std::invalid_argument("conj: string columns not supported");
            }
            return r;
        });

        // ---- Add new operations below this line ----
        // Example:
        //   register_fn("sinc", [](const DataFrame& df) {
        //       return detail::unary_to_double(df,
        //           [](double x)   { return x == 0.0 ? 1.0 : std::sin(x) / x; },
        //           [](const C& z) { return std::abs(z) == 0.0 ? 1.0 : std::abs(std::sin(z)/z); });
        //   });
    }
};

static const DFOpsRegistrar_ _df_ops_registrar_;

} // anonymous namespace

} // namespace exprdf
