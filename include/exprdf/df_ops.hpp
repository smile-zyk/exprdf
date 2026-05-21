#ifndef EXPRDF_DF_OPS_HPP
#define EXPRDF_DF_OPS_HPP

// df_ops.hpp -- registry, operators, and helpers for exprdf::DataFrame operations.
//
// Provides:
//   UnaryFn              -- std::function<shared_ptr<DataFrame>(const DataFrame&)>
//   unary_registry()     -- singleton map: name -> UnaryFn
//   register_fn()        -- insert / overwrite one entry
//   apply_fn()           -- dispatch by name
//   operator+/−/*/÷      -- element-wise arithmetic on last column (free functions)
//   zin()                -- input impedance from S11 (takes extra z0 param)
//   abs(), dB(), conj()  -- named shortcuts delegating to the registry
//
// Built-in registrations live in src/df_ops_builtins.cpp.
//
// HOW TO ADD A NEW OPERATION (no header needs to change):
//   1. Open src/df_ops_builtins.cpp.
//   2. Add a register_fn("my_func", ...) call inside DFOpsRegistrar_().
//      Use detail::unary_to_double / detail::unary_promote as helpers.
//   3. Done.
//      - C++  : exprdf::apply_fn("my_func", df)
//      - Python: exprdf.my_func(df)   (auto-discovered via registry loop in bindings)

#include <exprdf/exprdf.hpp>  // circular -- resolved by include guard
#include <cmath>
#include <complex>
#include <functional>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace exprdf {

// ============================================================
// Operation registry
// ============================================================

using UnaryFn = std::function<std::shared_ptr<DataFrame>(const DataFrame&)>;

// Singleton registry map (Meyers singleton -- thread-safe in C++11+).
inline std::map<std::string, UnaryFn>& unary_registry() {
    static std::map<std::string, UnaryFn> reg;
    return reg;
}

// Register a named unary function. Overwrites any existing entry.
inline void register_fn(const std::string& name, UnaryFn fn) {
    unary_registry()[name] = std::move(fn);
}

// Dispatch a registered function by name.
// Throws std::invalid_argument if the name is not registered.
inline std::shared_ptr<DataFrame> apply_fn(const std::string& name, const DataFrame& df) {
    auto& reg = unary_registry();
    auto it = reg.find(name);
    if (it == reg.end())
        throw std::invalid_argument("Unknown operation: '" + name + "'");
    return it->second(df);
}

// Retrieve a registered function by name.
// Throws std::invalid_argument if the name is not registered.
inline UnaryFn get_fn(const std::string& name) {
    auto& reg = unary_registry();
    auto it = reg.find(name);
    if (it == reg.end())
        throw std::invalid_argument("Unknown operation: '" + name + "'");
    return it->second;
}

// Check whether a name is registered.
inline bool has_fn(const std::string& name) {
    return unary_registry().count(name) != 0;
}

// ============================================================
// detail: helpers for building operation implementations
// ============================================================

namespace detail {

// C++11 arithmetic function objects
struct Arith_Add  { template<typename T> T operator()(T a, T b) const { return a + b; } };
struct Arith_Sub  { template<typename T> T operator()(T a, T b) const { return a - b; } };
struct Arith_Mul  { template<typename T> T operator()(T a, T b) const { return a * b; } };
struct Arith_Div  { template<typename T> T operator()(T a, T b) const { return a / b; } };
struct Arith_SubR { template<typename T> T operator()(T a, T b) const { return b - a; } }; // s - df
struct Arith_DivR { template<typename T> T operator()(T a, T b) const { return b / a; } }; // s / df

inline int numeric_rank(DType t) {
    switch (t) {
        case DType::Int:     return 0;
        case DType::Double:  return 1;
        case DType::Complex: return 2;
        default:             return -1; // String
    }
}

inline std::vector<double> to_double_vec(const Column& col) {
    if (col.tag == DType::Int) {
        const auto& src = col.as<int>();
        return std::vector<double>(src.begin(), src.end());
    }
    if (col.tag == DType::Double) return col.as<double>();
    throw std::invalid_argument(
        "Cannot widen '" + std::string(dtype_to_string(col.tag)) + "' to double");
}

inline std::vector<std::complex<double>> to_complex_vec(const Column& col) {
    using C = std::complex<double>;
    if (col.tag == DType::Int) {
        const auto& src = col.as<int>();
        std::vector<C> v; v.reserve(src.size());
        for (auto x : src) v.emplace_back(static_cast<double>(x), 0.0);
        return v;
    }
    if (col.tag == DType::Double) {
        const auto& src = col.as<double>();
        std::vector<C> v; v.reserve(src.size());
        for (auto x : src) v.emplace_back(x, 0.0);
        return v;
    }
    if (col.tag == DType::Complex) return col.as<C>();
    throw std::invalid_argument(
        "Cannot widen '" + std::string(dtype_to_string(col.tag)) + "' to complex");
}

template<typename Op>
inline std::shared_ptr<DataFrame> apply_binary_op_last(
    const DataFrame& self, const DataFrame& o, Op op)
{
    if (self.num_columns() == 0)
        throw std::invalid_argument("DataFrame has no columns");
    if (o.num_columns() == 0)
        throw std::invalid_argument("Other DataFrame has no columns");
    if (self.num_rows() != o.num_rows())
        throw std::invalid_argument(
            "Row count mismatch: " + std::to_string(self.num_rows()) +
            " vs " + std::to_string(o.num_rows()));

    const std::string& ln = self.column_name(self.num_columns() - 1);
    const Column& ca = self.get_column(ln);
    const Column& cb = o.get_column(o.column_name(o.num_columns() - 1));

    int ra = numeric_rank(ca.tag), rb = numeric_rank(cb.tag);
    if (ra < 0 || rb < 0)
        throw std::invalid_argument("Arithmetic on string columns is not supported");

    DType rt = (ra >= rb) ? ca.tag : cb.tag;
    auto result = self.copy();

    if (rt == DType::Int) {
        auto& va = result->get_column(ln).as<int>();
        const auto& vb = cb.as<int>();
        for (std::size_t i = 0; i < va.size(); ++i) va[i] = op(va[i], vb[i]);
    } else if (rt == DType::Double) {
        std::vector<double> va = to_double_vec(ca);
        std::vector<double> vb = to_double_vec(cb);
        std::vector<double> vc(va.size());
        for (std::size_t i = 0; i < va.size(); ++i) vc[i] = op(va[i], vb[i]);
        Column nc = make_column<double>(vc);
        nc.quantity = ca.quantity;
        result->get_column(ln) = std::move(nc);
    } else { // Complex
        using C = std::complex<double>;
        std::vector<C> va = to_complex_vec(ca);
        std::vector<C> vb = to_complex_vec(cb);
        std::vector<C> vc(va.size());
        for (std::size_t i = 0; i < va.size(); ++i) vc[i] = op(va[i], vb[i]);
        Column nc = make_column<C>(vc);
        nc.quantity = ca.quantity;
        result->get_column(ln) = std::move(nc);
    }
    return result;
}

template<typename Op>
inline std::shared_ptr<DataFrame> apply_scalar_op_last(
    const DataFrame& self, double s, Op op)
{
    if (self.num_columns() == 0)
        throw std::invalid_argument("DataFrame has no columns");
    const std::string& ln = self.column_name(self.num_columns() - 1);
    const Column& cc = self.get_column(ln);
    auto result = self.copy();
    switch (cc.tag) {
        case DType::Int: {
            const auto& src = cc.as<int>();
            std::vector<double> vc(src.size());
            for (std::size_t i = 0; i < src.size(); ++i)
                vc[i] = op(static_cast<double>(src[i]), s);
            Column nc = make_column<double>(vc);
            nc.quantity = cc.quantity;
            result->get_column(ln) = std::move(nc);
            break;
        }
        case DType::Double: {
            auto& ra = result->get_column(ln).as<double>();
            for (auto& v : ra) v = op(v, s);
            break;
        }
        case DType::Complex: {
            auto& ra = result->get_column(ln).as<std::complex<double>>();
            for (auto& v : ra) v = op(v, std::complex<double>(s, 0.0));
            break;
        }
        case DType::String:
            throw std::invalid_argument("Arithmetic on string columns is not supported");
    }
    return result;
}

// unary_to_double: int/double/complex -> double column.
// fn_d applied to int/double (int is widened), fn_c applied to complex.
inline std::shared_ptr<DataFrame> unary_to_double(
    const DataFrame& self,
    std::function<double(double)> fn_d,
    std::function<double(const std::complex<double>&)> fn_c)
{
    if (self.num_columns() == 0)
        throw std::invalid_argument("DataFrame has no columns");
    const std::string& ln = self.column_name(self.num_columns() - 1);
    const Column& cc = self.get_column(ln);
    std::vector<double> out; out.reserve(cc.size());
    switch (cc.tag) {
        case DType::Int:    for (auto x : cc.as<int>())    out.push_back(fn_d(x)); break;
        case DType::Double: for (auto x : cc.as<double>()) out.push_back(fn_d(x)); break;
        case DType::Complex:
            for (const auto& z : cc.as<std::complex<double>>()) out.push_back(fn_c(z)); break;
        case DType::String:
            throw std::invalid_argument("unary op: string columns not supported");
    }
    auto r = self.copy();
    Column nc = make_column<double>(out); nc.quantity = cc.quantity;
    r->get_column(ln) = std::move(nc);
    return r;
}

// unary_promote: int -> double, double -> double, complex -> complex.
// fn_d applied to int/double, fn_c applied to complex.
inline std::shared_ptr<DataFrame> unary_promote(
    const DataFrame& self,
    std::function<double(double)> fn_d,
    std::function<std::complex<double>(const std::complex<double>&)> fn_c)
{
    using C = std::complex<double>;
    if (self.num_columns() == 0)
        throw std::invalid_argument("DataFrame has no columns");
    const std::string& ln = self.column_name(self.num_columns() - 1);
    const Column& cc = self.get_column(ln);
    auto r = self.copy();
    switch (cc.tag) {
        case DType::Int: {
            std::vector<double> out; out.reserve(cc.size());
            for (auto x : cc.as<int>()) out.push_back(fn_d(x));
            Column nc = make_column<double>(out); nc.quantity = cc.quantity;
            r->get_column(ln) = std::move(nc); break;
        }
        case DType::Double:
            for (auto& x : r->get_column(ln).as<double>()) x = fn_d(x); break;
        case DType::Complex:
            for (auto& z : r->get_column(ln).as<C>()) z = fn_c(z); break;
        case DType::String:
            throw std::invalid_argument("unary op: string columns not supported");
    }
    return r;
}

} // namespace detail

// ============================================================
// Arithmetic free-function operators (element-wise on last column)
// ============================================================

inline std::shared_ptr<DataFrame> operator+(const DataFrame& a, const DataFrame& b) {
    return detail::apply_binary_op_last(a, b, detail::Arith_Add());
}
inline std::shared_ptr<DataFrame> operator-(const DataFrame& a, const DataFrame& b) {
    return detail::apply_binary_op_last(a, b, detail::Arith_Sub());
}
inline std::shared_ptr<DataFrame> operator*(const DataFrame& a, const DataFrame& b) {
    return detail::apply_binary_op_last(a, b, detail::Arith_Mul());
}
inline std::shared_ptr<DataFrame> operator/(const DataFrame& a, const DataFrame& b) {
    return detail::apply_binary_op_last(a, b, detail::Arith_Div());
}

inline std::shared_ptr<DataFrame> operator+(const DataFrame& df, double s) {
    return detail::apply_scalar_op_last(df, s, detail::Arith_Add());
}
inline std::shared_ptr<DataFrame> operator-(const DataFrame& df, double s) {
    return detail::apply_scalar_op_last(df, s, detail::Arith_Sub());
}
inline std::shared_ptr<DataFrame> operator*(const DataFrame& df, double s) {
    return detail::apply_scalar_op_last(df, s, detail::Arith_Mul());
}
inline std::shared_ptr<DataFrame> operator/(const DataFrame& df, double s) {
    return detail::apply_scalar_op_last(df, s, detail::Arith_Div());
}

inline std::shared_ptr<DataFrame> operator+(double s, const DataFrame& df) { return df + s; }
inline std::shared_ptr<DataFrame> operator*(double s, const DataFrame& df) { return df * s; }
inline std::shared_ptr<DataFrame> operator-(double s, const DataFrame& df) {
    return detail::apply_scalar_op_last(df, s, detail::Arith_SubR());
}
inline std::shared_ptr<DataFrame> operator/(double s, const DataFrame& df) {
    return detail::apply_scalar_op_last(df, s, detail::Arith_DivR());
}

inline std::shared_ptr<DataFrame> operator-(const DataFrame& df) {
    using C = std::complex<double>;
    if (df.num_columns() == 0)
        throw std::invalid_argument("unary -: DataFrame has no columns");
    const std::string& ln = df.column_name(df.num_columns() - 1);
    auto r = df.copy();
    switch (df.get_column(ln).tag) {
        case DType::Int:     for (auto& x : r->get_column(ln).as<int>())    x = -x; break;
        case DType::Double:  for (auto& x : r->get_column(ln).as<double>()) x = -x; break;
        case DType::Complex: for (auto& z : r->get_column(ln).as<C>())      z = -z; break;
        case DType::String:  throw std::invalid_argument("unary -: string columns not supported");
    }
    return r;
}

// Use apply_fn("name", df) to call any registered operation by name.
// Use get_fn("name") to obtain the function object from the registry.
// Non-registry functions (zin, etc.) are in <exprdf/df_ops_builtins.hpp>.

} // namespace exprdf

// df_ops_builtins.hpp declares non-registry ops (zin, ...) and includes this file,
// so simply including either header gives access to everything.
#include <exprdf/df_ops_builtins.hpp>

#endif // EXPRDF_DF_OPS_HPP
