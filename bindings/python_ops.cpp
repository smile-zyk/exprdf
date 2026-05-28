#include "python_ops.hpp"

#include <exprdf/exprdf.hpp>
#include <pybind11/numpy.h>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <complex>
#include <stdexcept>

namespace exprdf {
namespace pyops {
namespace {

using DComplex = std::complex<double>;

struct ArithAdd  { template<typename T> T operator()(T a, T b) const { return a + b; } };
struct ArithSub  { template<typename T> T operator()(T a, T b) const { return a - b; } };
struct ArithMul  { template<typename T> T operator()(T a, T b) const { return a * b; } };
struct ArithDiv  { template<typename T> T operator()(T a, T b) const { return a / b; } };

OpSpec make_op(
    const char* name,
    const std::vector<ArgSpec>& args,
    std::size_t min_args,
    std::size_t max_args,
    py::object (*impl)(const py::args&),
    const char* doc,
    bool export_to_module,
    ModuleBindMode bind_mode)
{
    OpSpec spec = {
        name,
        args,
        min_args,
        max_args,
        impl,
        doc,
        export_to_module,
        bind_mode
    };
    return spec;
}

std::vector<ArgSpec> prepend_df_arg(std::initializer_list<ArgSpec> tail_args) {
    std::vector<ArgSpec> args;
    args.reserve(1 + tail_args.size());
    args.push_back(ArgSpec{ArgType::DataFrame, "df"});
    args.insert(args.end(), tail_args.begin(), tail_args.end());
    return args;
}

OpSpec make_internal_op(
    const char* name,
    std::initializer_list<ArgSpec> args,
    std::size_t min_args,
    std::size_t max_args,
    py::object (*impl)(const py::args&),
    const char* doc)
{
    return make_op(name, std::vector<ArgSpec>(args), min_args, max_args, impl, doc, false, ModuleBindMode::None);
}

OpSpec make_export_op_legacy(
    const char* name,
    std::initializer_list<ArgSpec> args,
    std::size_t min_args,
    std::size_t max_args,
    py::object (*impl)(const py::args&),
    const char* doc)
{
    return make_op(name, std::vector<ArgSpec>(args), min_args, max_args, impl, doc, true, ModuleBindMode::LegacyArgs);
}

OpSpec make_export_op_df_first(
    const char* name,
    std::initializer_list<ArgSpec> tail_args,
    std::size_t min_args,
    std::size_t max_args,
    py::object (*impl)(const py::args&),
    const char* doc)
{
    return make_op(name, prepend_df_arg(tail_args), min_args, max_args, impl, doc, true, ModuleBindMode::DataFrameFirstArgs);
}

template <typename T>
bool is_array_with_dtype(py::handle value) {
    if (!py::isinstance<py::array>(value)) return false;
    py::array arr = py::reinterpret_borrow<py::array>(value);
    return arr.dtype().is(py::dtype::of<T>());
}

const char* type_name(ArgType t) {
    switch (t) {
        case ArgType::DataFrame: return "DataFrame";
        case ArgType::Int: return "int";
        case ArgType::Double: return "float";
        case ArgType::Complex: return "complex";
        case ArgType::String: return "str";
        case ArgType::Bool: return "bool";
        case ArgType::Array: return "numpy.ndarray";
        case ArgType::ArrayInt: return "numpy.ndarray[dtype=int]";
        case ArgType::ArrayDouble: return "numpy.ndarray[dtype=float64]";
        case ArgType::ArrayComplex: return "numpy.ndarray[dtype=complex128]";
        case ArgType::Any: return "any";
    }
    return "unknown";
}

bool accepts_arg(py::handle value, ArgType t) {
    switch (t) {
        case ArgType::DataFrame:
            return py::isinstance<exprdf::DataFrame>(value);
        case ArgType::Int:
            return py::isinstance<py::int_>(value) && !py::isinstance<py::bool_>(value);
        case ArgType::Double:
            return py::isinstance<py::float_>(value) ||
                   (py::isinstance<py::int_>(value) && !py::isinstance<py::bool_>(value));
        case ArgType::Complex:
            return PyComplex_Check(value.ptr()) ||
                   py::isinstance<py::float_>(value) ||
                   (py::isinstance<py::int_>(value) && !py::isinstance<py::bool_>(value));
        case ArgType::String:
            return py::isinstance<py::str>(value);
        case ArgType::Bool:
            return py::isinstance<py::bool_>(value);
        case ArgType::Array:
            return py::isinstance<py::array>(value);
        case ArgType::ArrayInt:
            return is_array_with_dtype<int>(value);
        case ArgType::ArrayDouble:
            return is_array_with_dtype<double>(value);
        case ArgType::ArrayComplex:
            return is_array_with_dtype<DComplex>(value);
        case ArgType::Any:
            return true;
    }
    return false;
}

int numeric_rank(DType t) {
    switch (t) {
        case DType::Int: return 0;
        case DType::Double: return 1;
        case DType::Complex: return 2;
        default: return -1;
    }
}

std::vector<double> to_double_vec(const Column& col) {
    if (col.tag == DType::Int) {
        const auto& src = col.as<int>();
        return std::vector<double>(src.begin(), src.end());
    }
    if (col.tag == DType::Double) return col.as<double>();
    throw std::invalid_argument(
        "Cannot widen '" + std::string(dtype_to_string(col.tag)) + "' to double");
}

std::vector<DComplex> to_complex_vec(const Column& col) {
    if (col.tag == DType::Int) {
        const auto& src = col.as<int>();
        std::vector<DComplex> v;
        v.reserve(src.size());
        for (auto x : src) v.emplace_back(static_cast<double>(x), 0.0);
        return v;
    }
    if (col.tag == DType::Double) {
        const auto& src = col.as<double>();
        std::vector<DComplex> v;
        v.reserve(src.size());
        for (auto x : src) v.emplace_back(x, 0.0);
        return v;
    }
    if (col.tag == DType::Complex) return col.as<DComplex>();
    throw std::invalid_argument(
        "Cannot widen '" + std::string(dtype_to_string(col.tag)) + "' to complex");
}

template<typename Op>
std::shared_ptr<DataFrame> apply_binary_op_last(const DataFrame& self, const DataFrame& other, Op op) {
    self.ensure_has_columns();
    other.ensure_has_columns("Other DataFrame");
    if (self.num_rows() != other.num_rows())
        throw std::invalid_argument(
            "Row count mismatch: " + std::to_string(self.num_rows()) +
            " vs " + std::to_string(other.num_rows()));

    const Column& ca = self.last_column();
    const Column& cb = other.last_column();

    int ra = numeric_rank(ca.tag);
    int rb = numeric_rank(cb.tag);
    if (ra < 0 || rb < 0)
        throw std::invalid_argument("Arithmetic on string columns is not supported");

    DType rt = (ra >= rb) ? ca.tag : cb.tag;
    auto result = self.copy();

    if (rt == DType::Int) {
        auto& va = result->last_column_as<int>();
        const auto& vb = cb.as<int>();
        for (std::size_t i = 0; i < va.size(); ++i) va[i] = op(va[i], vb[i]);
    } else if (rt == DType::Double) {
        std::vector<double> va = to_double_vec(ca);
        std::vector<double> vb = to_double_vec(cb);
        std::vector<double> vc(va.size());
        for (std::size_t i = 0; i < va.size(); ++i) vc[i] = op(va[i], vb[i]);
        Column nc = make_column<double>(vc);
        nc.quantity = ca.quantity;
        result->last_column() = std::move(nc);
    } else {
        std::vector<DComplex> va = to_complex_vec(ca);
        std::vector<DComplex> vb = to_complex_vec(cb);
        std::vector<DComplex> vc(va.size());
        for (std::size_t i = 0; i < va.size(); ++i) vc[i] = op(va[i], vb[i]);
        Column nc = make_column<DComplex>(vc);
        nc.quantity = ca.quantity;
        result->last_column() = std::move(nc);
    }
    return result;
}

template <typename T>
std::shared_ptr<DataFrame> array_to_df(const DataFrame& self, const py::array& arr, const char* op_name) {
    auto buf = arr.request();
    if (buf.ndim != 1) {
        throw py::type_error(std::string(op_name) + " expects 1-D numpy array");
    }
    if (buf.size == 0) {
        throw std::invalid_argument("array is empty");
    }

    const T* ptr = static_cast<const T*>(buf.ptr);
    std::size_t n = static_cast<std::size_t>(buf.size);
    std::size_t nrows = self.num_rows();
    if (n != 1 && n != nrows) {
        throw std::invalid_argument(
            "array length (" + std::to_string(n) +
            ") must be 1 (scalar) or num_rows (" + std::to_string(nrows) + ")");
    }

    self.ensure_has_columns();

    const std::string& cname = self.last_column_name();
    auto tmp = std::make_shared<DataFrame>();
    tmp->add_column<T>(
        cname,
        n == 1 ? std::vector<T>(nrows, ptr[0]) : std::vector<T>(ptr, ptr + n));
    return tmp;
}

std::shared_ptr<DataFrame> scalar_to_df(const DataFrame& self, py::handle value, const char* op_name) {
    self.ensure_has_columns();
    const std::string& cname = self.last_column_name();
    const std::size_t nrows = self.num_rows();

    auto tmp = std::make_shared<DataFrame>();
    if (PyComplex_Check(value.ptr())) {
        auto c = py::cast<DComplex>(value);
        tmp->add_column<DComplex>(cname, std::vector<DComplex>(nrows, c));
        return tmp;
    }
    if (py::isinstance<py::float_>(value)) {
        auto d = py::cast<double>(value);
        tmp->add_column<double>(cname, std::vector<double>(nrows, d));
        return tmp;
    }
    if (py::isinstance<py::int_>(value) && !py::isinstance<py::bool_>(value)) {
        auto i = py::cast<int>(value);
        tmp->add_column<int>(cname, std::vector<int>(nrows, i));
        return tmp;
    }
    throw py::type_error(
        std::string(op_name) +
        " expects DataFrame, scalar number, or 1-D numpy array");
}

std::shared_ptr<DataFrame> coerce_to_df(const DataFrame& self, py::handle value, const char* op_name) {
    if (py::isinstance<py::array>(value)) {
        py::array arr = py::reinterpret_borrow<py::array>(value);

        if (auto a = py::array_t<DComplex>::ensure(arr)) {
            return array_to_df<DComplex>(self, a, op_name);
        }
        if (auto a = py::array_t<double>::ensure(arr)) {
            return array_to_df<double>(self, a, op_name);
        }
        if (auto a = py::array_t<int>::ensure(arr)) {
            return array_to_df<int>(self, a, op_name);
        }
        throw py::type_error(
            std::string(op_name) +
            " only supports numeric numpy arrays with dtype int/float64/complex128, got dtype='" +
            py::str(arr.dtype()).cast<std::string>() + "'");
    }

    return scalar_to_df(self, value, op_name);
}

std::shared_ptr<DataFrame> unary_to_double(
    const DataFrame& self,
    const std::function<double(double)>& fn_d,
    const std::function<double(const DComplex&)>& fn_c)
{
    self.ensure_has_columns();
    const Column& cc = self.last_column();
    std::vector<double> out;
    out.reserve(cc.size());
    switch (cc.tag) {
        case DType::Int:
            for (auto x : cc.as<int>()) out.push_back(fn_d(x));
            break;
        case DType::Double:
            for (auto x : cc.as<double>()) out.push_back(fn_d(x));
            break;
        case DType::Complex:
            for (const auto& z : cc.as<DComplex>()) out.push_back(fn_c(z));
            break;
        case DType::String:
            throw std::invalid_argument("unary op: string columns not supported");
    }
    Column nc = make_column<double>(out);
    nc.quantity = cc.quantity;
    return self.copy_with_last_column(nc);
}

std::shared_ptr<DataFrame> unary_promote(
    const DataFrame& self,
    const std::function<double(double)>& fn_d,
    const std::function<DComplex(const DComplex&)>& fn_c)
{
    self.ensure_has_columns();
    const Column& cc = self.last_column();
    auto r = self.copy();
    switch (cc.tag) {
        case DType::Int: {
            std::vector<double> out;
            out.reserve(cc.size());
            for (auto x : cc.as<int>()) out.push_back(fn_d(x));
            Column nc = make_column<double>(out);
            nc.quantity = cc.quantity;
            r->last_column() = std::move(nc);
            break;
        }
        case DType::Double:
            for (auto& x : r->last_column_as<double>()) x = fn_d(x);
            break;
        case DType::Complex:
            for (auto& z : r->last_column_as<DComplex>()) z = fn_c(z);
            break;
        case DType::String:
            throw std::invalid_argument("unary op: string columns not supported");
    }
    return r;
}

std::shared_ptr<DataFrame> negate_last(const DataFrame& df) {
    df.ensure_has_columns();
    auto r = df.copy();
    switch (df.last_column().tag) {
        case DType::Int:
            for (auto& x : r->last_column_as<int>()) x = -x;
            break;
        case DType::Double:
            for (auto& x : r->last_column_as<double>()) x = -x;
            break;
        case DType::Complex:
            for (auto& z : r->last_column_as<DComplex>()) z = -z;
            break;
        case DType::String:
            throw std::invalid_argument("unary -: string columns not supported");
    }
    return r;
}

const DataFrame& require_df(const py::args& args) {
    return args[0].cast<const DataFrame&>();
}

const DataFrame& require_rhs_df(const DataFrame& self, py::handle rhs, const char* op_name, std::shared_ptr<DataFrame>& owned) {
    if (py::isinstance<exprdf::DataFrame>(rhs)) {
        return rhs.cast<const DataFrame&>();
    }
    owned = coerce_to_df(self, rhs, op_name);
    return *owned;
}

py::object op_add(const py::args& args) {
    const DataFrame& df = require_df(args);
    std::shared_ptr<DataFrame> rhs_owned;
    const DataFrame& rhs = require_rhs_df(df, args[1], "add", rhs_owned);
    return py::cast(apply_binary_op_last(df, rhs, ArithAdd()));
}

py::object op_sub(const py::args& args) {
    const DataFrame& df = require_df(args);
    std::shared_ptr<DataFrame> rhs_owned;
    const DataFrame& rhs = require_rhs_df(df, args[1], "sub", rhs_owned);
    return py::cast(apply_binary_op_last(df, rhs, ArithSub()));
}

py::object op_mul(const py::args& args) {
    const DataFrame& df = require_df(args);
    std::shared_ptr<DataFrame> rhs_owned;
    const DataFrame& rhs = require_rhs_df(df, args[1], "mul", rhs_owned);
    return py::cast(apply_binary_op_last(df, rhs, ArithMul()));
}

py::object op_truediv(const py::args& args) {
    const DataFrame& df = require_df(args);
    std::shared_ptr<DataFrame> rhs_owned;
    const DataFrame& rhs = require_rhs_df(df, args[1], "truediv", rhs_owned);
    return py::cast(apply_binary_op_last(df, rhs, ArithDiv()));
}

py::object op_radd(const py::args& args) {
    return op_add(args);
}

py::object op_rmul(const py::args& args) {
    return op_mul(args);
}

py::object op_rsub(const py::args& args) {
    const DataFrame& df = require_df(args);
    std::shared_ptr<DataFrame> lhs_owned;
    const DataFrame& lhs = require_rhs_df(df, args[1], "rsub", lhs_owned);
    return py::cast(apply_binary_op_last(lhs, df, ArithSub()));
}

py::object op_rtruediv(const py::args& args) {
    const DataFrame& df = require_df(args);
    std::shared_ptr<DataFrame> lhs_owned;
    const DataFrame& lhs = require_rhs_df(df, args[1], "rtruediv", lhs_owned);
    return py::cast(apply_binary_op_last(lhs, df, ArithDiv()));
}

py::object op_neg(const py::args& args) {
    return py::cast(negate_last(require_df(args)));
}

py::object op_abs(const py::args& args) {
    const DataFrame& df = require_df(args);
    df.ensure_has_columns();
    const Column& cc = df.last_column();
    auto r = df.copy();
    switch (cc.tag) {
        case DType::Int:
            for (auto& x : r->last_column_as<int>()) x = std::abs(x);
            break;
        case DType::Double:
            for (auto& x : r->last_column_as<double>()) x = std::abs(x);
            break;
        case DType::Complex: {
            const auto& src = cc.as<DComplex>();
            std::vector<double> out(src.size());
            for (std::size_t i = 0; i < src.size(); ++i) out[i] = std::abs(src[i]);
            Column nc = make_column<double>(out);
            nc.quantity = cc.quantity;
            r->last_column() = std::move(nc);
            break;
        }
        case DType::String:
            throw std::invalid_argument("abs: string columns not supported");
    }
    return py::cast(r);
}

py::object op_mag(const py::args& args) { return op_abs(args); }

py::object op_real(const py::args& args) {
    const DataFrame& df = require_df(args);
    return py::cast(unary_to_double(
        df,
        [](double x) { return x; },
        [](const DComplex& z) { return z.real(); }));
}

py::object op_imag(const py::args& args) {
    const DataFrame& df = require_df(args);
    return py::cast(unary_to_double(
        df,
        [](double) { return 0.0; },
        [](const DComplex& z) { return z.imag(); }));
}

py::object op_phase(const py::args& args) {
    const DataFrame& df = require_df(args);
    return py::cast(unary_to_double(
        df,
        [](double x) { return std::atan2(0.0, x); },
        [](const DComplex& z) { return std::arg(z); }));
}

py::object op_dB(const py::args& args) {
    const DataFrame& df = require_df(args);
    return py::cast(unary_to_double(
        df,
        [](double x) { return 20.0 * std::log10(std::abs(x)); },
        [](const DComplex& z) { return 20.0 * std::log10(std::abs(z)); }));
}

py::object op_dBm(const py::args& args) {
    const DataFrame& df = require_df(args);
    return py::cast(unary_to_double(
        df,
        [](double x) { return 20.0 * std::log10(std::abs(x)) + 10.0; },
        [](const DComplex& z) { return 20.0 * std::log10(std::abs(z)) + 10.0; }));
}

py::object op_wtodBm(const py::args& args) {
    const DataFrame& df = require_df(args);
    df.ensure_has_columns();
    const Column& cc = df.last_column();
    std::vector<double> out;
    out.reserve(cc.size());
    switch (cc.tag) {
        case DType::Int:
            for (auto x : cc.as<int>()) out.push_back(10.0 * std::log10(double(x) * 1000.0));
            break;
        case DType::Double:
            for (auto x : cc.as<double>()) out.push_back(10.0 * std::log10(x * 1000.0));
            break;
        case DType::Complex:
            throw std::invalid_argument(
                "wtodBm: complex input not supported (use dBm for complex magnitude)");
        case DType::String:
            throw std::invalid_argument("wtodBm: string columns not supported");
    }
    Column nc = make_column<double>(out);
    nc.quantity = cc.quantity;
    return py::cast(df.copy_with_last_column(nc));
}

py::object op_sqr(const py::args& args) {
    const DataFrame& df = require_df(args);
    df.ensure_has_columns();
    auto r = df.copy();
    switch (df.last_column().tag) {
        case DType::Int:
            for (auto& x : r->last_column_as<int>()) x = x * x;
            break;
        case DType::Double:
            for (auto& x : r->last_column_as<double>()) x = x * x;
            break;
        case DType::Complex:
            for (auto& z : r->last_column_as<DComplex>()) z = z * z;
            break;
        case DType::String:
            throw std::invalid_argument("sqr: string columns not supported");
    }
    return py::cast(r);
}

py::object op_sqrt(const py::args& args) {
    const DataFrame& df = require_df(args);
    return py::cast(unary_promote(
        df,
        [](double x) { return std::sqrt(x); },
        [](const DComplex& z) { return std::sqrt(z); }));
}

py::object op_exp(const py::args& args) {
    const DataFrame& df = require_df(args);
    return py::cast(unary_promote(
        df,
        [](double x) { return std::exp(x); },
        [](const DComplex& z) { return std::exp(z); }));
}

py::object op_ln(const py::args& args) {
    const DataFrame& df = require_df(args);
    return py::cast(unary_promote(
        df,
        [](double x) { return std::log(x); },
        [](const DComplex& z) { return std::log(z); }));
}

py::object op_log10(const py::args& args) {
    const DataFrame& df = require_df(args);
    return py::cast(unary_promote(
        df,
        [](double x) { return std::log10(x); },
        [](const DComplex& z) { return std::log10(z); }));
}

py::object op_conj(const py::args& args) {
    const DataFrame& df = require_df(args);
    df.ensure_has_columns();
    const Column& cc = df.last_column();
    auto r = df.copy();
    switch (cc.tag) {
        case DType::Int:
        case DType::Double:
            break;
        case DType::Complex:
            for (auto& z : r->last_column_as<DComplex>()) z = std::conj(z);
            break;
        case DType::String:
            throw std::invalid_argument("conj: string columns not supported");
    }
    return py::cast(r);
}

py::object op_zin(const py::args& args) {
    const DataFrame& df = require_df(args);
    const DComplex z0 = (args.size() >= 2) ? args[1].cast<DComplex>() : DComplex(50.0, 0.0);
    df.ensure_has_columns("zin: DataFrame");

    const Column& cc = df.last_column();
    if (cc.tag == DType::String) {
        throw std::invalid_argument("zin: string columns not supported");
    }

    std::vector<DComplex> src = to_complex_vec(cc);
    std::vector<DComplex> out;
    out.reserve(src.size());
    for (std::size_t i = 0; i < src.size(); ++i) {
        DComplex denom = DComplex(1.0, 0.0) - src[i];
        if (std::abs(denom) == 0.0) {
            throw std::invalid_argument("zin: S11 = 1 leads to division by zero");
        }
        out.push_back(z0 * (DComplex(1.0, 0.0) + src[i]) / denom);
    }

    Column nc = make_column<DComplex>(out);
    nc.quantity = cc.quantity;
    return py::cast(df.copy_with_last_column(nc));
}

const std::vector<OpSpec>& all_ops() {
    static const std::vector<OpSpec> specs = {
        make_internal_op("add", {{ArgType::DataFrame, "df"}, {ArgType::Any, "rhs"}}, 2, 2, &op_add, "internal add"),
        make_internal_op("sub", {{ArgType::DataFrame, "df"}, {ArgType::Any, "rhs"}}, 2, 2, &op_sub, "internal sub"),
        make_internal_op("mul", {{ArgType::DataFrame, "df"}, {ArgType::Any, "rhs"}}, 2, 2, &op_mul, "internal mul"),
        make_internal_op("truediv", {{ArgType::DataFrame, "df"}, {ArgType::Any, "rhs"}}, 2, 2, &op_truediv, "internal truediv"),
        make_internal_op("radd", {{ArgType::DataFrame, "df"}, {ArgType::Any, "lhs"}}, 2, 2, &op_radd, "internal radd"),
        make_internal_op("rsub", {{ArgType::DataFrame, "df"}, {ArgType::Any, "lhs"}}, 2, 2, &op_rsub, "internal rsub"),
        make_internal_op("rmul", {{ArgType::DataFrame, "df"}, {ArgType::Any, "lhs"}}, 2, 2, &op_rmul, "internal rmul"),
        make_internal_op("rtruediv", {{ArgType::DataFrame, "df"}, {ArgType::Any, "lhs"}}, 2, 2, &op_rtruediv, "internal rtruediv"),
        make_internal_op("neg", {{ArgType::DataFrame, "df"}}, 1, 1, &op_neg, "internal neg"),
        make_export_op_df_first("abs", {}, 1, 1, &op_abs, "abs(df): magnitude/absolute value on last column"),
        make_export_op_df_first("mag", {}, 1, 1, &op_mag, "mag(df): alias of abs(df)"),
        make_export_op_df_first("real", {}, 1, 1, &op_real, "real(df): real part on last column"),
        make_export_op_df_first("imag", {}, 1, 1, &op_imag, "imag(df): imag part on last column"),
        make_export_op_df_first("phase", {}, 1, 1, &op_phase, "phase(df): phase in radians on last column"),
        make_export_op_df_first("dB", {}, 1, 1, &op_dB, "dB(df): 20*log10(|x|) on last column"),
        make_export_op_df_first("dBm", {}, 1, 1, &op_dBm, "dBm(df): 20*log10(|x|)+10 on last column"),
        make_export_op_df_first("wtodBm", {}, 1, 1, &op_wtodBm, "wtodBm(df): convert watt to dBm on last column"),
        make_export_op_df_first("sqr", {}, 1, 1, &op_sqr, "sqr(df): x^2 on last column"),
        make_export_op_df_first("sqrt", {}, 1, 1, &op_sqrt, "sqrt(df): square root on last column"),
        make_export_op_df_first("exp", {}, 1, 1, &op_exp, "exp(df): exponential on last column"),
        make_export_op_df_first("ln", {}, 1, 1, &op_ln, "ln(df): natural log on last column"),
        make_export_op_df_first("log10", {}, 1, 1, &op_log10, "log10(df): base-10 log on last column"),
        make_export_op_df_first("conj", {}, 1, 1, &op_conj, "conj(df): conjugate on last column"),
        make_export_op_df_first("zin", {{ArgType::Complex, "z0"}}, 1, 2, &op_zin, "zin(df, z0=50): input impedance Zin = Z0*(1+S11)/(1-S11) on last column")
    };
    return specs;
}

py::object invoke_df_first(const std::string& name, const DataFrame& df, const py::args& rest) {
    py::tuple full_args(1 + rest.size());
    full_args[0] = py::cast(df);
    for (std::size_t i = 0; i < rest.size(); ++i) {
        full_args[i + 1] = py::reinterpret_borrow<py::object>(rest[i]);
    }
    return invoke(name, full_args);
}

std::string to_lower_ascii(std::string s) {
    for (std::size_t i = 0; i < s.size(); ++i) {
        const unsigned char ch = static_cast<unsigned char>(s[i]);
        s[i] = static_cast<char>(std::tolower(ch));
    }
    return s;
}

void register_module_op(py::module_& m, const OpSpec& spec) {
    if (!spec.export_to_module || spec.bind_mode == ModuleBindMode::None) return;

    const std::string op_name = spec.name;
    const std::string lower_name = to_lower_ascii(op_name);
    switch (spec.bind_mode) {
        case ModuleBindMode::LegacyArgs:
            m.def(
                spec.name.c_str(),
                [op_name](py::args args) {
                    return invoke(op_name, py::tuple(args));
                },
                spec.doc);
            if (lower_name != op_name) {
                m.def(
                    lower_name.c_str(),
                    [op_name](py::args args) {
                        return invoke(op_name, py::tuple(args));
                    },
                    spec.doc);
            }
            break;
        case ModuleBindMode::DataFrameFirstArgs:
            m.def(
                spec.name.c_str(),
                [op_name](const exprdf::DataFrame& df, py::args args) {
                    return invoke_df_first(op_name, df, args);
                },
                py::arg("df"),
                spec.doc);
            if (lower_name != op_name) {
                m.def(
                    lower_name.c_str(),
                    [op_name](const exprdf::DataFrame& df, py::args args) {
                        return invoke_df_first(op_name, df, args);
                    },
                    py::arg("df"),
                    spec.doc);
            }
            break;
        case ModuleBindMode::None:
            break;
    }
}

const OpSpec& find_spec(const std::string& name) {
    const auto& specs = all_ops();
    for (std::size_t i = 0; i < specs.size(); ++i) {
        if (specs[i].name == name) return specs[i];
    }
    throw py::key_error("Unknown python operation: '" + name + "'");
}

void validate_args(const OpSpec& spec, const py::tuple& args) {
    if (args.size() < spec.min_args || args.size() > spec.max_args) {
        throw py::type_error(
            spec.name + " expects " + std::to_string(spec.min_args) +
            (spec.min_args == spec.max_args
                ? " arguments"
                : (".." + std::to_string(spec.max_args) + " arguments")) +
            ", got " + std::to_string(args.size()));
    }

    const std::size_t check_n = std::min<std::size_t>(args.size(), spec.args.size());
    for (std::size_t i = 0; i < check_n; ++i) {
        if (!accepts_arg(args[i], spec.args[i].type)) {
            throw py::type_error(
                spec.name + " arg " + std::to_string(i + 1) +
                " ('" + std::string(spec.args[i].name) + "') expects " +
                type_name(spec.args[i].type));
        }
    }
}

} // namespace

py::object invoke(const std::string& name, const py::tuple& args) {
    const OpSpec& spec = find_spec(name);
    validate_args(spec, args);
    py::args vargs = py::reinterpret_borrow<py::args>(args);
    return spec.impl(vargs);
}

void bind_module_functions(py::module_& m) {
    const auto& specs = all_ops();
    for (std::size_t i = 0; i < specs.size(); ++i) {
        register_module_op(m, specs[i]);
    }
}

} // namespace pyops
} // namespace exprdf
