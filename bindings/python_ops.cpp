#include "python_ops.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <complex>
#include <limits>
#include <stdexcept>

#include <exprdf/exprdf.hpp>
#include <pybind11/numpy.h>

namespace exprdf {
namespace pyops {
namespace {

using DComplex = std::complex<double>;

struct ArithAdd  { template<typename T> T operator()(T a, T b) const { return a + b; } };
struct ArithSub  { template<typename T> T operator()(T a, T b) const { return a - b; } };
struct ArithMul  { template<typename T> T operator()(T a, T b) const { return a * b; } };
struct ArithDiv  { template<typename T> T operator()(T a, T b) const { return a / b; } };
struct ArithPow  {
    template<typename T>
    T operator()(T a, T b) const { return static_cast<T>(std::pow(a, b)); }
};

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

OpSpec make_op_legacy(
    const char* name,
    const std::vector<ArgSpec>& args,
    std::size_t min_args,
    std::size_t max_args,
    py::object (*impl)(const py::args&),
    const char* doc,
    bool export_to_module,
    ModuleBindMode bind_mode)
{
    return make_op(name, args, min_args, max_args, impl, doc, export_to_module, bind_mode);
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

std::vector<double> to_double_vec(
    const Column& col,
    const std::function<double(const DComplex&)>& complex_to_double = nullptr) {
    if (col.tag == DType::Int) {
        const auto& src = col.as<int>();
        return std::vector<double>(src.begin(), src.end());
    }
    if (col.tag == DType::Double) return col.as<double>();
    if (col.tag == DType::Complex) {
        if (!complex_to_double) {
            throw std::invalid_argument(
                "Cannot widen '" + std::string(dtype_to_string(col.tag)) + "' to double");
        }
        std::vector<double> out;
        const auto& src = col.as<DComplex>();
        out.reserve(src.size());
        for (const auto& z : src) out.push_back(complex_to_double(z));
        return out;
    }
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

template <typename T>
std::unique_ptr<Column> make_column_like(const Column& shape_src, const std::vector<T>& values) {
    switch (shape_src.shape.size()) {
        case 0:
            return Column::from_scalar<T>(values);
        case 1:
            return Column::from_list_flat<T>(values, shape_src.shape[0]);
        case 2:
            return Column::from_matrix_flat<T>(values, shape_src.shape[0], shape_src.shape[1]);
        default:
            throw std::invalid_argument("Unsupported column shape");
    }
}

template <typename T>
std::vector<T> broadcast_rows_to_shape(const std::vector<T>& values, const Column& shape_src) {
    const std::size_t nrows = shape_src.num_rows();
    const std::size_t epr = shape_src.elem_per_row();
    if (values.size() == nrows * epr) {
        return values;
    }
    if (values.size() == nrows && epr == 1) {
        return values;
    }
    if (values.size() == nrows) {
        std::vector<T> out;
        out.reserve(nrows * epr);
        for (const auto& value : values) {
            for (std::size_t i = 0; i < epr; ++i) {
                out.push_back(value);
            }
        }
        return out;
    }
    if (values.size() == 1) {
        return std::vector<T>(nrows * epr, values[0]);
    }
    throw std::invalid_argument("Shape broadcast failed: row count mismatch");
}

template <typename T>
std::vector<T> broadcast_for_binary_op(const Column& src, const std::vector<T>& values, std::size_t target_epr) {
    const std::size_t src_epr = src.elem_per_row();
    if (src_epr == target_epr) {
        return values;
    }
    if (src_epr == 1) {
        std::vector<T> out;
        out.reserve(src.num_rows() * target_epr);
        for (const auto& value : values) {
            for (std::size_t i = 0; i < target_epr; ++i) {
                out.push_back(value);
            }
        }
        return out;
    }
    throw std::invalid_argument("Shape mismatch between list/matrix columns");
}

template <typename T>
void replace_last_with_same_shape(const std::shared_ptr<DataFrame>& df,
                                  const Column& shape_src,
                                  const std::vector<T>& values) {
    auto nc = make_column_like<T>(shape_src, values);
    nc->quantity = shape_src.quantity;
    df->replace_last_column(std::move(nc));
}

template<typename Op>
std::shared_ptr<DataFrame> apply_binary_op_last(
    const std::shared_ptr<DataFrame>& self,
    const std::shared_ptr<DataFrame>& other,
    Op op) {
    self->ensure_has_columns();
    other->ensure_has_columns("Other DataFrame");
    if (self->num_rows() != other->num_rows())
        throw std::invalid_argument(
            "Row count mismatch: " + std::to_string(self->num_rows()) +
            " vs " + std::to_string(other->num_rows()));

    const Column& ca = self->last_column();
    const Column& cb = other->last_column();

    int ra = numeric_rank(ca.tag);
    int rb = numeric_rank(cb.tag);
    if (ra < 0 || rb < 0)
        throw std::invalid_argument("Arithmetic on string columns is not supported");

    const std::size_t epr_a = ca.elem_per_row();
    const std::size_t epr_b = cb.elem_per_row();
    const std::size_t target_epr = (epr_a >= epr_b) ? epr_a : epr_b;
    if (epr_a != epr_b && epr_a != 1 && epr_b != 1) {
        throw std::invalid_argument(
            "Arithmetic on columns with different list/matrix shapes is not supported");
    }

    DType rt = (ra >= rb) ? ca.tag : cb.tag;
    auto result = self->copy();

    if (rt == DType::Int) {
        std::vector<int> va = broadcast_for_binary_op(ca, ca.as<int>(), target_epr);
        std::vector<int> vb = broadcast_for_binary_op(cb, cb.as<int>(), target_epr);
        for (std::size_t i = 0; i < va.size(); ++i) va[i] = op(va[i], vb[i]);
        replace_last_with_same_shape(result, ca, va);
    } else if (rt == DType::Double) {
        std::vector<double> va = broadcast_for_binary_op(ca, to_double_vec(ca), target_epr);
        std::vector<double> vb = broadcast_for_binary_op(cb, to_double_vec(cb), target_epr);
        std::vector<double> vc(va.size());
        for (std::size_t i = 0; i < va.size(); ++i) vc[i] = op(va[i], vb[i]);
        replace_last_with_same_shape(result, ca, vc);
    } else {
        std::vector<DComplex> va = broadcast_for_binary_op(ca, to_complex_vec(ca), target_epr);
        std::vector<DComplex> vb = broadcast_for_binary_op(cb, to_complex_vec(cb), target_epr);
        std::vector<DComplex> vc(va.size());
        for (std::size_t i = 0; i < va.size(); ++i) vc[i] = op(va[i], vb[i]);
        replace_last_with_same_shape(result, ca, vc);
    }
    return result;
}

template <typename T>
std::shared_ptr<DataFrame> array_to_df(
    const std::shared_ptr<DataFrame>& self,
    const py::array& arr,
    const char* op_name) {
    auto buf = arr.request();
    if (buf.ndim != 1) {
        throw py::type_error(std::string(op_name) + " expects 1-D numpy array");
    }
    if (buf.size == 0) {
        throw std::invalid_argument("array is empty");
    }

    const T* ptr = static_cast<const T*>(buf.ptr);
    std::size_t n = static_cast<std::size_t>(buf.size);
    std::size_t nrows = self->num_rows();
    if (n != 1 && n != nrows) {
        throw std::invalid_argument(
            "array length (" + std::to_string(n) +
            ") must be 1 (scalar) or num_rows (" + std::to_string(nrows) + ")");
    }

    self->ensure_has_columns();

    const std::string& cname = self->last_column_name();
    const Column& shape_src = self->last_column();
    auto tmp = std::make_shared<DataFrame>();
    std::vector<T> values = (n == 1)
        ? std::vector<T>(nrows, ptr[0])
        : std::vector<T>(ptr, ptr + n);
    values = broadcast_rows_to_shape(values, shape_src);
    auto nc = make_column_like<T>(shape_src, values);
    nc->quantity = shape_src.quantity;
    tmp->add_column(cname, std::move(nc));
    return tmp;
}

std::shared_ptr<DataFrame> scalar_to_df(
    const std::shared_ptr<DataFrame>& self,
    py::handle value,
    const char* op_name) {
    self->ensure_has_columns();
    const std::string& cname = self->last_column_name();
    const std::size_t nrows = self->num_rows();
    const Column& shape_src = self->last_column();

    auto tmp = std::make_shared<DataFrame>();
    if (PyComplex_Check(value.ptr())) {
        auto c = py::cast<DComplex>(value);
        auto nc = make_column_like<DComplex>(shape_src, std::vector<DComplex>(nrows, c));
        nc->quantity = shape_src.quantity;
        tmp->add_column(cname, std::move(nc));
        return tmp;
    }
    if (py::isinstance<py::float_>(value)) {
        auto d = py::cast<double>(value);
        auto nc = make_column_like<double>(shape_src, std::vector<double>(nrows, d));
        nc->quantity = shape_src.quantity;
        tmp->add_column(cname, std::move(nc));
        return tmp;
    }
    if (py::isinstance<py::int_>(value) && !py::isinstance<py::bool_>(value)) {
        auto i = py::cast<int>(value);
        auto nc = make_column_like<int>(shape_src, std::vector<int>(nrows, i));
        nc->quantity = shape_src.quantity;
        tmp->add_column(cname, std::move(nc));
        return tmp;
    }

    // NumPy scalar support (e.g. numpy.int64/float64/complex128):
    // use 0-D dtype kind to preserve numeric intent.
    if (py::array arr = py::array::ensure(value)) {
        if (arr.ndim() == 0) {
            const std::string kind = py::str(arr.dtype().attr("kind")).cast<std::string>();
            if (!kind.empty()) {
                const char k = kind[0];
                if (k == 'i' || k == 'u') {
                    long long v = py::cast<long long>(value);
                    if (v < static_cast<long long>(std::numeric_limits<int>::min()) ||
                        v > static_cast<long long>(std::numeric_limits<int>::max())) {
                        throw py::type_error(
                            std::string(op_name) + " integer scalar out of int range");
                    }
                    auto nc = make_column_like<int>(shape_src, std::vector<int>(nrows, static_cast<int>(v)));
                    nc->quantity = shape_src.quantity;
                    tmp->add_column(cname, std::move(nc));
                    return tmp;
                }
                if (k == 'f') {
                    double d = py::cast<double>(value);
                    auto nc = make_column_like<double>(shape_src, std::vector<double>(nrows, d));
                    nc->quantity = shape_src.quantity;
                    tmp->add_column(cname, std::move(nc));
                    return tmp;
                }
                if (k == 'c') {
                    DComplex c = py::cast<DComplex>(value);
                    auto nc = make_column_like<DComplex>(shape_src, std::vector<DComplex>(nrows, c));
                    nc->quantity = shape_src.quantity;
                    tmp->add_column(cname, std::move(nc));
                    return tmp;
                }
            }
        }
    }

    throw py::type_error(
        std::string(op_name) +
        " expects DataFrame, scalar number, or 1-D numpy array");
}

std::shared_ptr<DataFrame> coerce_to_df(
    const std::shared_ptr<DataFrame>& self,
    py::handle value,
    const char* op_name) {
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
    const std::shared_ptr<DataFrame>& self,
    const std::function<double(double)>& fn_d,
    const std::function<double(const DComplex&)>& fn_c)
{
    self->ensure_has_columns();
    const Column& cc = self->last_column();
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
        auto result = self->copy();
        replace_last_with_same_shape(result, cc, out);
        return result;
}

std::shared_ptr<DataFrame> unary_promote(
    const std::shared_ptr<DataFrame>& self,
    const std::function<double(double)>& fn_d,
    const std::function<DComplex(const DComplex&)>& fn_c)
{
    self->ensure_has_columns();
    const Column& cc = self->last_column();
    auto r = self->copy();
    switch (cc.tag) {
        case DType::Int: {
            std::vector<double> out;
            out.reserve(cc.size());
            for (auto x : cc.as<int>()) out.push_back(fn_d(x));
            replace_last_with_same_shape(r, cc, out);
            break;
        }
        case DType::Double: {
            std::vector<double> d = r->last_column_as<double>();
            for (auto& x : d) x = fn_d(x);
            replace_last_with_same_shape(r, cc, d);
            break;
        }
        case DType::Complex: {
            std::vector<DComplex> d = r->last_column_as<DComplex>();
            for (auto& z : d) z = fn_c(z);
            replace_last_with_same_shape(r, cc, d);
            break;
        }
        case DType::String:
            throw std::invalid_argument("unary op: string columns not supported");
    }
    return r;
}

std::shared_ptr<DataFrame> negate_last(const std::shared_ptr<DataFrame>& df) {
    df->ensure_has_columns();
    auto r = df->copy();
    switch (df->last_column().tag) {
        case DType::Int: {
            std::vector<int> d = r->last_column_as<int>();
            for (auto& x : d) x = -x;
            replace_last_with_same_shape(r, df->last_column(), d); break;
        }
        case DType::Double: {
            std::vector<double> d = r->last_column_as<double>();
            for (auto& x : d) x = -x;
            replace_last_with_same_shape(r, df->last_column(), d); break;
        }
        case DType::Complex: {
            std::vector<DComplex> d = r->last_column_as<DComplex>();
            for (auto& z : d) z = -z;
            replace_last_with_same_shape(r, df->last_column(), d); break;
        }
        case DType::String:
            throw std::invalid_argument("unary -: string columns not supported");
    }
    return r;
}

template <typename T>
T require_arg(const py::args& args, int index) {
    if (index < 0) {
        throw py::index_error("argument index must be non-negative");
    }

    const std::size_t pos = static_cast<std::size_t>(index);
    if (pos >= args.size()) {
        throw py::index_error(
            "argument " + std::to_string(index) +
            " out of range (size=" + std::to_string(args.size()) + ")");
    }

    try {
        return args[pos].cast<T>();
    } catch (const py::cast_error&) {
        throw py::type_error(
            "argument " + std::to_string(index) + " has incorrect type");
    }
}

std::shared_ptr<DataFrame> df_handle_to_ptr(py::handle value) {
    try {
        return value.cast<std::shared_ptr<DataFrame>>();
    } catch (const py::cast_error&) {
        DataFrame& ref = value.cast<DataFrame&>();
        py::object owner = py::reinterpret_borrow<py::object>(value);
        return std::shared_ptr<DataFrame>(&ref, [owner](DataFrame*) {});
    }
}

std::shared_ptr<DataFrame> require_df_arg(const py::args& args, int index) {
    if (index < 0) {
        throw py::index_error("argument index must be non-negative");
    }

    const std::size_t pos = static_cast<std::size_t>(index);
    if (pos >= args.size()) {
        throw py::index_error(
            "argument " + std::to_string(index) +
            " out of range (size=" + std::to_string(args.size()) + ")");
    }

    py::handle value = args[pos];
    if (!py::isinstance<exprdf::DataFrame>(value)) {
        throw py::type_error("argument " + std::to_string(index) + " expects DataFrame");
    }
    return df_handle_to_ptr(value);
}

std::shared_ptr<DataFrame> require_rhs_df(
    const std::shared_ptr<DataFrame>& self,
    py::handle rhs,
    const char* op_name,
    std::shared_ptr<DataFrame>& owned) {
    if (py::isinstance<exprdf::DataFrame>(rhs)) {
        return df_handle_to_ptr(rhs);
    }
    owned = coerce_to_df(self, rhs, op_name);
    return owned;
}

py::object op_add(const py::args& args) {
    std::shared_ptr<DataFrame> df = require_df_arg(args, 0);
    std::shared_ptr<DataFrame> rhs_owned;
    std::shared_ptr<DataFrame> rhs = require_rhs_df(df, args[1], "add", rhs_owned);
    return py::cast(apply_binary_op_last(df, rhs, ArithAdd()));
}

py::object op_sub(const py::args& args) {
    std::shared_ptr<DataFrame> df = require_df_arg(args, 0);
    std::shared_ptr<DataFrame> rhs_owned;
    std::shared_ptr<DataFrame> rhs = require_rhs_df(df, args[1], "sub", rhs_owned);
    return py::cast(apply_binary_op_last(df, rhs, ArithSub()));
}

py::object op_mul(const py::args& args) {
    std::shared_ptr<DataFrame> df = require_df_arg(args, 0);
    std::shared_ptr<DataFrame> rhs_owned;
    std::shared_ptr<DataFrame> rhs = require_rhs_df(df, args[1], "mul", rhs_owned);
    return py::cast(apply_binary_op_last(df, rhs, ArithMul()));
}

py::object op_truediv(const py::args& args) {
    std::shared_ptr<DataFrame> df = require_df_arg(args, 0);
    std::shared_ptr<DataFrame> rhs_owned;
    std::shared_ptr<DataFrame> rhs = require_rhs_df(df, args[1], "truediv", rhs_owned);
    return py::cast(apply_binary_op_last(df, rhs, ArithDiv()));
}

py::object op_pow(const py::args& args) {
    std::shared_ptr<DataFrame> df = require_df_arg(args, 0);
    std::shared_ptr<DataFrame> rhs_owned;
    std::shared_ptr<DataFrame> rhs = require_rhs_df(df, args[1], "pow", rhs_owned);
    return py::cast(apply_binary_op_last(df, rhs, ArithPow()));
}

py::object op_radd(const py::args& args) {
    return op_add(args);
}

py::object op_rmul(const py::args& args) {
    return op_mul(args);
}

py::object op_rsub(const py::args& args) {
    std::shared_ptr<DataFrame> df = require_df_arg(args, 0);
    std::shared_ptr<DataFrame> lhs_owned;
    std::shared_ptr<DataFrame> lhs = require_rhs_df(df, args[1], "rsub", lhs_owned);
    return py::cast(apply_binary_op_last(lhs, df, ArithSub()));
}

py::object op_rtruediv(const py::args& args) {
    std::shared_ptr<DataFrame> df = require_df_arg(args, 0);
    std::shared_ptr<DataFrame> lhs_owned;
    std::shared_ptr<DataFrame> lhs = require_rhs_df(df, args[1], "rtruediv", lhs_owned);
    return py::cast(apply_binary_op_last(lhs, df, ArithDiv()));
}

py::object op_rpow(const py::args& args) {
    std::shared_ptr<DataFrame> df = require_df_arg(args, 0);
    std::shared_ptr<DataFrame> lhs_owned;
    std::shared_ptr<DataFrame> lhs = require_rhs_df(df, args[1], "rpow", lhs_owned);
    return py::cast(apply_binary_op_last(lhs, df, ArithPow()));
}

py::object op_neg(const py::args& args) {
    return py::cast(negate_last(require_df_arg(args, 0)));
}

py::object op_abs(const py::args& args) {
    std::shared_ptr<DataFrame> df = require_df_arg(args, 0);
    df->ensure_has_columns();
    const Column& cc = df->last_column();
    auto r = df->copy();
    switch (cc.tag) {
        case DType::Int: {
            std::vector<int> d = cc.as<int>();
            for (auto& x : d) x = std::abs(x);
            replace_last_with_same_shape(r, cc, d); break;
        }
        case DType::Double: {
            std::vector<double> d = cc.as<double>();
            for (auto& x : d) x = std::abs(x);
            replace_last_with_same_shape(r, cc, d); break;
        }
        case DType::Complex: {
            const auto& src = cc.as<DComplex>();
            std::vector<double> out(src.size());
            for (std::size_t i = 0; i < src.size(); ++i) out[i] = std::abs(src[i]);
            replace_last_with_same_shape(r, cc, out);
            break;
        }
        case DType::String:
            throw std::invalid_argument("abs: string columns not supported");
    }
    return py::cast(r);
}

py::object op_mag(const py::args& args) { return op_abs(args); }

py::object op_real(const py::args& args) {
    std::shared_ptr<DataFrame> df = require_df_arg(args, 0);
    return py::cast(unary_to_double(
        df,
        [](double x) { return x; },
        [](const DComplex& z) { return z.real(); }));
}

py::object op_imag(const py::args& args) {
    std::shared_ptr<DataFrame> df = require_df_arg(args, 0);
    return py::cast(unary_to_double(
        df,
        [](double) { return 0.0; },
        [](const DComplex& z) { return z.imag(); }));
}

py::object op_phase(const py::args& args) {
    std::shared_ptr<DataFrame> df = require_df_arg(args, 0);
    return py::cast(unary_to_double(
        df,
        [](double x) { return std::atan2(0.0, x); },
        [](const DComplex& z) { return std::arg(z); }));
}

py::object op_dB(const py::args& args) {
    std::shared_ptr<DataFrame> df = require_df_arg(args, 0);
    return py::cast(unary_to_double(
        df,
        [](double x) { return 20.0 * std::log10(std::abs(x)); },
        [](const DComplex& z) { return 20.0 * std::log10(std::abs(z)); }));
}

py::object op_dBm(const py::args& args) {
    std::shared_ptr<DataFrame> df = require_df_arg(args, 0);
    return py::cast(unary_to_double(
        df,
        [](double x) { return 20.0 * std::log10(std::abs(x)) + 10.0; },
        [](const DComplex& z) { return 20.0 * std::log10(std::abs(z)) + 10.0; }));
}

py::object op_wtodBm(const py::args& args) {
    std::shared_ptr<DataFrame> df = require_df_arg(args, 0);
    df->ensure_has_columns();
    const Column& cc = df->last_column();
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
    auto result = df->copy();
    replace_last_with_same_shape(result, cc, out);
    return py::cast(result);
}

py::object op_sqr(const py::args& args) {
    std::shared_ptr<DataFrame> df = require_df_arg(args, 0);
    df->ensure_has_columns();
    auto r = df->copy();
    const Column& cc = df->last_column();
    switch (cc.tag) {
        case DType::Int: {
            std::vector<int> d = cc.as<int>();
            for (auto& x : d) x = x * x;
            replace_last_with_same_shape(r, cc, d); break;
        }
        case DType::Double: {
            std::vector<double> d = cc.as<double>();
            for (auto& x : d) x = x * x;
            replace_last_with_same_shape(r, cc, d); break;
        }
        case DType::Complex: {
            std::vector<DComplex> d = cc.as<DComplex>();
            for (auto& z : d) z = z * z;
            replace_last_with_same_shape(r, cc, d); break;
        }
        case DType::String:
            throw std::invalid_argument("sqr: string columns not supported");
    }
    return py::cast(r);
}

py::object op_sqrt(const py::args& args) {
    std::shared_ptr<DataFrame> df = require_df_arg(args, 0);
    return py::cast(unary_promote(
        df,
        [](double x) { return std::sqrt(x); },
        [](const DComplex& z) { return std::sqrt(z); }));
}

py::object op_exp(const py::args& args) {
    std::shared_ptr<DataFrame> df = require_df_arg(args, 0);
    return py::cast(unary_promote(
        df,
        [](double x) { return std::exp(x); },
        [](const DComplex& z) { return std::exp(z); }));
}

py::object op_ln(const py::args& args) {
    std::shared_ptr<DataFrame> df = require_df_arg(args, 0);
    return py::cast(unary_promote(
        df,
        [](double x) { return std::log(x); },
        [](const DComplex& z) { return std::log(z); }));
}

py::object op_log10(const py::args& args) {
    std::shared_ptr<DataFrame> df = require_df_arg(args, 0);
    return py::cast(unary_promote(
        df,
        [](double x) { return std::log10(x); },
        [](const DComplex& z) { return std::log10(z); }));
}

py::object op_conj(const py::args& args) {
    std::shared_ptr<DataFrame> df = require_df_arg(args, 0);
    df->ensure_has_columns();
    const Column& cc = df->last_column();
    auto r = df->copy();
    switch (cc.tag) {
        case DType::Int:
        case DType::Double:
            break;
        case DType::Complex: {
            std::vector<DComplex> d = cc.as<DComplex>();
            for (auto& z : d) z = std::conj(z);
            replace_last_with_same_shape(r, cc, d); break;
        }
        case DType::String:
            throw std::invalid_argument("conj: string columns not supported");
    }
    return py::cast(r);
}

py::object op_zin(const py::args& args) {
    std::shared_ptr<DataFrame> df = require_df_arg(args, 0);
    const DComplex z0 = (args.size() >= 2) ? args[1].cast<DComplex>() : DComplex(50.0, 0.0);
    df->ensure_has_columns("zin: DataFrame");

    const Column& cc = df->last_column();
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

    auto result = df->copy();
    replace_last_with_same_shape(result, cc, out);
    return py::cast(result);
}

// ---------------------------------------------------------------
// vs / plot_vs -- attach an independent variable for (re-)plotting
//
// Mirror of ADS' vs() / plot_vs():
//   vs(dependent, independent[, indepName]): plain data constructor.
//     Builds a new dataset whose X axis (independent) is `independent`
//     and whose data is the last column of `dependent`.
//   plot_vs(dependent, independent): plot-directed remap.
//     * If `independent` is one of `dependent`'s own independent (index)
//       dimensions, the data is swapped so that dimension becomes the
//       innermost (X-axis) dimension.  (e.g. S-param db(S11) with
//       dependency [Cval, Freq] and plot_vs(dbS11, Cval) -> [Freq, Cval]).
//     * If it is a dissimilar vector whose size matches an existing axis,
//       that axis' values are spliced with `independent` and moved innermost
//       (e.g. CvalH = Cval/2; plot_vs(dbS11, CvalH)).
//     * Otherwise (same size as the whole dataset) it behaves like vs().
// ---------------------------------------------------------------

// Row permutation that re-orders rows so that index dims appear in
// `target_order` (old dim ordinal indices) with row-major layout
// (last dim = innermost / X axis).  Requires uniform (Cartesian) or
// regular-grouped dims; ragged (irregular) dims are rejected.
std::vector<std::size_t> permute_rows_to_dims_order(
    const std::shared_ptr<DataFrame>& df,
    const std::vector<std::size_t>& target_order) {
    const std::size_t n = df->num_indices();
    if (n == 0) return {};
    if (target_order.size() != n)
        throw std::invalid_argument("internal: dimension order size mismatch");

    std::vector<std::size_t> lc(n); // level counts for stride math
    for (std::size_t i = 0; i < n; ++i) {
        const IndexDim& d = df->get_index_dim(i);
        if (!d.is_uniform() && !d.is_regular_grouped())
            throw std::invalid_argument(
                "plot_vs: dimension permutation on a ragged index ('" + d.name +
                "') is not supported");
        lc[i] = d.level_count();
        if (lc[i] == 0)
            throw std::invalid_argument("plot_vs: zero-size index dimension");
    }

    const std::vector<std::size_t> old_strides = df->strides();
    std::vector<std::size_t> new_strides(n); // strides in the new order
    new_strides[n - 1] = 1;
    for (std::size_t j = n - 1; j > 0; --j)
        new_strides[j - 1] = new_strides[j] * lc[target_order[j]];

    const std::size_t N = df->num_rows();
    std::vector<std::size_t> new_to_old(N);
    for (std::size_t nf = 0; nf < N; ++nf) {
        std::size_t rem = nf, of = 0;
        for (std::size_t j = 0; j < n; ++j) {
            const std::size_t ord = rem / new_strides[j];
            rem %= new_strides[j];
            of += ord * old_strides[target_order[j]];
        }
        new_to_old[nf] = of;
    }
    return new_to_old;
}

struct AxisSplice {
    std::string old_name;   // column being replaced (an existing axis)
    std::string new_name;   // its new name (the independent's column name)
    std::unique_ptr<Column> values; // replacement column (full-length, original layout)
};

// Rebuild a DataFrame from `df` by gathering rows through `new_to_old` and
// then promoting `new_dim_names` (in order) back to index dimensions.
// Optional `splice` replaces one column (old_name) with new values (new_name).
std::shared_ptr<DataFrame> rebuild_permuted(
    const std::shared_ptr<DataFrame>& df,
    const std::vector<std::size_t>& new_to_old,
    const std::vector<std::string>& new_dim_names,
    const AxisSplice* splice = nullptr) {
    auto result = std::make_shared<DataFrame>();
    for (const auto& cname : df->column_names()) {
        if (splice && cname == splice->old_name) {
            Column gathered = splice->values->gather(new_to_old);
            result->add_column(splice->new_name,
                               std::unique_ptr<Column>(new Column(std::move(gathered))));
            continue;
        }
        Column src = df->get_column(cname).clone();
        Column gathered = src.gather(new_to_old);
        result->add_column(cname, std::unique_ptr<Column>(new Column(std::move(gathered))));
    }
    if (!new_dim_names.empty())
        result->set_index(new_dim_names);
    return result;
}

// Broadcast an independent column's scalar values onto the full row count,
// repeating them along the given axis dimension.  Result is a scalar column
// of num_rows() elements in the ORIGINAL layout (caller gathers afterwards).
std::unique_ptr<Column> scatter_axis_values(
    const std::shared_ptr<DataFrame>& df,
    std::size_t axis_dim,
    const Column& indep_col) {
    if (!indep_col.shape.empty())
        throw std::invalid_argument("plot_vs: independent values must be a scalar column");
    const std::size_t L = df->get_index_dim(axis_dim).level_count();
    if (indep_col.num_rows() != L)
        throw std::invalid_argument(
            "plot_vs: independent row count (" + std::to_string(indep_col.num_rows()) +
            ") does not match axis dimension size (" + std::to_string(L) + ")");
    const std::size_t N = df->num_rows();
    const std::vector<std::size_t> strides = df->strides();
    const std::size_t stride = strides[axis_dim];

    if (indep_col.tag == DType::String) {
        const auto& vals = indep_col.as<std::string>();
        std::vector<std::string> out(N);
        for (std::size_t r = 0; r < N; ++r) out[r] = vals[(r / stride) % L];
        auto c = make_column<std::string>(out);
        c->quantity = indep_col.quantity;
        return c;
    }
    if (indep_col.tag == DType::Complex) {
        const auto& vals = indep_col.as<DComplex>();
        std::vector<DComplex> out(N);
        for (std::size_t r = 0; r < N; ++r) out[r] = vals[(r / stride) % L];
        auto c = make_column<DComplex>(out);
        c->quantity = indep_col.quantity;
        return c;
    }
    // int / double -> promote to double
    std::vector<double> vals = to_double_vec(indep_col);
    std::vector<double> out(N);
    for (std::size_t r = 0; r < N; ++r) out[r] = vals[(r / stride) % L];
    auto c = make_column<double>(out);
    c->quantity = indep_col.quantity;
    return c;
}

// Fresh attach (vs() semantics): new dataset with `independent` as the sole
// index dimension and the dependent's last column as the data.
std::shared_ptr<DataFrame> attach_independent(
    const std::shared_ptr<DataFrame>& dependent,
    const std::shared_ptr<DataFrame>& independent,
    const std::string& name_override) {
    dependent->ensure_has_columns("vs/plot_vs: dependent DataFrame");
    independent->ensure_has_columns("vs/plot_vs: independent DataFrame");

    const std::string indep_src_name = independent->last_column_name();
    const std::string indep_name = name_override.empty() ? indep_src_name : name_override;
    const Column& indep_col = independent->get_column(indep_src_name);
    if (!indep_col.shape.empty())
        throw std::invalid_argument(
            "vs/plot_vs: independent values must be a scalar column (got a list/matrix column)");
    if (indep_col.num_rows() == 0)
        throw std::invalid_argument("vs/plot_vs: independent values cannot be empty");
    if (dependent->num_rows() != indep_col.num_rows())
        throw std::invalid_argument(
            "vs/plot_vs: row count mismatch: dependent has " +
            std::to_string(dependent->num_rows()) +
            " rows, independent has " + std::to_string(indep_col.num_rows()) + " rows");

    auto result = std::make_shared<DataFrame>();
    result->add_uniform_index_column(indep_name,
        std::unique_ptr<Column>(new Column(indep_col.clone())), indep_col.quantity);

    const std::string dep_name = dependent->last_column_name();
    const Column& dep_col = dependent->get_column(dep_name);
    result->add_column(dep_name, std::unique_ptr<Column>(new Column(dep_col.clone())));
    return result;
}

py::object op_vs(const py::args& args) {
    std::shared_ptr<DataFrame> dependent = require_df_arg(args, 0);
    std::shared_ptr<DataFrame> independent = require_df_arg(args, 1);
    std::string name_override;
    if (args.size() >= 3) name_override = args[2].cast<std::string>();
    return py::cast(attach_independent(dependent, independent, name_override));
}

py::object op_plot_vs(const py::args& args) {
    std::shared_ptr<DataFrame> dependent = require_df_arg(args, 0);
    std::shared_ptr<DataFrame> independent = require_df_arg(args, 1);
    dependent->ensure_has_columns("plot_vs: dependent DataFrame");
    independent->ensure_has_columns("plot_vs: independent DataFrame");

    const std::string indep_name = independent->last_column_name();
    std::vector<std::string> dnames = dependent->index_names();
    const std::size_t n = dnames.size();

    // Case A: `independent` is one of dependent's own independent dimensions.
    // Swap/slice so that it becomes the innermost (X-axis) dimension.
    for (std::size_t t = 0; t < n; ++t) {
        if (dnames[t] != indep_name) continue;
        std::vector<std::size_t> order;
        std::vector<std::string> new_names;
        for (std::size_t i = 0; i < n; ++i) {
            if (i == t) continue;
            order.push_back(i);
            new_names.push_back(dnames[i]);
        }
        order.push_back(t);
        new_names.push_back(indep_name);
        std::vector<std::size_t> perm = permute_rows_to_dims_order(dependent, order);
        return py::cast(rebuild_permuted(dependent, perm, new_names));
    }

    // Case B: independent is dissimilar to dependent's own independents.
    //   a) Its size matches an existing axis -> splice values onto that axis.
    //   b) Its size matches the whole dataset -> fresh attach like vs().
    const std::size_t indep_rows = independent->num_rows();
    std::size_t axis = n; // sentinel: not found
    for (std::size_t i = 0; i < n && axis == n; ++i) {
        const IndexDim& d = dependent->get_index_dim(i);
        if (d.level_count() == indep_rows) axis = i;
    }
    if (axis != n) {
        const std::string axis_name = dependent->get_index_dim(axis).name;
        const Column& indep_col = independent->get_column(indep_name);

        std::vector<std::size_t> order;
        std::vector<std::string> new_names;
        for (std::size_t i = 0; i < n; ++i) {
            if (i == axis) continue;
            order.push_back(i);
            new_names.push_back(dnames[i]);
        }
        order.push_back(axis);
        new_names.push_back(indep_name);

        std::vector<std::size_t> perm = permute_rows_to_dims_order(dependent, order);
        AxisSplice splice;
        splice.old_name = axis_name;
        splice.new_name = indep_name;
        splice.values = scatter_axis_values(dependent, axis, indep_col);
        return py::cast(rebuild_permuted(dependent, perm, new_names, &splice));
    }

    return py::cast(attach_independent(dependent, independent, ""));
}

const std::vector<OpSpec>& all_ops() {
    static const std::vector<OpSpec> specs = {
        make_internal_op("add", {{ArgType::DataFrame, "df"}, {ArgType::Any, "rhs"}}, 2, 2, &op_add, "internal add"),
        make_internal_op("sub", {{ArgType::DataFrame, "df"}, {ArgType::Any, "rhs"}}, 2, 2, &op_sub, "internal sub"),
        make_internal_op("mul", {{ArgType::DataFrame, "df"}, {ArgType::Any, "rhs"}}, 2, 2, &op_mul, "internal mul"),
        make_internal_op("truediv", {{ArgType::DataFrame, "df"}, {ArgType::Any, "rhs"}}, 2, 2, &op_truediv, "internal truediv"),
        make_internal_op("pow", {{ArgType::DataFrame, "df"}, {ArgType::Any, "rhs"}}, 2, 2, &op_pow, "internal pow"),
        make_internal_op("radd", {{ArgType::DataFrame, "df"}, {ArgType::Any, "lhs"}}, 2, 2, &op_radd, "internal radd"),
        make_internal_op("rsub", {{ArgType::DataFrame, "df"}, {ArgType::Any, "lhs"}}, 2, 2, &op_rsub, "internal rsub"),
        make_internal_op("rmul", {{ArgType::DataFrame, "df"}, {ArgType::Any, "lhs"}}, 2, 2, &op_rmul, "internal rmul"),
        make_internal_op("rtruediv", {{ArgType::DataFrame, "df"}, {ArgType::Any, "lhs"}}, 2, 2, &op_rtruediv, "internal rtruediv"),
        make_internal_op("rpow", {{ArgType::DataFrame, "df"}, {ArgType::Any, "lhs"}}, 2, 2, &op_rpow, "internal rpow"),
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
        make_export_op_df_first("zin", {{ArgType::Complex, "z0"}}, 1, 2, &op_zin, "zin(df, z0=50): input impedance Zin = Z0*(1+S11)/(1-S11) on last column"),
        make_op("vs",
            {{ArgType::DataFrame, "dependent"}, {ArgType::DataFrame, "independent"}, {ArgType::String, "indepName"}},
            2, 3, &op_vs,
            "vs(dependent, independent[, indepName]): build a new dataset where 'independent' is the independent (X-axis) and the last column of 'dependent' is the data",
            true, ModuleBindMode::DataFrameFirstArgs),
        make_op("plot_vs",
            {{ArgType::DataFrame, "dependent"}, {ArgType::DataFrame, "independent"}},
            2, 2, &op_plot_vs,
            "plot_vs(dependent, independent): returns a new dataset re-oriented so that 'independent' (an existing index of 'dependent', a vector matching one axis size, or a full-size vector) is the innermost (X-axis) independent. Change of plot direction with a dissimilar vector uses the values of the argument as the new axis values.",
            true, ModuleBindMode::DataFrameFirstArgs)
    };
    return specs;
}

py::object invoke_df_first(
    const std::string& name,
    const std::shared_ptr<DataFrame>& df,
    const py::args& rest) {
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
                [op_name](std::shared_ptr<exprdf::DataFrame> df, py::args args) {
                    return invoke_df_first(op_name, df, args);
                },
                py::arg("df"),
                spec.doc);
            if (lower_name != op_name) {
                m.def(
                    lower_name.c_str(),
                    [op_name](std::shared_ptr<exprdf::DataFrame> df, py::args args) {
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
