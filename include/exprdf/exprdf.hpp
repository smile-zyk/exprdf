#ifndef EXPRDF_HPP
#define EXPRDF_HPP

// exprdf.hpp — header-only DataFrame library
//
// Supports 4 column types: int, double, string, complex<double>.
// Multi-index (Cartesian product) for N-dimensional parameter sweeps.
// No RTTI, no virtual dispatch.

#include <string>
#include <vector>
#include <unordered_map>
#include <complex>
#include <cmath>
#include <stdexcept>
#include <sstream>
#include <algorithm>
#include <memory>
#include <cassert>
#include <iomanip>
#include <fstream>
#include <exprdf/UnitFormat.hpp>

namespace exprdf {

// ============================================================
// DType — column type tag
// ============================================================
enum class DType { Int, Double, String, Complex };

inline const char* dtype_to_string(DType t) {
    switch (t) {
        case DType::Int:           return "int";
        case DType::Double:        return "double";
        case DType::String:        return "string";
        case DType::Complex:  return "complex";
    }
    return "unknown";
}

// DTypeTag<T>: compile-time mapping from C++ type to DType
template <typename T> struct DTypeTag;
template <> struct DTypeTag<int>                    { static const DType value = DType::Int; };
template <> struct DTypeTag<double>                 { static const DType value = DType::Double; };
template <> struct DTypeTag<std::string>            { static const DType value = DType::String; };
template <> struct DTypeTag<std::complex<double>>   { static const DType value = DType::Complex; };

// Approximate equality for floating-point comparisons
inline bool approx_equal(double a, double b, double eps = 1e-12) {
    double diff = std::abs(a - b);
    if (diff <= eps) return true;
    return diff <= eps * std::max(std::abs(a), std::abs(b));
}

inline bool approx_equal(const std::complex<double>& a,
                         const std::complex<double>& b, double eps = 1e-12) {
    return approx_equal(a.real(), b.real(), eps) &&
           approx_equal(a.imag(), b.imag(), eps);
}

// Type-aware equality: approx for floating-point, exact for others
template <typename T>
inline bool values_equal(const T& a, const T& b) { return a == b; }
template <>
inline bool values_equal<double>(const double& a, const double& b) { return approx_equal(a, b); }
template <>
inline bool values_equal<std::complex<double>>(const std::complex<double>& a, const std::complex<double>& b) { return approx_equal(a, b); }

// ============================================================
// Column — tagged union of 4 vector types
// ============================================================
class Column {
public:
    DType tag;
    std::string quantity;   // quantity key, e.g. "voltage" or "frequency"

    Column() : tag(DType::Int) {}

    // Construction helpers
    static Column from_int(const std::vector<int>& v)                       { Column c; c.tag = DType::Int; c.ints_ = v; return c; }
    static Column from_double(const std::vector<double>& v)                 { Column c; c.tag = DType::Double; c.doubles_ = v; return c; }
    static Column from_string(const std::vector<std::string>& v)            { Column c; c.tag = DType::String; c.strings_ = v; return c; }
    static Column from_complex(const std::vector<std::complex<double>>& v)  { Column c; c.tag = DType::Complex; c.complexes_ = v; return c; }

    std::size_t size() const {
        switch (tag) {
            case DType::Int:           return ints_.size();
            case DType::Double:        return doubles_.size();
            case DType::String:        return strings_.size();
            case DType::Complex:  return complexes_.size();
        }
        return 0;
    }

    std::string dtype_name() const { return dtype_to_string(tag); }

    std::string to_string(std::size_t row) const {
        std::ostringstream ss;
        switch (tag) {
            case DType::Int:           ss << ints_.at(row); break;
            case DType::Double:        ss << doubles_.at(row); break;
            case DType::String:        return strings_.at(row);
            case DType::Complex: {
                const auto& v = complexes_.at(row);
                ss << "(" << v.real();
                if (v.imag() >= 0) ss << "+";
                ss << v.imag() << "j)";
                break;
            }
        }
        return ss.str();
    }

    Column clone() const {
        Column c;
        c.tag = tag;
        c.quantity = quantity;
        c.ints_ = ints_;
        c.doubles_ = doubles_;
        c.strings_ = strings_;
        c.complexes_ = complexes_;
        return c;
    }

    Column slice(std::size_t start, std::size_t end) const {
        Column c;
        c.tag = tag;
        c.quantity = quantity;
        switch (tag) {
            case DType::Int:
                c.ints_.assign(ints_.begin() + start, ints_.begin() + end); break;
            case DType::Double:
                c.doubles_.assign(doubles_.begin() + start, doubles_.begin() + end); break;
            case DType::String:
                c.strings_.assign(strings_.begin() + start, strings_.begin() + end); break;
            case DType::Complex:
                c.complexes_.assign(complexes_.begin() + start, complexes_.begin() + end); break;
        }
        return c;
    }

    // Repeat each element n times: [a,b] repeat_each(3) -> [a,a,a,b,b,b]
    Column repeat_each(std::size_t n) const {
        Column c;
        c.tag = tag;
        c.quantity = quantity;
        switch (tag) {
            case DType::Int:
                c.ints_.reserve(ints_.size() * n);
                for (const auto& v : ints_)
                    for (std::size_t i = 0; i < n; ++i) c.ints_.push_back(v);
                break;
            case DType::Double:
                c.doubles_.reserve(doubles_.size() * n);
                for (const auto& v : doubles_)
                    for (std::size_t i = 0; i < n; ++i) c.doubles_.push_back(v);
                break;
            case DType::String:
                c.strings_.reserve(strings_.size() * n);
                for (const auto& v : strings_)
                    for (std::size_t i = 0; i < n; ++i) c.strings_.push_back(v);
                break;
            case DType::Complex:
                c.complexes_.reserve(complexes_.size() * n);
                for (const auto& v : complexes_)
                    for (std::size_t i = 0; i < n; ++i) c.complexes_.push_back(v);
                break;
        }
        return c;
    }

    // Gather specific rows by index
    Column gather(const std::vector<std::size_t>& row_indices) const {
        Column c;
        c.tag = tag;
        c.quantity = quantity;
        switch (tag) {
            case DType::Int:
                c.ints_.reserve(row_indices.size());
                for (auto i : row_indices) c.ints_.push_back(ints_.at(i));
                break;
            case DType::Double:
                c.doubles_.reserve(row_indices.size());
                for (auto i : row_indices) c.doubles_.push_back(doubles_.at(i));
                break;
            case DType::String:
                c.strings_.reserve(row_indices.size());
                for (auto i : row_indices) c.strings_.push_back(strings_.at(i));
                break;
            case DType::Complex:
                c.complexes_.reserve(row_indices.size());
                for (auto i : row_indices) c.complexes_.push_back(complexes_.at(i));
                break;
        }
        return c;
    }

    // Tile entire sequence n times: [a,b] tile(3) -> [a,b,a,b,a,b]
    Column tile(std::size_t n) const {
        Column c;
        c.tag = tag;
        c.quantity = quantity;
        switch (tag) {
            case DType::Int:
                c.ints_.reserve(ints_.size() * n);
                for (std::size_t i = 0; i < n; ++i)
                    for (const auto& v : ints_) c.ints_.push_back(v);
                break;
            case DType::Double:
                c.doubles_.reserve(doubles_.size() * n);
                for (std::size_t i = 0; i < n; ++i)
                    for (const auto& v : doubles_) c.doubles_.push_back(v);
                break;
            case DType::String:
                c.strings_.reserve(strings_.size() * n);
                for (std::size_t i = 0; i < n; ++i)
                    for (const auto& v : strings_) c.strings_.push_back(v);
                break;
            case DType::Complex:
                c.complexes_.reserve(complexes_.size() * n);
                for (std::size_t i = 0; i < n; ++i)
                    for (const auto& v : complexes_) c.complexes_.push_back(v);
                break;
        }
        return c;
    }

    // Extract unique values preserving first-appearance order
    Column extract_unique() const {
        Column c;
        c.tag = tag;
        c.quantity = quantity;
        switch (tag) {
            case DType::Int:
                for (auto& v : ints_)
                    if (std::find(c.ints_.begin(), c.ints_.end(), v) == c.ints_.end())
                        c.ints_.push_back(v);
                break;
            case DType::Double:
                for (auto& v : doubles_) {
                    bool found = false;
                    for (auto& u : c.doubles_)
                        if (approx_equal(u, v)) { found = true; break; }
                    if (!found) c.doubles_.push_back(v);
                }
                break;
            case DType::String:
                for (auto& v : strings_)
                    if (std::find(c.strings_.begin(), c.strings_.end(), v) == c.strings_.end())
                        c.strings_.push_back(v);
                break;
            case DType::Complex:
                for (auto& v : complexes_) {
                    bool found = false;
                    for (auto& u : c.complexes_)
                        if (approx_equal(u, v)) { found = true; break; }
                    if (!found) c.complexes_.push_back(v);
                }
                break;
        }
        return c;
    }

    // Compare value at row_a in this column with value at row_b in another column
    bool value_equals(std::size_t row_a, const Column& other, std::size_t row_b) const {
        if (tag != other.tag) return false;
        switch (tag) {
            case DType::Int:     return ints_[row_a] == other.ints_[row_b];
            case DType::Double:  return approx_equal(doubles_[row_a], other.doubles_[row_b]);
            case DType::String:  return strings_[row_a] == other.strings_[row_b];
            case DType::Complex: return approx_equal(complexes_[row_a], other.complexes_[row_b]);
        }
        return false;
    }

    // Typed accessors (caller must ensure tag matches T)
    template <typename T> const std::vector<T>& as() const;
    template <typename T> std::vector<T>& as();

private:
    std::vector<int>                    ints_;
    std::vector<double>                 doubles_;
    std::vector<std::string>            strings_;
    std::vector<std::complex<double>>   complexes_;
};

// ============================================================
// Column — template specializations
// ============================================================
template <> inline const std::vector<int>&                    Column::as<int>()                    const { return ints_; }
template <> inline const std::vector<double>&                 Column::as<double>()                 const { return doubles_; }
template <> inline const std::vector<std::string>&            Column::as<std::string>()            const { return strings_; }
template <> inline const std::vector<std::complex<double>>&   Column::as<std::complex<double>>()   const { return complexes_; }

template <> inline std::vector<int>&                    Column::as<int>()                    { return ints_; }
template <> inline std::vector<double>&                 Column::as<double>()                 { return doubles_; }
template <> inline std::vector<std::string>&            Column::as<std::string>()            { return strings_; }
template <> inline std::vector<std::complex<double>>&   Column::as<std::complex<double>>()   { return complexes_; }

// make_column<T>: construct Column from typed vector
template <typename T>
inline Column make_column(const std::vector<T>& data);

template <> inline Column make_column<int>(const std::vector<int>& v)                                   { return Column::from_int(v); }
template <> inline Column make_column<double>(const std::vector<double>& v)                              { return Column::from_double(v); }
template <> inline Column make_column<std::string>(const std::vector<std::string>& v)                    { return Column::from_string(v); }
template <> inline Column make_column<std::complex<double>>(const std::vector<std::complex<double>>& v)  { return Column::from_complex(v); }

// ============================================================
// DataFrame — tabular data with optional multi-index
// ============================================================

// Index dimension kind
enum class IndexKind { Uniform, Varying };

// Index dimension descriptor
struct IndexDim {
    std::string name;
    IndexKind kind;
    Column levels;              // Uniform: unique level values
    std::size_t group_size;     // Varying: per-group element count

    std::size_t dim_size() const {
        return kind == IndexKind::Uniform ? levels.size() : group_size;
    }

    bool is_uniform() const { return kind == IndexKind::Uniform; }
    bool is_varying() const { return kind == IndexKind::Varying; }

    template <typename T>
    static IndexDim create_uniform(const std::string& name,
                                   const std::vector<T>& lvls,
                                   const std::string& quantity = unit_format::quantity::unitless) {
        Column col = make_column<T>(lvls);
        col.quantity = quantity;
        return {name, IndexKind::Uniform, col, 0};
    }

    static IndexDim create_uniform(const std::string& name, Column lvls) {
        return {name, IndexKind::Uniform, lvls, 0};
    }

    static IndexDim create_varying(const std::string& name,
                                   std::size_t gs,
                                   const std::string& quantity = unit_format::quantity::unitless) {
        Column col;
        col.quantity = quantity;
        return {name, IndexKind::Varying, col, gs};
    }
};

class DataFrame {
public:
    DataFrame() {}

    // --- Column management ---

    template <typename T>
    void add_column(const std::string& name, const std::vector<T>& data,
                    const std::string& quantity = unit_format::quantity::unitless) {
        if (!columns_.empty()) {
            if (data.size() != num_rows()) {
                throw std::invalid_argument(
                    "Column '" + name + "' has " + std::to_string(data.size()) +
                    " rows, expected " + std::to_string(num_rows()));
            }
        }
        if (has_column(name)) {
            throw std::invalid_argument("Column '" + name + "' already exists");
        }
        col_order_.push_back(name);
        Column col = make_column<T>(data);
        col.quantity = quantity;
        columns_[name] = col;
    }

    template <typename T>
    void insert_column(std::size_t pos, const std::string& name,
                       const std::vector<T>& data,
                       const std::string& quantity = unit_format::quantity::unitless) {
        if (!columns_.empty()) {
            if (data.size() != num_rows()) {
                throw std::invalid_argument(
                    "Column '" + name + "' has " + std::to_string(data.size()) +
                    " rows, expected " + std::to_string(num_rows()));
            }
        }
        if (has_column(name)) {
            throw std::invalid_argument("Column '" + name + "' already exists");
        }
        if (pos > col_order_.size()) {
            throw std::out_of_range(
                "Insert position " + std::to_string(pos) +
                " out of range (size=" + std::to_string(col_order_.size()) + ")");
        }
        col_order_.insert(col_order_.begin() + pos, name);
        Column col = make_column<T>(data);
        col.quantity = quantity;
        columns_[name] = col;
    }

    template <typename T>
    void prepend_column(const std::string& name, const std::vector<T>& data,
                        const std::string& quantity = unit_format::quantity::unitless) {
        insert_column<T>(0, name, data, quantity);
    }

    void remove_column(const std::string& name) {
        auto it = columns_.find(name);
        if (it == columns_.end()) {
            throw std::invalid_argument("Column '" + name + "' not found");
        }
        columns_.erase(it);
        col_order_.erase(
            std::remove(col_order_.begin(), col_order_.end(), name),
            col_order_.end());
    }

    bool has_column(const std::string& name) const {
        return columns_.find(name) != columns_.end();
    }

    std::vector<std::string> column_names() const {
        return col_order_;
    }

    // --- Column access (by position) ---

    const std::string& column_name(std::size_t index) const {
        if (index >= col_order_.size()) {
            throw std::out_of_range(
                "Column index " + std::to_string(index) +
                " out of range (size=" + std::to_string(col_order_.size()) + ")");
        }
        return col_order_[index];
    }

    const Column& get_column(std::size_t index) const {
        return get_col(column_name(index));
    }

    Column& get_column(std::size_t index) {
        return get_col(column_name(index));
    }

    template <typename T>
    const std::vector<T>& get_column_as(std::size_t index) const {
        return get_column_as<T>(column_name(index));
    }

    template <typename T>
    std::vector<T>& get_column_as(std::size_t index) {
        return get_column_as<T>(column_name(index));
    }

    // --- Column access (by name) ---

    const Column& get_column(const std::string& name) const {
        return get_col(name);
    }

    Column& get_column(const std::string& name) {
        return get_col(name);
    }

    template <typename T>
    const std::vector<T>& get_column_as(const std::string& name) const {
        const Column& col = get_col(name);
        if (col.tag != DTypeTag<T>::value) {
            throw std::invalid_argument(
                "Column '" + name + "' type mismatch, stored as " + col.dtype_name());
        }
        return col.as<T>();
    }

    template <typename T>
    std::vector<T>& get_column_as(const std::string& name) {
        Column& col = get_col(name);
        if (col.tag != DTypeTag<T>::value) {
            throw std::invalid_argument(
                "Column '" + name + "' type mismatch, stored as " + col.dtype_name());
        }
        return col.as<T>();
    }

    std::string column_dtype(const std::string& name) const {
        return get_col(name).dtype_name();
    }

    std::string column_quantity(const std::string& name) const {
        return get_col(name).quantity;
    }

    void set_column_quantity(const std::string& name, const std::string& q) {
        get_col(name).quantity = q;
    }

    // --- Shape & element access ---

    std::size_t num_rows() const {
        if (columns_.empty()) return 0;
        return columns_.begin()->second.size();
    }

    std::size_t num_columns() const {
        return columns_.size();
    }
    template <typename T>
    T at(const std::string& col, std::size_t row) const {
        return get_column_as<T>(col).at(row);
    }

    template <typename T>
    void set(const std::string& col, std::size_t row, const T& value) {
        get_column_as<T>(col).at(row) = value;
    }

    // --- Row slicing ---

    // slice(start, end): returns rows [start, end)
    std::shared_ptr<DataFrame> slice(std::size_t start, std::size_t end) const {
        if (start > end || end > num_rows()) {
            throw std::out_of_range("Invalid slice range");
        }
        auto result = std::make_shared<DataFrame>();
        for (const auto& name : col_order_) {
            auto it = columns_.find(name);
            Column sliced = it->second.slice(start, end);
            result->col_order_.push_back(name);
            result->columns_[name] = sliced;
        }
        result->type_ = type_;
        return result;
    }

    std::shared_ptr<DataFrame> head(std::size_t n = 5) const {
        std::size_t rows = num_rows();
        return slice(0, n < rows ? n : rows);
    }

    std::shared_ptr<DataFrame> tail(std::size_t n = 5) const {
        std::size_t rows = num_rows();
        std::size_t start = n < rows ? rows - n : 0;
        return slice(start, rows);
    }

    // --- Pretty print ---

    std::string to_string(std::size_t max_rows = 20) const {
        if (columns_.empty()) return "[Empty DataFrame]";

        std::size_t nrows = num_rows();
        std::size_t display_rows = max_rows < nrows ? max_rows : nrows;

        // Compute column widths
        std::vector<std::size_t> widths;
        for (const auto& name : col_order_) {
            std::size_t w = name.size();
            auto it = columns_.find(name);
            for (std::size_t r = 0; r < display_rows; ++r) {
                std::size_t len = it->second.to_string(r).size();
                if (len > w) w = len;
            }
            widths.push_back(w);
        }

        std::ostringstream ss;

        // Header (name + quantity key if present)
        std::vector<std::string> headers;
        for (const auto& name : col_order_) {
            auto it = columns_.find(name);
            if (it->second.quantity.empty()) {
                headers.push_back(name);
            } else {
                headers.push_back(name + "(" + it->second.quantity + ")");
            }
        }
        // Recompute widths with headers
        for (std::size_t c = 0; c < headers.size(); ++c) {
            if (headers[c].size() > widths[c]) widths[c] = headers[c].size();
        }

        for (std::size_t c = 0; c < col_order_.size(); ++c) {
            if (c > 0) ss << " | ";
            ss << std::setw(static_cast<int>(widths[c])) << std::right << headers[c];
        }
        ss << "\n";

        // Separator
        for (std::size_t c = 0; c < col_order_.size(); ++c) {
            if (c > 0) ss << "-+-";
            for (std::size_t i = 0; i < widths[c]; ++i) ss << '-';
        }
        ss << "\n";

        // Rows
        for (std::size_t r = 0; r < display_rows; ++r) {
            for (std::size_t c = 0; c < col_order_.size(); ++c) {
                if (c > 0) ss << " | ";
                auto it = columns_.find(col_order_[c]);
                ss << std::setw(static_cast<int>(widths[c])) << std::right
                   << it->second.to_string(r);
            }
            ss << "\n";
        }

        if (display_rows < nrows) {
            ss << "... (" << nrows << " rows total)\n";
        }

        ss << "[" << nrows << " rows x " << col_order_.size() << " columns";
        if (!index_dims_.empty()) {
            ss << ", indices:";
            for (std::size_t i = 0; i < index_dims_.size(); ++i) {
                if (i > 0) ss << " x";
                ss << " " << index_dims_[i].name << "(" << index_dims_[i].dim_size() << ")";
            }
        }
        ss << "]";
        return ss.str();
    }

    // --- Copy ---
    std::shared_ptr<DataFrame> copy() const {
        auto result = std::make_shared<DataFrame>();
        for (const auto& name : col_order_) {
            auto it = columns_.find(name);
            result->col_order_.push_back(name);
            result->columns_[name] = it->second.clone();
        }
        result->index_dims_ = index_dims_;
        result->type_ = type_;
        result->name_ = name_;
        result->path_ = path_;
        return result;
    }

    // --- Multi-index construction ---

    // Add an index dimension. Existing columns are Cartesian-expanded.
    // Must be called before any dependent (data) columns are added.
    template <typename T>
    void add_uniform_index(const std::string& name, const std::vector<T>& levels,
                   const std::string& quantity = unit_format::quantity::unitless) {
        if (levels.empty())
            throw std::invalid_argument("Index '" + name + "' levels cannot be empty");
        if (has_column(name))
            throw std::invalid_argument("Column '" + name + "' already exists");
        if (col_order_.size() > index_dims_.size())
            throw std::invalid_argument(
                "Cannot add index after dependent columns have been added");
        if (make_column<T>(levels).extract_unique().size() != levels.size())
            throw std::invalid_argument("Index '" + name + "' has duplicate levels");

        std::size_t new_n = levels.size();
        std::size_t old_rows = num_rows();

        if (old_rows == 0) {
            // First index: add levels as column data directly
            col_order_.push_back(name);
            Column col = make_column<T>(levels);
            col.quantity = quantity;
            columns_[name] = col;
        } else {
            // Expand existing columns: repeat each value new_n times
            for (auto& pair : columns_) {
                pair.second = pair.second.repeat_each(new_n);
            }
            // Add new column: tile levels old_rows times
            col_order_.push_back(name);
            Column col = make_column<T>(levels);
            col.quantity = quantity;
            columns_[name] = col.tile(old_rows);
        }

        index_dims_.push_back(IndexDim::create_uniform(name, levels, quantity));
    }

    // Add a varying index dimension. Unlike uniform indices (Cartesian),
    // the values in this dimension can differ per outer-index group.
    // `values` must have exactly `old_rows * group_size` elements.
    // The first index dimension must be uniform.
    template <typename T>
    void add_varying_index(const std::string& name,
                           const std::vector<T>& values,
                           std::size_t group_size,
                           const std::string& quantity = unit_format::quantity::unitless) {
        if (group_size == 0)
            throw std::invalid_argument("group_size cannot be zero");
        if (has_column(name))
            throw std::invalid_argument("Column '" + name + "' already exists");
        if (col_order_.size() > index_dims_.size())
            throw std::invalid_argument(
                "Cannot add index after dependent columns have been added");

        std::size_t old_rows = num_rows();

        if (old_rows == 0)
            throw std::invalid_argument(
                "First index dimension cannot be varying; use add_uniform_index() first");
        if (values.size() != old_rows * group_size)
            throw std::invalid_argument(
                "Expected " + std::to_string(old_rows * group_size) +
                " values (old_rows * group_size), got " +
                std::to_string(values.size()));

        // Expand existing columns: repeat each value group_size times
        for (auto& pair : columns_) {
            pair.second = pair.second.repeat_each(group_size);
        }
        col_order_.push_back(name);
        Column col = make_column<T>(values);
        col.quantity = quantity;
        columns_[name] = col;

        index_dims_.push_back(IndexDim::create_varying(name, group_size, quantity));
    }

    // Add a varying index by per-group values to avoid manual flattening.
    // groups.size() must equal current row count, and each group must have equal size.
    template <typename T>
    void add_varying_index_groups(const std::string& name,
                                  const std::vector<std::vector<T>>& groups,
                                  const std::string& quantity = unit_format::quantity::unitless) {
        std::size_t old_rows = num_rows();
        if (old_rows == 0)
            throw std::invalid_argument(
                "First index dimension cannot be varying; use add_uniform_index() first");
        if (groups.empty())
            throw std::invalid_argument("groups cannot be empty");
        if (groups.size() != old_rows)
            throw std::invalid_argument(
                "Expected " + std::to_string(old_rows) +
                " groups, got " + std::to_string(groups.size()));

        std::size_t group_size = groups[0].size();
        if (group_size == 0)
            throw std::invalid_argument("group_size cannot be zero");

        std::vector<T> flat;
        flat.reserve(old_rows * group_size);
        for (std::size_t i = 0; i < groups.size(); ++i) {
            if (groups[i].size() != group_size)
                throw std::invalid_argument(
                    "All groups must have the same size; group " + std::to_string(i) +
                    " has size " + std::to_string(groups[i].size()) +
                    ", expected " + std::to_string(group_size));
            flat.insert(flat.end(), groups[i].begin(), groups[i].end());
        }

        add_varying_index<T>(name, flat, group_size, quantity);
    }

    // Build a DataFrame from the Cartesian product of index dimensions
    static std::shared_ptr<DataFrame> from_product(
        const std::vector<std::pair<std::string, Column>>& indices) {
        auto df = std::make_shared<DataFrame>();
        for (const auto& idx : indices) {
            switch (idx.second.tag) {
                case DType::Int:
                    df->add_uniform_index<int>(idx.first, idx.second.as<int>(), idx.second.quantity);
                    break;
                case DType::Double:
                    df->add_uniform_index<double>(idx.first, idx.second.as<double>(), idx.second.quantity);
                    break;
                case DType::String:
                    df->add_uniform_index<std::string>(idx.first, idx.second.as<std::string>(), idx.second.quantity);
                    break;
                case DType::Complex:
                    df->add_uniform_index<std::complex<double>>(idx.first, idx.second.as<std::complex<double>>(), idx.second.quantity);
                    break;
            }
        }
        return df;
    }

    // --- Multi-index queries ---

    std::size_t num_indices() const { return index_dims_.size(); }

    std::vector<std::string> index_names() const {
        std::vector<std::string> names;
        for (const auto& dim : index_dims_)
            names.push_back(dim.name);
        return names;
    }

    std::vector<std::string> independent_names() const { return index_names(); }

    bool is_index(const std::string& name) const {
        for (const auto& dim : index_dims_)
            if (dim.name == name) return true;
        return false;
    }

    std::vector<std::string> dependent_names() const {
        std::vector<std::string> names;
        for (const auto& n : col_order_)
            if (!is_index(n)) names.push_back(n);
        return names;
    }

    const IndexDim& get_index_dim(std::size_t dim) const {
        if (dim >= index_dims_.size())
            throw std::out_of_range("Index dimension out of range");
        return index_dims_[dim];
    }

    const IndexDim& get_index_dim(const std::string& name) const {
        for (const auto& dim : index_dims_)
            if (dim.name == name) return dim;
        throw std::invalid_argument("'" + name + "' is not an index column");
    }

    // --- Index promotion / demotion ---

    // Promote columns to index dimensions; reorders rows into Cartesian order.
    // e.g. set_index({"a","b"}) with a=[1,1,1,2,2,2], b=[10,20,30,10,20,30]
    void set_index(const std::vector<std::string>& names) {
        if (names.empty()) return;
        if (!index_dims_.empty())
            throw std::invalid_argument("DataFrame already has indices; call reset_index() first");

        // Validate all names exist
        for (const auto& name : names) {
            if (!has_column(name))
                throw std::invalid_argument("Column '" + name + "' not found");
        }

        std::size_t nrows = num_rows();
        if (nrows == 0)
            throw std::invalid_argument("Cannot set_index on an empty DataFrame");

        bool uniform_ok = true;
        std::vector<IndexDim> new_dims;
        try {
            // First attempt: all-uniform Cartesian model (existing behavior).
            for (const auto& name : names) {
                const Column& col = get_col(name);
                Column levels = col.extract_unique();
                levels.quantity = col.quantity;
                new_dims.push_back(IndexDim::create_uniform(name, levels));
            }

            std::size_t expected_rows = 1;
            for (const auto& dim : new_dims)
                expected_rows *= dim.dim_size();
            if (expected_rows != nrows)
                throw std::invalid_argument(
                    "Row count " + std::to_string(nrows) +
                    " does not match Cartesian product size " + std::to_string(expected_rows));

            std::size_t n = new_dims.size();
            std::vector<std::size_t> s(n);
            s[n - 1] = 1;
            for (std::size_t i = n - 1; i > 0; --i)
                s[i - 1] = s[i] * new_dims[i].dim_size();

            std::vector<std::size_t> perm(nrows, nrows); // perm[target] = source
            std::vector<bool> target_used(nrows, false);

            for (std::size_t r = 0; r < nrows; ++r) {
                std::size_t flat = 0;
                for (std::size_t i = 0; i < n; ++i) {
                    const Column& col = get_col(names[i]);
                    const Column& levels = new_dims[i].levels;
                    std::size_t dim_size = levels.size();
                    std::size_t level_idx = dim_size; // sentinel
                    for (std::size_t j = 0; j < dim_size; ++j) {
                        if (col.value_equals(r, levels, j)) {
                            level_idx = j;
                            break;
                        }
                    }
                    if (level_idx == dim_size)
                        throw std::invalid_argument(
                            "Column '" + names[i] + "' at row " + std::to_string(r) +
                            " has a value not found in unique levels");
                    flat += level_idx * s[i];
                }
                if (target_used[flat])
                    throw std::invalid_argument(
                        "Duplicate index combination at row " + std::to_string(r) +
                        " — data is not a valid Cartesian product");
                target_used[flat] = true;
                perm[flat] = r;
            }

            for (std::size_t i = 0; i < nrows; ++i) {
                if (!target_used[i])
                    throw std::invalid_argument(
                        "Missing index combination — data is not a valid Cartesian product");
            }

            bool is_identity = true;
            for (std::size_t i = 0; i < nrows; ++i) {
                if (perm[i] != i) { is_identity = false; break; }
            }
            if (!is_identity) {
                for (auto& pair : columns_)
                    pair.second = pair.second.gather(perm);
            }
        } catch (const std::invalid_argument&) {
            uniform_ok = false;
        }

        if (!uniform_ok) {
            // Fallback: infer mixed uniform/varying dimensions from current grouped layout.
            new_dims.clear();
            std::size_t span = nrows;

            for (const auto& name : names) {
                const Column& col = get_col(name);
                if (span == 0)
                    throw std::invalid_argument("Invalid grouped layout for set_index");

                // Determine inner block size from first run in the first chunk.
                std::size_t run_len = 1;
                while (run_len < span && col.value_equals(run_len, col, 0))
                    ++run_len;

                if (span % run_len != 0)
                    throw std::invalid_argument(
                        "Column '" + name + "' cannot form a valid grouped index layout");

                std::size_t dim_count = span / run_len;
                std::size_t chunk_count = nrows / span;
                bool is_uniform_dim = true;

                std::vector<std::size_t> rep_rows;
                rep_rows.reserve(dim_count);
                for (std::size_t k = 0; k < dim_count; ++k)
                    rep_rows.push_back(k * run_len);

                for (std::size_t c = 0; c < chunk_count; ++c) {
                    std::size_t base = c * span;
                    for (std::size_t k = 0; k < dim_count; ++k) {
                        std::size_t seg_start = base + k * run_len;
                        for (std::size_t t = 1; t < run_len; ++t) {
                            if (!col.value_equals(seg_start + t, col, seg_start)) {
                                throw std::invalid_argument(
                                    "Column '" + name + "' is not constant inside grouped blocks");
                            }
                        }
                        if (!col.value_equals(seg_start, col, rep_rows[k]))
                            is_uniform_dim = false;
                    }
                }

                if (is_uniform_dim) {
                    Column levels = col.gather(rep_rows);
                    levels.quantity = col.quantity;
                    new_dims.push_back(IndexDim::create_uniform(name, levels));
                } else {
                    new_dims.push_back(IndexDim::create_varying(name, dim_count, col.quantity));
                }

                span = run_len;
            }

            if (span != 1)
                throw std::invalid_argument(
                    "set_index grouped-layout inference failed: duplicate rows per index combination");
        }

        // Set inferred/validated index dims.
        index_dims_ = new_dims;

        // Reorder col_order_: index columns first, then dependent columns
        std::vector<std::string> new_order;
        for (const auto& name : names)
            new_order.push_back(name);
        for (const auto& name : col_order_) {
            if (std::find(names.begin(), names.end(), name) == names.end())
                new_order.push_back(name);
        }
        col_order_ = new_order;
    }

    void set_index(const std::string& name) {
        set_index(std::vector<std::string>{name});
    }

    void set_index(std::initializer_list<std::string> names) {
        set_index(std::vector<std::string>(names));
    }

    // Demote all index dimensions back to regular columns.
    void reset_index() {
        index_dims_.clear();
    }

    // --- Multi-index addressing ---

    // Strides for row-major Cartesian layout
    std::vector<std::size_t> strides() const {
        std::size_t n = index_dims_.size();
        if (n == 0) return {};
        std::vector<std::size_t> s(n);
        s[n - 1] = 1;
        for (std::size_t i = n - 1; i > 0; --i)
            s[i - 1] = s[i] * index_dims_[i].dim_size();
        return s;
    }

    // Convert per-dimension indices → flat row index
    std::size_t flat_index(const std::vector<std::size_t>& indices) const {
        if (indices.size() != index_dims_.size())
            throw std::invalid_argument(
                "Expected " + std::to_string(index_dims_.size()) +
                " indices, got " + std::to_string(indices.size()));
        auto s = strides();
        std::size_t row = 0;
        for (std::size_t i = 0; i < indices.size(); ++i) {
            if (indices[i] >= index_dims_[i].dim_size())
                throw std::out_of_range(
                    "Index " + std::to_string(indices[i]) +
                    " out of range for dimension '" + index_dims_[i].name + "'");
            row += indices[i] * s[i];
        }
        return row;
    }

    // Convert flat row index → per-dimension indices
    std::vector<std::size_t> multi_index(std::size_t flat) const {
        std::size_t n = index_dims_.size();
        if (n == 0)
            throw std::invalid_argument("No index dimensions");
        if (flat >= num_rows())
            throw std::out_of_range(
                "Flat index " + std::to_string(flat) +
                " out of range (num_rows=" + std::to_string(num_rows()) + ")");
        auto s = strides();
        std::vector<std::size_t> result(n);
        for (std::size_t i = 0; i < n; ++i) {
            result[i] = flat / s[i];
            flat %= s[i];
        }
        return result;
    }

    // --- Multi-index selection ---

    // loc: fix innermost dimensions (right-aligned).
    //   loc({i})   — fix last dim at i
    //   loc({i,j}) — fix last two dims at i, j
    // Returns sub-DataFrame with remaining outer dimensions.
    std::shared_ptr<DataFrame> loc(const std::vector<std::size_t>& indices) const {
        if (indices.empty()) return copy();
        std::size_t n = index_dims_.size();
        std::size_t k = indices.size();
        if (k > n)
            throw std::invalid_argument(
                "Too many indices: got " + std::to_string(k) +
                ", have " + std::to_string(n) + " dimensions");

        // Right-align: indices map to dims [n-k, n-k+1, ..., n-1]
        std::size_t n_outer = n - k;
        auto s = strides();

        // Validate and compute offset within inner block
        std::size_t offset = 0;
        for (std::size_t i = 0; i < k; ++i) {
            std::size_t dim = n_outer + i;
            if (indices[i] >= index_dims_[dim].dim_size())
                throw std::out_of_range(
                    "Index " + std::to_string(indices[i]) +
                    " out of range for dimension '" + index_dims_[dim].name + "'");
            offset += indices[i] * s[dim];
        }

        // inner_block = product of sizes of fixed dims [n_outer..n-1]
        std::size_t inner_block = 1;
        for (std::size_t i = n_outer; i < n; ++i)
            inner_block *= index_dims_[i].dim_size();

        std::size_t outer_count = num_rows() / inner_block;

        // Build gathered row indices (evenly spaced)
        std::vector<std::size_t> row_indices;
        row_indices.reserve(outer_count);
        for (std::size_t j = 0; j < outer_count; ++j)
            row_indices.push_back(j * inner_block + offset);

        // Collect names of fixed (inner) index columns to exclude
        std::vector<std::string> fixed_names;
        for (std::size_t i = n_outer; i < n; ++i)
            fixed_names.push_back(index_dims_[i].name);

        // Build result DataFrame by gathering rows, excluding fixed index columns
        auto result = std::make_shared<DataFrame>();
        for (const auto& name : col_order_) {
            if (std::find(fixed_names.begin(), fixed_names.end(), name) != fixed_names.end())
                continue;
            auto it = columns_.find(name);
            Column gathered = it->second.gather(row_indices);
            result->col_order_.push_back(name);
            result->columns_[name] = gathered;
        }

        // Set remaining (outer) index levels
        for (std::size_t i = 0; i < n_outer; ++i)
            result->index_dims_.push_back(index_dims_[i]);

        result->type_ = type_;
        return result;
    }

    std::shared_ptr<DataFrame> loc(std::initializer_list<std::size_t> indices) const {
        return loc(std::vector<std::size_t>(indices));
    }

    // sub: extract a sub-DataFrame by column name.
    //   dependent column  → all index columns + that column
    //   independent column → all outer index columns up to and including it
    std::shared_ptr<DataFrame> sub(const std::string& name) const {
        if (!has_column(name))
            throw std::invalid_argument("Column '" + name + "' not found");

        if (!is_index(name)) {
            // Dependent column: all independents + this dependent
            auto result = std::make_shared<DataFrame>();
            // Copy index columns
            for (const auto& dim : index_dims_) {
                result->col_order_.push_back(dim.name);
                result->columns_[dim.name] = columns_.find(dim.name)->second.clone();
            }
            result->index_dims_ = index_dims_;
            // Copy dependent column
            result->col_order_.push_back(name);
            result->columns_[name] = columns_.find(name)->second.clone();
            result->type_ = type_;
            return result;
        } else {
            // Independent column: all outer independents up to and including it
            std::size_t target_dim = 0;
            for (std::size_t i = 0; i < index_dims_.size(); ++i) {
                if (index_dims_[i].name == name) { target_dim = i; break; }
            }
            // Fixed inner dims = [target_dim+1 .. n-1], free outer = [0..target_dim]
            std::size_t n_outer = target_dim + 1;
            // Gather unique rows for outer dims only (first row of each block)
            std::size_t inner_block = 1;
            for (std::size_t i = n_outer; i < index_dims_.size(); ++i)
                inner_block *= index_dims_[i].dim_size();
            std::size_t outer_count = num_rows() / inner_block;
            std::vector<std::size_t> row_indices;
            row_indices.reserve(outer_count);
            for (std::size_t j = 0; j < outer_count; ++j)
                row_indices.push_back(j * inner_block);

            auto result = std::make_shared<DataFrame>();
            // Gather only outer index columns
            for (std::size_t i = 0; i < n_outer; ++i) {
                const auto& dim = index_dims_[i];
                result->col_order_.push_back(dim.name);
                result->columns_[dim.name] = columns_.find(dim.name)->second.gather(row_indices);
                result->index_dims_.push_back(dim);
            }
            result->type_ = type_;
            return result;
        }
    }

    // --- Metadata ---
    const std::string& path() const { return path_; }
    void set_path(const std::string& p) { path_ = p; }

    const std::string& type() const { return type_; }
    void set_type(const std::string& t) { type_ = t; }

    const std::string& name() const { return name_; }
    void set_name(const std::string& n) { name_ = n; }

    // --- CSV export ---

    std::string to_csv(char delimiter = ',') const {
        if (columns_.empty()) return "";
        std::ostringstream ss;

        // Header row
        for (std::size_t c = 0; c < col_order_.size(); ++c) {
            if (c > 0) ss << delimiter;
            ss << escape_csv_field(col_order_[c]);
        }
        ss << "\n";

        // Data rows
        std::size_t nrows = num_rows();
        for (std::size_t r = 0; r < nrows; ++r) {
            for (std::size_t c = 0; c < col_order_.size(); ++c) {
                if (c > 0) ss << delimiter;
                auto it = columns_.find(col_order_[c]);
                ss << escape_csv_field(it->second.to_string(r));
            }
            ss << "\n";
        }
        return ss.str();
    }

    void save_csv(const std::string& filename, char delimiter = ',') const {
        std::ofstream ofs(filename);
        if (!ofs.is_open())
            throw std::runtime_error("Cannot open file '" + filename + "' for writing");
        ofs << to_csv(delimiter);
    }

    // --- Rename ---

    DataFrame& rename(const std::string& old_name, const std::string& new_name) {
        if (old_name == new_name) return *this;
        if (!has_column(old_name))
            throw std::invalid_argument("Column '" + old_name + "' not found");
        if (has_column(new_name))
            throw std::invalid_argument("Column '" + new_name + "' already exists");

        // Move data in map
        columns_[new_name] = columns_[old_name];
        columns_.erase(old_name);

        // Update col_order_
        for (auto& n : col_order_) {
            if (n == old_name) { n = new_name; break; }
        }

        // Update index_dims_ if it's an index column
        for (auto& dim : index_dims_) {
            if (dim.name == old_name) { dim.name = new_name; break; }
        }
        return *this;
    }

    DataFrame& rename(std::size_t index, const std::string& new_name) {
        if (index >= col_order_.size())
            throw std::out_of_range(
                "Column index " + std::to_string(index) +
                " out of range (size=" + std::to_string(col_order_.size()) + ")");
        return rename(col_order_[index], new_name);
    }

    DataFrame& rename_last(const std::string& new_name) {
        if (col_order_.empty())
            throw std::invalid_argument("DataFrame has no columns");
        return rename(col_order_.size() - 1, new_name);
    }

private:
    static std::string escape_csv_field(const std::string& field) {
        if (field.find_first_of(",\"\r\n") == std::string::npos) return field;
        std::string escaped = "\"";
        for (char c : field) {
            if (c == '"') escaped += "\"\"";
            else escaped += c;
        }
        escaped += '"';
        return escaped;
    }

    std::vector<std::string> col_order_;
    std::unordered_map<std::string, Column> columns_;
    std::vector<IndexDim> index_dims_;
    std::string path_;
    std::string type_;
    std::string name_;

    const Column& get_col(const std::string& name) const {
        auto it = columns_.find(name);
        if (it == columns_.end())
            throw std::invalid_argument("Column '" + name + "' not found");
        return it->second;
    }

    Column& get_col(const std::string& name) {
        auto it = columns_.find(name);
        if (it == columns_.end())
            throw std::invalid_argument("Column '" + name + "' not found");
        return it->second;
    }
};
 
} // namespace exprdf

#endif // EXPRDF_HPP
