#ifndef EXPRDF_HPP
#define EXPRDF_HPP

// exprdf.hpp -- header-only DataFrame library  (requires C++11)
//
// Column types : int, double, string, complex<double>
//                (type-erased via ColumnStorageBase / ColumnStorage<T>)
// Multi-index  : Uniform (Cartesian product) and Grouped (equal or ragged)
//                dimensions for N-dimensional parameter sweeps.
// Index API    : add_uniform_index, add_grouped_index, add_grouped_index_groups,
//                set_index (auto-infers Uniform / Grouped from column data).

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

namespace exprdf {

// ============================================================
// DType -- column type tag
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

// Floating-point approximate equality (absolute + relative tolerance)
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

// Type-aware equality: approximate for floating-point types, exact otherwise
template <typename T>
inline bool values_equal(const T& a, const T& b) { return a == b; }
template <>
inline bool values_equal<double>(const double& a, const double& b) { return approx_equal(a, b); }
template <>
inline bool values_equal<std::complex<double>>(const std::complex<double>& a, const std::complex<double>& b) { return approx_equal(a, b); }

// ============================================================
// ColumnStorageBase / ColumnStorage<T> -- type-erased column storage
//
// ColumnStorageBase  : abstract base; declares all per-element and bulk
//                      operations needed by Column.
// ColumnStorage<T>   : concrete template; holds std::vector<T> and
//                      implements every virtual method.
// Supported types are those with a DTypeTag<T> specialisation.
// ============================================================

// Type-to-string helpers (specialised for each supported type)
template <typename T>
inline std::string column_val_to_string(const T& v) {
    std::ostringstream ss; ss << v; return ss.str();
}
template <>
inline std::string column_val_to_string<std::string>(const std::string& v) { return v; }
template <>
inline std::string column_val_to_string<std::complex<double>>(const std::complex<double>& v) {
    std::ostringstream ss;
    ss << "(" << v.real();
    if (v.imag() >= 0) ss << "+";
    ss << v.imag() << "j)";
    return ss.str();
}

// Abstract base -- type-erased operations on a column's data
struct ColumnStorageBase {
    virtual ~ColumnStorageBase() = default;
    virtual std::size_t size() const = 0;
    virtual std::string to_string_at(std::size_t row) const = 0;
    virtual std::shared_ptr<ColumnStorageBase> do_clone() const = 0;
    virtual std::shared_ptr<ColumnStorageBase> do_slice(std::size_t start, std::size_t end) const = 0;
    virtual std::shared_ptr<ColumnStorageBase> do_repeat_each(std::size_t n) const = 0;
    virtual std::shared_ptr<ColumnStorageBase> do_gather(const std::vector<std::size_t>& indices) const = 0;
    virtual std::shared_ptr<ColumnStorageBase> do_repeat_variable(const std::vector<std::size_t>& counts) const = 0;
    virtual std::shared_ptr<ColumnStorageBase> do_tile(std::size_t n) const = 0;
    virtual std::shared_ptr<ColumnStorageBase> do_extract_unique() const = 0;
    virtual bool value_equals_at(std::size_t row_a, const ColumnStorageBase& other, std::size_t row_b) const = 0;
};

// Concrete storage for a supported type T.
// To add a new type: add a DTypeTag<T> specialisation, a DType enum entry,
// and a make_column<T> explicit instantiation -- this template does the rest.
template <typename T>
struct ColumnStorage : ColumnStorageBase {
    std::vector<T> data;

    ColumnStorage() = default;
    explicit ColumnStorage(const std::vector<T>& v) : data(v) {}

    std::size_t size() const override { return data.size(); }

    std::string to_string_at(std::size_t row) const override {
        return column_val_to_string<T>(data.at(row));
    }

    std::shared_ptr<ColumnStorageBase> do_clone() const override {
        return std::make_shared<ColumnStorage<T>>(data);
    }

    std::shared_ptr<ColumnStorageBase> do_slice(std::size_t start, std::size_t end) const override {
        auto s = std::make_shared<ColumnStorage<T>>();
        s->data.assign(data.begin() + start, data.begin() + end);
        return s;
    }

    std::shared_ptr<ColumnStorageBase> do_repeat_each(std::size_t n) const override {
        auto s = std::make_shared<ColumnStorage<T>>();
        s->data.reserve(data.size() * n);
        for (const auto& v : data)
            for (std::size_t i = 0; i < n; ++i) s->data.push_back(v);
        return s;
    }

    std::shared_ptr<ColumnStorageBase> do_gather(const std::vector<std::size_t>& indices) const override {
        auto s = std::make_shared<ColumnStorage<T>>();
        s->data.reserve(indices.size());
        for (auto i : indices) s->data.push_back(data.at(i));
        return s;
    }

    std::shared_ptr<ColumnStorageBase> do_repeat_variable(const std::vector<std::size_t>& counts) const override {
        auto s = std::make_shared<ColumnStorage<T>>();
        for (std::size_t i = 0; i < data.size(); ++i)
            for (std::size_t j = 0; j < counts[i]; ++j) s->data.push_back(data[i]);
        return s;
    }

    std::shared_ptr<ColumnStorageBase> do_tile(std::size_t n) const override {
        auto s = std::make_shared<ColumnStorage<T>>();
        s->data.reserve(data.size() * n);
        for (std::size_t i = 0; i < n; ++i)
            for (const auto& v : data) s->data.push_back(v);
        return s;
    }

    std::shared_ptr<ColumnStorageBase> do_extract_unique() const override {
        auto s = std::make_shared<ColumnStorage<T>>();
        for (const auto& v : data) {
            bool found = false;
            for (const auto& u : s->data)
                if (values_equal(u, v)) { found = true; break; }
            if (!found) s->data.push_back(v);
        }
        return s;
    }

    bool value_equals_at(std::size_t row_a, const ColumnStorageBase& other, std::size_t row_b) const override {
        // Caller guarantees same DType tag -> same T
        return values_equal(data[row_a], static_cast<const ColumnStorage<T>&>(other).data[row_b]);
    }
};

// ============================================================
// Column -- type-tagged column backed by ColumnStorage<T>
// ============================================================
class Column {
public:
    DType tag;
    std::string quantity;   // quantity key, e.g. "voltage" or "frequency"

    Column() : tag(DType::Int), storage_(std::make_shared<ColumnStorage<int>>()) {}

    // Named construction helpers -- prefer the generic make_column<T>() free function.
    static Column from_int(const std::vector<int>& v) {
        Column c; c.tag = DType::Int;
        c.storage_ = std::make_shared<ColumnStorage<int>>(v); return c;
    }
    static Column from_double(const std::vector<double>& v) {
        Column c; c.tag = DType::Double;
        c.storage_ = std::make_shared<ColumnStorage<double>>(v); return c;
    }
    static Column from_string(const std::vector<std::string>& v) {
        Column c; c.tag = DType::String;
        c.storage_ = std::make_shared<ColumnStorage<std::string>>(v); return c;
    }
    static Column from_complex(const std::vector<std::complex<double>>& v) {
        Column c; c.tag = DType::Complex;
        c.storage_ = std::make_shared<ColumnStorage<std::complex<double>>>(v); return c;
    }

    std::size_t size() const { return storage_->size(); }

    std::string dtype_name() const { return dtype_to_string(tag); }

    std::string to_string(std::size_t row) const { return storage_->to_string_at(row); }

    Column clone() const {
        Column c; c.tag = tag; c.quantity = quantity;
        c.storage_ = storage_->do_clone(); return c;
    }

    Column slice(std::size_t start, std::size_t end) const {
        Column c; c.tag = tag; c.quantity = quantity;
        c.storage_ = storage_->do_slice(start, end); return c;
    }

    // Repeat each element n times: [a,b] repeat_each(3) -> [a,a,a,b,b,b]
    Column repeat_each(std::size_t n) const {
        Column c; c.tag = tag; c.quantity = quantity;
        c.storage_ = storage_->do_repeat_each(n); return c;
    }

    // Gather specific rows by index
    Column gather(const std::vector<std::size_t>& row_indices) const {
        Column c; c.tag = tag; c.quantity = quantity;
        c.storage_ = storage_->do_gather(row_indices); return c;
    }

    // Repeat each element a variable number of times:
    // [a,b] repeat_variable({3,2}) -> [a,a,a,b,b]
    // counts.size() must equal size()
    Column repeat_variable(const std::vector<std::size_t>& counts) const {
        Column c; c.tag = tag; c.quantity = quantity;
        c.storage_ = storage_->do_repeat_variable(counts); return c;
    }

    // Tile entire sequence n times: [a,b] tile(3) -> [a,b,a,b,a,b]
    Column tile(std::size_t n) const {
        Column c; c.tag = tag; c.quantity = quantity;
        c.storage_ = storage_->do_tile(n); return c;
    }

    // Extract unique values preserving first-appearance order
    Column extract_unique() const {
        Column c; c.tag = tag; c.quantity = quantity;
        c.storage_ = storage_->do_extract_unique(); return c;
    }

    // Compare value at row_a in this column with value at row_b in another column
    bool value_equals(std::size_t row_a, const Column& other, std::size_t row_b) const {
        if (tag != other.tag) return false;
        return storage_->value_equals_at(row_a, *other.storage_, row_b);
    }

    // Typed accessors -- caller must ensure T matches tag (undefined behaviour otherwise).
    // Works for any T that has a ColumnStorage<T> specialisation.
    template <typename T>
    const std::vector<T>& as() const {
        return static_cast<const ColumnStorage<T>*>(storage_.get())->data;
    }
    template <typename T>
    std::vector<T>& as() {
        return static_cast<ColumnStorage<T>*>(storage_.get())->data;
    }

private:
    std::shared_ptr<ColumnStorageBase> storage_;
};

// make_column<T>: construct a Column from a typed vector.
// Unsupported types produce a compile error (no DTypeTag<T> specialisation).
template <typename T>
inline Column make_column(const std::vector<T>& data);

template <> inline Column make_column<int>(const std::vector<int>& v)                                   { return Column::from_int(v); }
template <> inline Column make_column<double>(const std::vector<double>& v)                              { return Column::from_double(v); }
template <> inline Column make_column<std::string>(const std::vector<std::string>& v)                    { return Column::from_string(v); }
template <> inline Column make_column<std::complex<double>>(const std::vector<std::complex<double>>& v)  { return Column::from_complex(v); }

// ============================================================
// IndexKind / IndexDim -- multi-index dimension descriptors
// ============================================================

// IndexKind::Uniform  -- all outer groups share the same ordered level set;
//                       supports Cartesian stride arithmetic.
// IndexKind::Grouped  -- inner values can differ per outer group.
//   Regular Grouped   : all group_lengths are equal (stride arithmetic OK).
//   Ragged Grouped    : group_lengths differ (no stride arithmetic).
enum class IndexKind { Uniform, Grouped };

// Index dimension descriptor -- one axis of the multi-index.
struct IndexDim {
    std::string name;
    IndexKind kind;
    Column levels;                          // Uniform: unique ordered level values
    std::vector<std::size_t> group_lengths; // Grouped: inner element count per outer group
    std::size_t num_outer = 1;              // Uniform: number of outer groups
                                            //          (1 for the outermost dim; product of
                                            //           parent level counts for inner dims)
                                            // Grouped: unused -- use group_lengths.size()

    // num_groups(): number of outer groups for this dimension.
    //   Uniform  -> num_outer
    //   Grouped  -> group_lengths.size()
    std::size_t num_groups() const {
        return kind == IndexKind::Uniform ? num_outer : group_lengths.size();
    }

    // level_count(): element count used in stride / flat-index calculations.
    //   Uniform         -> levels.size()        (number of unique levels)
    //   Regular Grouped -> group_lengths[0]     (common group size)
    //   Ragged Grouped  -> 0                    (stride math not applicable)
    std::size_t level_count() const {
        if (kind == IndexKind::Grouped && is_regular_grouped()) return group_lengths[0];
        return levels.size();
    }

    // max_group_length(): largest inner size across all grouped groups.
    std::size_t max_group_length() const {
        if (group_lengths.empty()) return 0;
        return *std::max_element(group_lengths.begin(), group_lengths.end());
    }

    // is_regular_grouped(): true when every group has the same inner size.
    // Regular grouped dimensions support stride arithmetic.
    bool is_regular_grouped() const {
        if (!is_grouped() || group_lengths.empty()) return false;
        for (std::size_t i = 1; i < group_lengths.size(); ++i)
            if (group_lengths[i] != group_lengths[0]) return false;
        return true;
    }

    bool is_uniform() const { return kind == IndexKind::Uniform; }
    bool is_grouped() const { return kind == IndexKind::Grouped; }

    template <typename T>
    static IndexDim create_uniform(const std::string& name,
                                   const std::vector<T>& lvls,
                                   const std::string& quantity = "",
                                   std::size_t num_outer = 1) {
        Column col = make_column<T>(lvls);
        col.quantity = quantity;
        return {name, IndexKind::Uniform, col, {}, num_outer};
    }

    static IndexDim create_uniform(const std::string& name, Column lvls,
                                   std::size_t num_outer = 1) {
        return {name, IndexKind::Uniform, lvls, {}, num_outer};
    }

    // Create a Grouped dimension; lengths[i] = inner element count for outer group i.
    static IndexDim create_grouped(const std::string& name,
                                   const std::vector<std::size_t>& lengths,
                                   const std::string& quantity = "") {
        Column col;
        col.quantity = quantity;
        return {name, IndexKind::Grouped, col, lengths, 0};
    }
};

// ============================================================
// DataFrame -- tabular data with an optional multi-index
//
// Columns are stored in insertion order (col_order_) and looked
// up by name via an unordered_map.  Index dimensions (index_dims_)
// describe the multi-dimensional layout of the rows.
// ============================================================
class DataFrame {
public:
    DataFrame() {}

    // --- Column management ---

    template <typename T>
    void add_column(const std::string& name, const std::vector<T>& data,
                    const std::string& quantity = "") {
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
                       const std::string& quantity = "") {
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
                        const std::string& quantity = "") {
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
    // Throws std::out_of_range when index >= num_columns().

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
    // Throws std::invalid_argument when the column is not found.

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

    // slice(start, end): returns a new DataFrame with rows [start, end)
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

        // Build row-index labels when multi-index is present (e.g. "0,2,1")
        bool has_idx = !index_dims_.empty();
        // Ragged dims can't use multi_index() labels; uniform and regular grouped can.
        bool has_ragged_dim = false;
        for (const auto& dim : index_dims_)
            if (dim.is_grouped() && !dim.is_regular_grouped()) { has_ragged_dim = true; break; }
        std::vector<std::string> row_labels;
        std::size_t idx_col_w = 0;
        if (has_idx) {
            for (std::size_t r = 0; r < display_rows; ++r) {
                std::string label;
                if (!has_ragged_dim) {
                    std::vector<std::size_t> mi = multi_index(r);
                    for (std::size_t d = 0; d < mi.size(); ++d) {
                        if (d > 0) label += ',';
                        label += std::to_string(mi[d]);
                    }
                } else {
                    // For ragged layouts, show row number only
                    label = std::to_string(r);
                }
                if (label.size() > idx_col_w) idx_col_w = label.size();
                row_labels.push_back(label);
            }
            // header placeholder width (blank)
            if (idx_col_w == 0) idx_col_w = 1;
        }

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

        // Print header row: optional blank index column first
        if (has_idx) {
            for (std::size_t i = 0; i < idx_col_w; ++i) ss << ' ';
            ss << " | ";
        }
        for (std::size_t c = 0; c < col_order_.size(); ++c) {
            if (c > 0) ss << " | ";
            ss << std::setw(static_cast<int>(widths[c])) << std::right << headers[c];
        }
        ss << "\n";

        // Separator
        if (has_idx) {
            for (std::size_t i = 0; i < idx_col_w; ++i) ss << '-';
            ss << "-+-";
        }
        for (std::size_t c = 0; c < col_order_.size(); ++c) {
            if (c > 0) ss << "-+-";
            for (std::size_t i = 0; i < widths[c]; ++i) ss << '-';
        }
        ss << "\n";

        // Rows
        for (std::size_t r = 0; r < display_rows; ++r) {
            if (has_idx) {
                ss << std::setw(static_cast<int>(idx_col_w)) << std::right << row_labels[r];
                ss << " | ";
            }
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
                if (index_dims_[i].is_grouped()) {
                    if (index_dims_[i].is_regular_grouped())
                        ss << " " << index_dims_[i].name << "(grouped:" << index_dims_[i].level_count() << ")";
                    else
                        ss << " " << index_dims_[i].name << "(ragged)";
                } else
                    ss << " " << index_dims_[i].name << "(" << index_dims_[i].level_count() << ")";
            }
        }
        ss << "]";
        return ss.str();
    }

    // --- Copy ---
    // Returns a deep copy of all columns and index dimensions.
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

    // add_uniform_index: append a Uniform dimension.
    //   All existing row data is Cartesian-expanded (each row repeated levels.size() times).
    //   Must be called before any dependent (data) columns are added.
    //   Duplicate levels are rejected.
    template <typename T>
    void add_uniform_index(const std::string& name, const std::vector<T>& levels,
                   const std::string& quantity = "") {
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
        std::size_t outer = (old_rows == 0) ? 1 : old_rows;

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

        index_dims_.push_back(IndexDim::create_uniform(name, levels, quantity, outer));
    }

    // add_grouped_index: append a Regular Grouped dimension (equal-sized groups, flat input).
    //   values.size() must equal num_rows() * group_size.
    //   Unlike Uniform, inner values can differ per outer group.
    //   Cannot be the first dimension (requires an existing outer dimension).
    template <typename T>
    void add_grouped_index(const std::string& name,
                           const std::vector<T>& values,
                           std::size_t group_size,
                           const std::string& quantity = "") {
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
                "First index dimension cannot be grouped; use add_uniform_index() first");
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

        index_dims_.push_back(IndexDim::create_grouped(name, std::vector<std::size_t>(old_rows, group_size), quantity));
    }

    // add_grouped_index_groups: append a Grouped dimension from per-group value lists.
    //   groups.size() must equal num_rows() (one list per current outer row).
    //   Inner lists may differ in size -> produces a Ragged Grouped dimension.
    //   Cannot be the first dimension (requires an existing outer dimension).
    template <typename T>
    void add_grouped_index_groups(const std::string& name,
                                  const std::vector<std::vector<T>>& groups,
                                  const std::string& quantity = "") {
        std::size_t old_rows = num_rows();
        if (old_rows == 0)
            throw std::invalid_argument(
                "First index dimension cannot be grouped; use add_uniform_index() first");
        if (groups.empty())
            throw std::invalid_argument("groups cannot be empty");
        if (groups.size() != old_rows)
            throw std::invalid_argument(
                "Expected " + std::to_string(old_rows) +
                " groups (one per current row), got " + std::to_string(groups.size()));
        if (has_column(name))
            throw std::invalid_argument("Column '" + name + "' already exists");
        if (col_order_.size() > index_dims_.size())
            throw std::invalid_argument(
                "Cannot add index after dependent columns have been added");

        // Collect per-group lengths
        std::vector<std::size_t> lengths;
        lengths.reserve(old_rows);
        for (const auto& g : groups) {
            if (g.empty())
                throw std::invalid_argument("Each grouped group must have at least one element");
            lengths.push_back(g.size());
        }

        // Expand every existing column: row i is repeated lengths[i] times
        for (auto& pair : columns_)
            pair.second = pair.second.repeat_variable(lengths);

        // Flatten groups and add as new column
        std::vector<T> flat;
        flat.reserve(num_rows()); // after expansion
        for (const auto& g : groups)
            flat.insert(flat.end(), g.begin(), g.end());

        col_order_.push_back(name);
        Column col = make_column<T>(flat);
        col.quantity = quantity;
        columns_[name] = col;

        index_dims_.push_back(IndexDim::create_grouped(name, lengths, quantity));
    }

    // add_grouped_index_groups overload: flat data + explicit per-group sizes.
    //   group_data.size() must equal sum(group_sizes).
    //   group_sizes.size() must equal num_rows() (one size per current outer row).
    template <typename T>
    void add_grouped_index_groups(const std::string& name,
                                  const std::vector<T>& group_data,
                                  const std::vector<std::size_t>& group_sizes,
                                  const std::string& quantity = "") {
        std::size_t total = 0;
        for (auto sz : group_sizes) total += sz;
        if (total != group_data.size())
            throw std::invalid_argument(
                "sum(group_sizes)=" + std::to_string(total) +
                " does not match group_data.size()=" + std::to_string(group_data.size()));
        std::vector<std::vector<T>> groups;
        groups.reserve(group_sizes.size());
        std::size_t offset = 0;
        for (auto sz : group_sizes) {
            groups.emplace_back(group_data.begin() + offset, group_data.begin() + offset + sz);
            offset += sz;
        }
        add_grouped_index_groups<T>(name, groups, quantity);
    }

    // from_product: construct a DataFrame from an ordered list of (name, levels) pairs.
    //   Applies add_uniform_index for each entry in sequence.
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

    std::size_t num_indices() const { return index_dims_.size(); } // number of index dimensions

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

    // set_index: promote existing columns to index dimensions.
    //   Automatically infers the IndexKind for each column:
    //     Uniform         -- same values, same lengths, same order across all outer groups
    //     Regular Grouped -- same run lengths across outer groups, but values may differ
    //     Ragged          -- rejected; use add_grouped_index_groups() instead
    //   Rows are reordered so that the resulting layout is consistent with the inferred dims.
    //   Fails if the DataFrame already has index dimensions (call reset_index() first).
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
            std::size_t outer = 1;
            for (const auto& name : names) {
                const Column& col = get_col(name);
                Column levels = col.extract_unique();
                levels.quantity = col.quantity;
                new_dims.push_back(IndexDim::create_uniform(name, levels, outer));
                outer *= levels.size();
            }

            std::size_t expected_rows = 1;
            for (const auto& dim : new_dims)
                expected_rows *= dim.level_count();
            if (expected_rows != nrows)
                throw std::invalid_argument(
                    "Row count " + std::to_string(nrows) +
                    " does not match Cartesian product size " + std::to_string(expected_rows));

            std::size_t n = new_dims.size();
            std::vector<std::size_t> s(n);
            s[n - 1] = 1;
            for (std::size_t i = n - 1; i > 0; --i)
                s[i - 1] = s[i] * new_dims[i].level_count();

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
                        " -- data is not a valid Cartesian product");
                target_used[flat] = true;
                perm[flat] = r;
            }

            for (std::size_t i = 0; i < nrows; ++i) {
                if (!target_used[i])
                    throw std::invalid_argument(
                        "Missing index combination -- data is not a valid Cartesian product");
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
            // Fallback: infer mixed Uniform/Grouped dimensions.
            // We track "current groups" as (start_row, length) spans.
            // Each dim splits the current groups into sub-groups.
            new_dims.clear();

            using GSpan = std::pair<std::size_t, std::size_t>; // (start, len)
            std::vector<GSpan> cur_groups;
            cur_groups.push_back(std::make_pair(std::size_t(0), nrows));

            for (const auto& name : names) {
                const Column& col = get_col(name);

                // For each current group, find its consecutive-equal runs
                std::vector<std::vector<GSpan>> per_group_runs;
                per_group_runs.reserve(cur_groups.size());
                for (const auto& g : cur_groups) {
                    std::size_t gs = g.first, glen = g.second;
                    std::vector<GSpan> runs;
                    std::size_t r = gs;
                    while (r < gs + glen) {
                        std::size_t run_end = r + 1;
                        while (run_end < gs + glen && col.value_equals(run_end, col, r))
                            ++run_end;
                        runs.push_back(std::make_pair(r, run_end - r));
                        r = run_end;
                    }
                    per_group_runs.push_back(std::move(runs));
                }

                // Determine kind: check if all groups have the same sub-run count and lengths
                std::size_t ref_run_count = per_group_runs[0].size();
                bool same_count = true;
                bool same_lens  = true;
                bool same_vals  = true;

                for (std::size_t gi = 0; gi < per_group_runs.size(); ++gi) {
                    if (per_group_runs[gi].size() != ref_run_count) {
                        same_count = false; same_lens = false; same_vals = false;
                        break;
                    }
                    for (std::size_t ri = 0; ri < ref_run_count; ++ri) {
                        if (per_group_runs[gi][ri].second != per_group_runs[0][ri].second)
                            same_lens = false;
                        if (!col.value_equals(per_group_runs[gi][ri].first,
                                              col, per_group_runs[0][ri].first))
                            same_vals = false;
                    }
                }

                if (same_count && same_lens && same_vals) {
                    // Uniform: all groups have same run count, same lengths, same values
                    std::vector<std::size_t> rep_rows;
                    rep_rows.reserve(ref_run_count);
                    for (const auto& run : per_group_runs[0])
                        rep_rows.push_back(run.first);
                    Column levels = col.gather(rep_rows);
                    levels.quantity = col.quantity;
                    new_dims.push_back(IndexDim::create_uniform(name, levels, cur_groups.size()));
                } else if (same_count && same_lens) {
                    // Check all runs within a group have the same length
                    // (required for is_regular_grouped: all group_lengths equal).
                    std::size_t group_size = per_group_runs[0][0].second;
                    bool same_inner_lens = true;
                    for (std::size_t ri = 1; ri < ref_run_count; ++ri) {
                        if (per_group_runs[0][ri].second != group_size) {
                            same_inner_lens = false;
                            break;
                        }
                    }
                    if (!same_inner_lens) {
                        throw std::invalid_argument(
                            "set_index: column '" + name +
                            "' has non-uniform sub-group lengths within each group. "
                            "Use add_grouped_index_groups() to construct such dimensions manually.");
                    }
                    // Regular grouped: every sub-group has the same length across all outer groups.
                    std::size_t total_groups = cur_groups.size() * ref_run_count;
                    new_dims.push_back(IndexDim::create_grouped(
                        name,
                        std::vector<std::size_t>(total_groups, group_size),
                        col.quantity));
                } else {
                    // Ragged (different run counts or lengths): not supported by set_index.
                    // Use add_grouped_index_groups() to construct ragged dimensions manually.
                    throw std::invalid_argument(
                        "set_index: column '" + name +
                        "' has ragged structure (unequal group sizes or counts). "
                        "Use add_grouped_index_groups() to construct ragged dimensions manually.");
                }

                // Next level: sub-runs become new current groups
                cur_groups.clear();
                for (const auto& runs : per_group_runs)
                    for (const auto& run : runs)
                        cur_groups.push_back(run);
            }

            // Validate: every final group must be exactly 1 row
            for (const auto& g : cur_groups) {
                if (g.second != 1)
                    throw std::invalid_argument(
                        "set_index layout inference failed: final groups have more than 1 row");
            }
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

    // reset_index: demote all index dimensions back to regular data columns.
    //   Column data is preserved; only the index_dims_ metadata is cleared.
    void reset_index() {
        index_dims_.clear();
    }

    // --- Multi-index addressing ---

    // strides(): row-major strides for Uniform (and Regular Grouped) dimensions.
    // Result is undefined if any dimension is Ragged Grouped.
    std::vector<std::size_t> strides() const {
        std::size_t n = index_dims_.size();
        if (n == 0) return {};
        std::vector<std::size_t> s(n);
        s[n - 1] = 1;
        for (std::size_t i = n - 1; i > 0; --i)
            s[i - 1] = s[i] * index_dims_[i].level_count();
        return s;
    }

    // split_runs: split rows [start, start+len) into consecutive-equal-value spans.
    // Returns (start_row, length) pairs in encounter order.
    std::vector<std::pair<std::size_t,std::size_t>>
    split_runs(std::size_t start, std::size_t len, const std::string& col_name) const {
        std::vector<std::pair<std::size_t,std::size_t>> runs;
        if (len == 0) return runs;
        const Column& col = get_col(col_name);
        std::size_t r = start;
        while (r < start + len) {
            std::size_t end = r + 1;
            while (end < start + len && col.value_equals(end, col, r)) ++end;
            runs.push_back({r, end - r});
            r = end;
        }
        return runs;
    }

    // flat_index: convert per-dimension ordinals -> flat row index.
    //   indices.size() must equal num_indices().
    //   For Uniform dim d   : indices[d] selects among level_count() unique levels.
    //   For Grouped dim d   : indices[d] selects the i-th consecutive run within the
    //                          current outer group.
    //   Uses stride arithmetic when all dims are Uniform/Regular Grouped;
    //   falls back to a run-scan for ragged layouts.
    std::size_t flat_index(const std::vector<std::size_t>& indices) const {
        if (indices.size() != index_dims_.size())
            throw std::invalid_argument(
                "Expected " + std::to_string(index_dims_.size()) +
                " indices, got " + std::to_string(indices.size()));

        std::size_t n = index_dims_.size();

        // Fast path: all Uniform or regular Grouped AND num_rows == product of level_counts.
        // The second check catches set_index cases where regular_grouped groups don't tile
        // evenly across the outer uniform dimension (e.g. ragged set_index with unit groups).
        bool can_use_strides = true;
        for (const auto& d : index_dims_)
            if (!d.is_uniform() && !d.is_regular_grouped()) { can_use_strides = false; break; }
        if (can_use_strides) {
            std::size_t expected = 1;
            for (const auto& d : index_dims_) expected *= d.level_count();
            if (expected != num_rows()) can_use_strides = false;
        }
        if (can_use_strides) {
            auto s = strides();
            std::size_t row = 0;
            for (std::size_t i = 0; i < n; ++i) {
                if (indices[i] >= index_dims_[i].level_count())
                    throw std::out_of_range(
                        "Index " + std::to_string(indices[i]) +
                        " out of range for dimension '" + index_dims_[i].name + "'");
                row += indices[i] * s[i];
            }
            return row;
        }

        // General path: track the current (start, length) block as we descend each dim.
        std::size_t cur_start = 0, cur_len = num_rows();
        for (std::size_t d = 0; d < n; ++d) {
            std::size_t pos = indices[d];
            auto runs = split_runs(cur_start, cur_len, index_dims_[d].name);
            if (pos >= runs.size())
                throw std::out_of_range(
                    "Index " + std::to_string(pos) +
                    " out of range for dimension '" + index_dims_[d].name +
                    "' (has " + std::to_string(runs.size()) + " values in this group)");
            cur_start = runs[pos].first;
            cur_len   = runs[pos].second;
        }
        if (cur_len != 1)
            throw std::invalid_argument(
                "flat_index: specified indices do not uniquely identify a single row");
        return cur_start;
    }

    // multi_index: convert a flat row index -> per-dimension ordinals (inverse of flat_index).
    //   For each dimension the returned value is the ordinal of the consecutive-equal
    //   run that contains 'flat' within its outer group.
    std::vector<std::size_t> multi_index(std::size_t flat) const {
        std::size_t n = index_dims_.size();
        if (n == 0)
            throw std::invalid_argument("No index dimensions");
        if (flat >= num_rows())
            throw std::out_of_range(
                "Flat index " + std::to_string(flat) +
                " out of range (num_rows=" + std::to_string(num_rows()) + ")");

        // Fast path: all Uniform or regular Grouped AND num_rows == product of level_counts.
        bool can_use_strides = true;
        for (const auto& d : index_dims_)
            if (!d.is_uniform() && !d.is_regular_grouped()) { can_use_strides = false; break; }
        if (can_use_strides) {
            std::size_t expected = 1;
            for (const auto& d : index_dims_) expected *= d.level_count();
            if (expected != num_rows()) can_use_strides = false;
        }
        if (can_use_strides) {
            auto s = strides();
            std::vector<std::size_t> result(n);
            std::size_t rem = flat;
            for (std::size_t i = 0; i < n; ++i) {
                result[i] = rem / s[i];
                rem %= s[i];
            }
            return result;
        }

        // General path: for each dim, find which run (position) within the current
        // outer group contains 'flat', then narrow the group to that run.
        std::vector<std::size_t> result(n);
        std::size_t cur_start = 0, cur_len = num_rows();
        for (std::size_t d = 0; d < n; ++d) {
            auto runs = split_runs(cur_start, cur_len, index_dims_[d].name);
            // Find which run contains 'flat'
            std::size_t pos = runs.size(); // sentinel
            for (std::size_t r = 0; r < runs.size(); ++r) {
                if (flat >= runs[r].first && flat < runs[r].first + runs[r].second) {
                    pos = r;
                    break;
                }
            }
            if (pos == runs.size())
                throw std::invalid_argument("multi_index: internal error, row not found in runs");
            result[d] = pos;
            cur_start = runs[pos].first;
            cur_len   = runs[pos].second;
        }
        return result;
    }

    // --- Multi-index selection ---

    // loc: select rows by fixing the innermost N index dimensions (right-aligned).
    //   loc({i})      -- fix the last dim at position i
    //   loc({i, j})   -- fix the last two dims at i, j
    //   -1 (wildcard) -- keep all positions for that dimension (column remains in result)
    //   Grouped dims  : outer groups that do not contain position i are silently dropped.
    //   Returns a new DataFrame with the outer (unfixed, non-wildcard) dims as its index.
    std::shared_ptr<DataFrame> loc(const std::vector<int64_t>& indices) const {
        if (indices.empty()) return copy();
        std::size_t n = index_dims_.size();
        std::size_t k = indices.size();
        if (k > n)
            throw std::invalid_argument(
                "Too many indices: got " + std::to_string(k) +
                ", have " + std::to_string(n) + " dimensions");

        // Use stride path when all dims (including outer) are stride-compatible:
        // no ragged grouped, AND num_rows == product of all level_counts.
        std::size_t n_outer = n - k;
        bool any_ragged = false;
        for (const auto& d : index_dims_)
            if (d.is_grouped() && !d.is_regular_grouped()) { any_ragged = true; break; }
        if (!any_ragged) {
            std::size_t expected = 1;
            for (const auto& d : index_dims_) expected *= d.level_count();
            if (expected != num_rows()) any_ragged = true;
        }

        if (!any_ragged) {
            // --- Rectangular stride path (Uniform and/or regular Grouped) ---
            auto s = strides();

            // Validate non-wildcard indices
            for (std::size_t i = 0; i < k; ++i) {
                std::size_t dim = n_outer + i;
                if (indices[i] != -1) {
                    if (indices[i] < 0 || static_cast<std::size_t>(indices[i]) >= index_dims_[dim].level_count())
                        throw std::out_of_range(
                            "\"" + index_dims_[dim].name + "\" (dimension " + std::to_string(dim + 1) +
                            "): index " + std::to_string(indices[i]) +
                            " is outside range 0.." + std::to_string(static_cast<int64_t>(index_dims_[dim].level_count()) - 1));
                }
            }

            std::size_t inner_block = 1;
            for (std::size_t i = n_outer; i < n; ++i)
                inner_block *= index_dims_[i].level_count();

            std::vector<std::size_t> offsets = {0};
            for (std::size_t i = 0; i < k; ++i) {
                std::size_t dim = n_outer + i;
                if (indices[i] == -1) {
                    std::vector<std::size_t> expanded;
                    expanded.reserve(offsets.size() * index_dims_[dim].level_count());
                    for (auto off : offsets)
                        for (std::size_t v = 0; v < index_dims_[dim].level_count(); ++v)
                            expanded.push_back(off + v * s[dim]);
                    offsets = std::move(expanded);
                } else {
                    for (auto& off : offsets)
                        off += static_cast<std::size_t>(indices[i]) * s[dim];
                }
            }

            std::size_t outer_count = num_rows() / inner_block;
            std::vector<std::size_t> row_indices;
            row_indices.reserve(outer_count * offsets.size());
            for (std::size_t j = 0; j < outer_count; ++j)
                for (auto off : offsets)
                    row_indices.push_back(j * inner_block + off);

            std::vector<std::string> fixed_names;
            for (std::size_t i = 0; i < k; ++i)
                if (indices[i] != -1)
                    fixed_names.push_back(index_dims_[n_outer + i].name);

            auto result = std::make_shared<DataFrame>();
            for (const auto& name : col_order_) {
                if (std::find(fixed_names.begin(), fixed_names.end(), name) != fixed_names.end())
                    continue;
                auto it = columns_.find(name);
                result->col_order_.push_back(name);
                result->columns_[name] = it->second.gather(row_indices);
            }
            for (std::size_t i = 0; i < n_outer; ++i)
                result->index_dims_.push_back(index_dims_[i]);
            for (std::size_t i = 0; i < k; ++i)
                if (indices[i] == -1)
                    result->index_dims_.push_back(index_dims_[n_outer + i]);
            result->type_ = type_;
            return result;
        }

        // --- Grouped path ---
        // Use a column-value scan approach: group rows by consecutive equal values
        // in each index column, then pick the requested position within each group.
        //
        // We process dims from left to right.
        // cur_groups[i] = list of row indices belonging to the i-th current group.
        // Start: one group containing all rows.

        std::vector<std::vector<std::size_t>> cur_groups;
        {
            std::vector<std::size_t> all;
            all.reserve(num_rows());
            for (std::size_t r = 0; r < num_rows(); ++r) all.push_back(r);
            cur_groups.push_back(std::move(all));
        }

        // Helper: split a sorted list of row indices into sub-groups based on consecutive
        // equal values in the given column.
        auto split_by_col = [&](const std::vector<std::size_t>& rows,
                                 const std::string& col_name) -> std::vector<std::vector<std::size_t>> {
            std::vector<std::vector<std::size_t>> result;
            if (rows.empty()) return result;
            const Column& col = get_col(col_name);
            std::size_t start = 0;
            for (std::size_t i = 1; i <= rows.size(); ++i) {
                if (i == rows.size() || !col.value_equals(rows[i], col, rows[start])) {
                    std::vector<std::size_t> sub(rows.begin() + start, rows.begin() + i);
                    result.push_back(std::move(sub));
                    start = i;
                }
            }
            return result;
        };

        // Expand outer dims: split into sub-groups but keep all of them
        for (std::size_t d = 0; d < n_outer; ++d) {
            std::vector<std::vector<std::size_t>> next;
            for (auto& grp : cur_groups) {
                auto subs = split_by_col(grp, index_dims_[d].name);
                for (auto& s : subs) next.push_back(std::move(s));
            }
            cur_groups = std::move(next);
        }

        // Apply inner dims
        std::vector<std::string> fixed_names;
        std::vector<IndexDim> result_inner_dims;

        for (std::size_t i = 0; i < k; ++i) {
            std::size_t dim = n_outer + i;
            int64_t idx = indices[i];
            std::vector<std::vector<std::size_t>> next;

            for (auto& grp : cur_groups) {
                auto subs = split_by_col(grp, index_dims_[dim].name);
                if (idx == -1) {
                    // Wildcard: keep all sub-groups as new groups
                    for (auto& s : subs) next.push_back(std::move(s));
                } else {
                    std::size_t pos = static_cast<std::size_t>(idx);
                    if (pos < subs.size())
                        next.push_back(std::move(subs[pos]));
                    // else: this group has no position pos -> silently skip
                }
            }
            cur_groups = std::move(next);

            if (idx == -1)
                result_inner_dims.push_back(index_dims_[dim]);
            else
                fixed_names.push_back(index_dims_[dim].name);
        }

        // Flatten cur_groups into row_indices
        std::vector<std::size_t> row_indices;
        for (auto& grp : cur_groups)
            for (auto r : grp) row_indices.push_back(r);

        // Build result DataFrame
        auto result = std::make_shared<DataFrame>();
        for (const auto& name : col_order_) {
            if (std::find(fixed_names.begin(), fixed_names.end(), name) != fixed_names.end())
                continue;
            auto it = columns_.find(name);
            result->col_order_.push_back(name);
            result->columns_[name] = it->second.gather(row_indices);
        }
        for (std::size_t i = 0; i < n_outer; ++i)
            result->index_dims_.push_back(index_dims_[i]);
        for (auto& d : result_inner_dims)
            result->index_dims_.push_back(d);

        result->type_ = type_;
        return result;
    }

    std::shared_ptr<DataFrame> loc(std::initializer_list<int64_t> indices) const {
        return loc(std::vector<int64_t>(indices));
    }

    // sub: extract a sub-DataFrame by column name.
    //   Dependent column   -> all index columns + the named data column
    //   Independent column -> all index columns up to and including the named dim
    //                        (collapses inner dimensions, one row per outer group)
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
                inner_block *= index_dims_[i].level_count();
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

    // --- Metadata (inherited by loc / sub results) ---
    const std::string& path() const { return path_; }
    void set_path(const std::string& p) { path_ = p; }

    const std::string& type() const { return type_; }
    void set_type(const std::string& t) { type_ = t; }

    const std::string& name() const { return name_; }
    void set_name(const std::string& n) { name_ = n; }

    // --- CSV export ---
    // Fields containing the delimiter, double-quotes, or newlines are quoted.

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

    // --- Arithmetic operators (operate on the last column) ---
    // df1 + df2 : element-wise add of the last columns; result is a copy of *this.
    // Both DataFrames must have equal num_rows() and the same last-column dtype.
    // Scalar overloads accept double; int columns truncate the result back to int.
    std::shared_ptr<DataFrame> operator+(const DataFrame& o) const {
        return apply_binary_op_last(o, Arith_Add());
    }
    std::shared_ptr<DataFrame> operator-(const DataFrame& o) const {
        return apply_binary_op_last(o, Arith_Sub());
    }
    std::shared_ptr<DataFrame> operator*(const DataFrame& o) const {
        return apply_binary_op_last(o, Arith_Mul());
    }
    std::shared_ptr<DataFrame> operator/(const DataFrame& o) const {
        return apply_binary_op_last(o, Arith_Div());
    }

    std::shared_ptr<DataFrame> operator+(double s) const {
        return apply_scalar_op_last(s, Arith_Add());
    }
    std::shared_ptr<DataFrame> operator-(double s) const {
        return apply_scalar_op_last(s, Arith_Sub());
    }
    std::shared_ptr<DataFrame> operator*(double s) const {
        return apply_scalar_op_last(s, Arith_Mul());
    }
    std::shared_ptr<DataFrame> operator/(double s) const {
        return apply_scalar_op_last(s, Arith_Div());
    }

    // Scalar on left (commutative: +, *; non-commutative: -, /)
    friend std::shared_ptr<DataFrame> operator+(double s, const DataFrame& df) { return df + s; }
    friend std::shared_ptr<DataFrame> operator*(double s, const DataFrame& df) { return df * s; }
    friend std::shared_ptr<DataFrame> operator-(double s, const DataFrame& df) {
        return df.apply_scalar_op_last(s, Arith_SubR());
    }
    friend std::shared_ptr<DataFrame> operator/(double s, const DataFrame& df) {
        return df.apply_scalar_op_last(s, Arith_DivR());
    }

private:
    // --- C++11 arithmetic function objects (replaces C++14 generic lambdas) ---
    struct Arith_Add { template<typename T> T operator()(T a, T b) const { return a + b; } };
    struct Arith_Sub { template<typename T> T operator()(T a, T b) const { return a - b; } };
    struct Arith_Mul { template<typename T> T operator()(T a, T b) const { return a * b; } };
    struct Arith_Div { template<typename T> T operator()(T a, T b) const { return a / b; } };
    struct Arith_SubR { template<typename T> T operator()(T a, T b) const { return b - a; } }; // s - df
    struct Arith_DivR { template<typename T> T operator()(T a, T b) const { return b / a; } }; // s / df

    // --- Arithmetic helpers ---

    // Numeric type promotion rank: Int(0) < Double(1) < Complex(2); String is rejected.
    static int numeric_rank(DType t) {
        switch (t) {
            case DType::Int:     return 0;
            case DType::Double:  return 1;
            case DType::Complex: return 2;
            default:             return -1; // String
        }
    }

    // Widen a numeric column to std::vector<double>; rejects String/Complex columns.
    static std::vector<double> to_double_vec(const Column& col) {
        if (col.tag == DType::Int) {
            const auto& src = col.as<int>();
            return std::vector<double>(src.begin(), src.end());
        }
        if (col.tag == DType::Double) return col.as<double>();
        throw std::invalid_argument(
            "Cannot widen '" + std::string(dtype_to_string(col.tag)) + "' to double");
    }

    // Widen a numeric column to std::vector<complex<double>>; rejects String.
    static std::vector<std::complex<double>> to_complex_vec(const Column& col) {
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

    // Replace the last column of a DataFrame copy with a new column of possibly different type.
    void replace_last_column(const std::string& name, Column new_col) {
        new_col.quantity = get_col(name).quantity;
        columns_[name] = std::move(new_col);
    }

    // Apply element-wise binary op between last column of *this and last column of o.
    // Supports numeric type promotion: int op double -> double, * op complex -> complex.
    // String columns are always rejected.
    template<typename Op>
    std::shared_ptr<DataFrame> apply_binary_op_last(const DataFrame& o, Op op) const {
        if (col_order_.empty())
            throw std::invalid_argument("DataFrame has no columns");
        if (o.col_order_.empty())
            throw std::invalid_argument("Other DataFrame has no columns");
        if (num_rows() != o.num_rows())
            throw std::invalid_argument(
                "Row count mismatch: " + std::to_string(num_rows()) +
                " vs " + std::to_string(o.num_rows()));
        const std::string& ln = col_order_.back();
        const Column& ca = get_col(ln);
        const Column& cb = o.get_col(o.col_order_.back());

        int ra = numeric_rank(ca.tag), rb = numeric_rank(cb.tag);
        if (ra < 0 || rb < 0)
            throw std::invalid_argument("Arithmetic on string columns is not supported");

        // Determine the promoted result type
        DType rt = (ra >= rb) ? ca.tag : cb.tag;

        auto result = copy();

        if (rt == DType::Int) {
            // Both must be int (rank 0 == 0)
            auto& va = result->get_col(ln).as<int>();
            const auto& vb = cb.as<int>();
            for (std::size_t i = 0; i < va.size(); ++i) va[i] = op(va[i], vb[i]);
        } else if (rt == DType::Double) {
            std::vector<double> va = to_double_vec(ca);
            std::vector<double> vb = to_double_vec(cb);
            std::vector<double> vc(va.size());
            for (std::size_t i = 0; i < va.size(); ++i) vc[i] = op(va[i], vb[i]);
            Column nc = make_column<double>(vc);
            nc.quantity = ca.quantity;
            result->columns_[ln] = std::move(nc);
        } else { // Complex
            using C = std::complex<double>;
            std::vector<C> va = to_complex_vec(ca);
            std::vector<C> vb = to_complex_vec(cb);
            std::vector<C> vc(va.size());
            for (std::size_t i = 0; i < va.size(); ++i) vc[i] = op(va[i], vb[i]);
            Column nc = make_column<C>(vc);
            nc.quantity = ca.quantity;
            result->columns_[ln] = std::move(nc);
        }
        return result;
    }

    // Apply element-wise scalar op on last column.
    // Type promotion rules (scalar is always double):
    //   int    -> result is double (scalar promotes int column to double)
    //   double -> result is double
    //   complex-> result is complex (scalar is widened to complex(s, 0))
    //   string -> throws
    template<typename Op>
    std::shared_ptr<DataFrame> apply_scalar_op_last(double s, Op op) const {
        if (col_order_.empty())
            throw std::invalid_argument("DataFrame has no columns");
        const std::string& ln = col_order_.back();
        const Column& cc = get_col(ln);
        auto result = copy();
        switch (cc.tag) {
            case DType::Int: {
                // Promote int -> double
                const auto& src = cc.as<int>();
                std::vector<double> vc(src.size());
                for (std::size_t i = 0; i < src.size(); ++i)
                    vc[i] = op(static_cast<double>(src[i]), s);
                Column nc = make_column<double>(vc);
                nc.quantity = cc.quantity;
                result->columns_[ln] = std::move(nc);
                break;
            }
            case DType::Double: {
                auto& ra = result->get_col(ln).as<double>();
                for (auto& v : ra) v = op(v, s);
                break;
            }
            case DType::Complex: {
                auto& ra = result->get_col(ln).as<std::complex<double>>();
                for (auto& v : ra) v = op(v, std::complex<double>(s, 0.0));
                break;
            }
            case DType::String:
                throw std::invalid_argument(
                    "Arithmetic on string columns is not supported");
        }
        return result;
    }

    // Wrap a CSV field in double-quotes if it contains special characters.
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

    std::vector<std::string> col_order_;          // column names in insertion order
    std::unordered_map<std::string, Column> columns_; // column data keyed by name
    std::vector<IndexDim> index_dims_;             // multi-index dimensions (outermost first)
    std::string path_;                             // source file path (metadata only)
    std::string type_;                             // data-type tag   (metadata only)
    std::string name_;                             // dataset name    (metadata only)

    // Internal column lookup; throws std::invalid_argument on miss.
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
