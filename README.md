# exprdf

A header-only C++ DataFrame library for multi-dimensional parameter sweeps.

## Features

- **4 scalar column types**: `int`, `double`, `string`, `complex<double>`
- **List and matrix columns**: store per-row arrays (`add_list_column`) or matrices (`add_matrix_column`); access elements with 1-based indexing
- **Multi-index**:
	- `add_uniform_index(...)` — Cartesian / fully regular dimensions
	- `add_grouped_index(...)` — equal-sized groups, flat value list
	- `add_grouped_index_groups(...)` — ragged groups (each outer group may have a different number of inner elements)
- **Index kinds**: `Uniform`, `Grouped` (regular or ragged); `is_regular_grouped()` distinguishes the two
- **Index operations**: `set_index`, `reset_index`, `loc`, `sub`, `flat_index`, `multi_index`, `from_product`
- **Mixed index inference**: `set_index(...)` auto-infers Uniform and Regular-Grouped dimensions; ragged structures must be built manually
- **CSV export**: `to_csv()`, `save_csv()`
- **Pretty printing**: `to_string()` expands list/matrix columns into `name(k)` / `name(i,j)` sub-columns
- **Unit formatting**: auto-scaled display for frequency, voltage, time, etc.
- **pybind11 bindings**: full Python API; `df.name` returns a `ColumnGroupProxy` for list/matrix columns, callable as `df.name(k)` or `df.name(i,j)`
- **Qt5 viewer**: `DataFrameModel` + `DataFrameView` widget with lazy loading; list/matrix columns auto-expanded into separate view columns

## Requirements

- C++11 compiler (MSVC, GCC, Clang)
- CMake 3.10+
- vcpkg (set `VCPKG_ROOT` environment variable)
- Qt5 (optional, for viewer widget)
- Python 3 (optional, for pybind11 bindings)

## Build

```bash
cmake --preset default
cmake --build build --config Debug
```

## Test

```bash
# C++ tests
./build/Debug/test_dataframe.exe

# Python tests
python test_pybind.py
```

## Quick Start (C++)

```cpp
#include <exprdf/exprdf.hpp>

using namespace exprdf;

// --- Uniform (Cartesian) index ---
auto df = std::make_shared<DataFrame>();
df->add_uniform_index<double>("freq", {1e9, 2e9, 3e9}, "frequency");
df->add_uniform_index<std::string>("port", {"S11", "S21"});
df->add_column<std::complex<double>>("value", ...);

auto s = df->loc({0});   // fix last dim at position 0
df->save_csv("output.csv");

// --- Grouped index (equal-sized groups) ---
auto df2 = std::make_shared<DataFrame>();
df2->add_uniform_index<int>("bias", {1, 2});
df2->add_grouped_index<double>("freq",
    {1.0, 2.0, 3.0,   // bias=1
     1.5, 2.5, 3.5},  // bias=2
    3, "frequency");

// --- Ragged grouped index ---
auto df3 = std::make_shared<DataFrame>();
df3->add_uniform_index<int>("level", {0, 1});
df3->add_grouped_index_groups<int>("number", {{0, 1, 2}, {0, 1}});
df3->add_column<double>("val", {10, 20, 30, 40, 50});

auto sub = df3->loc({1});  // position 1 within each level group → 2 rows

// --- Inspect index dimensions ---
const auto& dim = df3->get_index_dim("number");
assert(dim.is_grouped());
assert(!dim.is_regular_grouped());        // ragged
assert(dim.group_lengths == std::vector<std::size_t>{3, 2});

// --- set_index infers Uniform + Regular Grouped; ragged data throws ---
df->set_index({"bias", "freq"});          // OK: regular grouped
// df->set_index({"level", "number"});    // throws: different group sizes → use add_grouped_index_groups

// --- List column: each row holds a fixed-length list ---
auto df_sp = std::make_shared<DataFrame>();
df_sp->add_uniform_index<double>("freq_GHz", {1.0, 2.0, 4.0, 8.0});
// Each row stores [S11_re, S11_im, S21_re, S21_im]
df_sp->add_list_column<double>("S",
    {{0.80,-0.12,0.50,0.40},
     {0.70,-0.25,0.55,0.55},
     {0.50,-0.45,0.48,0.60},
     {0.20,-0.60,0.30,0.52}});

// Access: 1-based index; returns scalar DataFrame (all rows)
auto s11_re = df_sp->get_list_element("S", 1);   // "S" col = [0.80,0.70,0.50,0.20]
auto s21_re = df_sp->get_list_element("S", 3);

// to_string() expands columns: S(1)  S(2)  S(3)  S(4)
std::cout << df_sp->to_string() << "\n";

// --- Matrix column: each row holds an m×n matrix ---
auto df_y = std::make_shared<DataFrame>();
df_y->add_uniform_index<double>("freq_GHz", {1.0, 2.0, 4.0});
// Each row: 2×2 Y-matrix
df_y->add_matrix_column<double>("Y",
    {{{0.010, 0.003}, {0.003, 0.008}},
     {{0.018, 0.005}, {0.005, 0.015}},
     {{0.030, 0.008}, {0.008, 0.025}}});

// Access: (row_idx, col_idx) 1-based
auto y11 = df_y->get_matrix_element("Y", 1, 1);  // diagonal
auto y12 = df_y->get_matrix_element("Y", 1, 2);  // off-diagonal

// to_string() expands: Y(1,1)  Y(1,2)  Y(2,1)  Y(2,2)
std::cout << df_y->to_string() << "\n";
```

## Quick Start (Python)

```python
import exprdf

# Uniform index
df = exprdf.DataFrame()
df.add_uniform_index("freq", [1e9, 2e9, 3e9], "frequency")
df.add_uniform_index("port", ["S11", "S21"])
df.add_column("value", [...])
sub = df.loc([0])
df.save_csv("output.csv")

# Grouped index (equal-sized groups)
df2 = exprdf.DataFrame()
df2.add_uniform_index("bias", [1, 2])
df2.add_grouped_index("freq", [1.0, 2.0, 3.0, 1.5, 2.5, 3.5], 3, "frequency")

# Ragged grouped index (must be built manually)
df3 = exprdf.DataFrame()
df3.add_uniform_index("level", [0, 1])
df3.add_grouped_index_groups("number", [[0, 1, 2], [0, 1]])
df3.add_column("val", [10.0, 20.0, 30.0, 40.0, 50.0])
sub = df3.loc(1)   # position 1 in each level group

# get_index_dim returns an IndexDim object (not a dict)
dim = df3.get_index_dim("number")
print(dim.kind)            # "grouped"
print(dim.group_lengths)   # [3, 2]
print(dim.is_regular_grouped())  # False

# set_index infers Uniform + Regular Grouped; ragged data throws
df.set_index("bias", "freq")   # OK
# df3_raw.set_index("level", "number")  # raises: ragged structure

# List column: each row holds a fixed-length list
df_sp = exprdf.DataFrame()
df_sp.add_uniform_index("freq_GHz", [1.0, 2.0, 4.0, 8.0])
df_sp.add_list_column("S", [[0.80,-0.12,0.50,0.40],
                             [0.70,-0.25,0.55,0.55],
                             [0.50,-0.45,0.48,0.60],
                             [0.20,-0.60,0.30,0.52]])

# df.S  → ColumnGroupProxy (not a direct value)
# df.S(k) → scalar DataFrame, 1-based k
s11_re = df_sp.S(1)           # DataFrame with "S" col = [0.80,0.70,0.50,0.20]
val = df_sp.S(1)[0]           # 0.80  — row 0 of that scalar DataFrame
print(df_sp.to_string())      # shows S(1)  S(2)  S(3)  S(4)

# Matrix column: each row holds an m×n matrix
df_y = exprdf.DataFrame()
df_y.add_uniform_index("freq_GHz", [1.0, 2.0, 4.0])
df_y.add_matrix_column("Y", [[[0.010,0.003],[0.003,0.008]],
                               [[0.018,0.005],[0.005,0.015]],
                               [[0.030,0.008],[0.008,0.025]]])

# df.Y(i,j) → scalar DataFrame, 1-based (i,j)
y11 = df_y.Y(1,1)             # DataFrame with "Y" col = [0.010,0.018,0.030]
val = df_y.Y(1,2)[0]          # 0.003
print(df_y.to_string())       # shows Y(1,1)  Y(1,2)  Y(2,1)  Y(2,2)
```

## Column Classification API

Each column falls into exactly one of four kinds. Use `column_kind(name)` to identify it, then pick the right access pattern.

| Kind | C++ enum | Python enum | Meaning |
|---|---|---|---|
| Independent | `ColumnKind::Independent` | `ColumnKind.Independent` | Index dimension (use `loc`, `flat_index`, etc.) |
| Scalar | `ColumnKind::Scalar` | `ColumnKind.Scalar` | Ordinary data column, single value per row |
| List | `ColumnKind::List` | `ColumnKind.List` | 1-D array per row; access with `get_list_element(name,k)` / `df.name(k)` |
| Matrix | `ColumnKind::Matrix` | `ColumnKind.Matrix` | 2-D array per row; access with `get_matrix_element(name,i,j)` / `df.name(i,j)` |

### Predicate methods

| C++ method | Python method | Returns true when… |
|---|---|---|
| `column_kind(name)` | `column_kind(name)` | Returns `ColumnKind` enum value |
| `is_independent(name)` | `is_independent(name)` | Column is an index dimension |
| `is_dependent(name)` | `is_dependent(name)` | Column is a data column |
| `is_scalar(name)` | `is_scalar(name)` | No shape (ordinary or index column) |
| `is_list(name)` | `is_list(name)` | Shape is 1-D (`[n]`) |
| `is_matrix(name)` | `is_matrix(name)` | Shape is 2-D (`[m,n]`) |
| `is_index(name)` | `is_index(name)` | Alias for `is_independent` (backward compat) |

### List / matrix column operations

| C++ method | Python method | Description |
|---|---|---|
| `add_list_column<T>(name, data)` | `add_list_column(name, data)` | Add list column (each row is a fixed-length list) |
| `add_matrix_column<T>(name, data)` | `add_matrix_column(name, data)` | Add matrix column (each row is an m×n matrix) |
| `column_shape(name)` | `column_shape(name)` | Returns `[]` (scalar), `[n]` (list), `[m,n]` (matrix) |
| `get_list_element(name, k)` | `get_list_element(name, k)` | Extract element k (1-based) → scalar DataFrame |
| `get_matrix_element(name, i, j)` | `get_matrix_element(name, i, j)` | Extract element (i,j) (1-based) → scalar DataFrame |
| — | `df.name` | Returns `ColumnGroupProxy` for list/matrix columns |
| — | `df.name(k)` | Shorthand for `get_list_element` |
| — | `df.name(i, j)` | Shorthand for `get_matrix_element` |

### Typical dispatch pattern (Python)

```python
for col in df.column_names():
    kind = df.column_kind(col)
    if kind == pdf.ColumnKind.Independent:
        pass  # index dim — use df.loc / df.get_index_dim
    elif kind == pdf.ColumnKind.Scalar:
        data = df.get_column(col)          # list of scalar values
    elif kind == pdf.ColumnKind.List:
        n = df.column_shape(col)[0]        # list length per row
        col1 = df[col + '(1)']             # or: df.S(1) via __getattr__
    elif kind == pdf.ColumnKind.Matrix:
        m, n = df.column_shape(col)        # matrix dimensions
        elem = getattr(df, col)(1, 1)      # or: df.M(1,1)
```

## IndexDim API

| Property / Method | Description |
|---|---|
| `dim.kind` | `"uniform"` or `"grouped"` |
| `dim.levels` | Level values (Uniform only; `None` for Grouped) |
| `dim.level_count` | Number of levels (Uniform) or common group size (Regular Grouped) |
| `dim.group_lengths` | Per-group inner counts (Grouped only) |
| `dim.num_groups` | Outer group count |
| `dim.max_group_length` | Largest group size |
| `dim.quantity` | Physical quantity string |
| `dim.is_uniform()` | True for Uniform kind |
| `dim.is_grouped()` | True for Grouped kind |
| `dim.is_regular_grouped()` | True when all group sizes are equal |

## set_index Rules

| Data structure | Result |
|---|---|
| All groups: same run count, same lengths, same values | `Uniform` dim |
| All groups: same run count, same lengths, different values | `Regular Grouped` dim |
| Groups with different run counts or lengths | **throws** — use `add_grouped_index_groups()` |


## Requirements

- C++11 compiler (MSVC, GCC, Clang)
- CMake 3.10+
- vcpkg (set `VCPKG_ROOT` environment variable)
- Qt5 (optional, for viewer widget)
- Python 3 (optional, for pybind11 bindings)

## Build

```bash
cmake --preset default
cmake --build build --config Debug
```

## Test

```bash
# C++ tests
./build/Debug/test_dataframe.exe

# Python tests
python test_pybind.py
```

## Quick Start (C++)

```cpp
#include <exprdf/exprdf.hpp>

using namespace exprdf;

// --- Uniform (Cartesian) index ---
auto df = std::make_shared<DataFrame>();
df->add_uniform_index<double>("freq", {1e9, 2e9, 3e9}, "frequency");
df->add_uniform_index<std::string>("port", {"S11", "S21"});
df->add_column<std::complex<double>>("value", ...);

auto s = df->loc({0});   // fix last dim at position 0
df->save_csv("output.csv");

// --- Grouped index (equal-sized groups) ---
auto df2 = std::make_shared<DataFrame>();
df2->add_uniform_index<int>("bias", {1, 2});
df2->add_grouped_index<double>("freq",
    {1.0, 2.0, 3.0,   // bias=1
     1.5, 2.5, 3.5},  // bias=2
    3, "frequency");

// --- Ragged grouped index (must be built manually; set_index will throw for ragged data) ---
auto df3 = std::make_shared<DataFrame>();
df3->add_uniform_index<int>("level", {0, 1});
df3->add_grouped_index_groups<int>("number", {{0, 1, 2}, {0, 1}});  // 3 entries for level 0, 2 for level 1
df3->add_column<double>("val", {10, 20, 30, 40, 50});

auto sub = df3->loc({1});  // position 1 within each level group → 2 rows

// --- Inspect index dimensions ---
const auto& dim = df3->get_index_dim("number");
assert(dim.is_grouped());
assert(!dim.is_regular_grouped());        // ragged
assert(dim.group_lengths == std::vector<std::size_t>{3, 2});

// --- set_index infers Uniform + Regular Grouped; ragged data throws ---
df->set_index({"bias", "freq"});          // OK: regular grouped
// df->set_index({"level", "number"});    // throws: different group sizes → use add_grouped_index_groups
```

## Quick Start (Python)

```python
import exprdf

# Uniform index
df = exprdf.DataFrame()
df.add_uniform_index("freq", [1e9, 2e9, 3e9], "frequency")
df.add_uniform_index("port", ["S11", "S21"])
df.add_column("value", [...])
sub = df.loc([0])
df.save_csv("output.csv")

# Grouped index (equal-sized groups)
df2 = exprdf.DataFrame()
df2.add_uniform_index("bias", [1, 2])
df2.add_grouped_index("freq", [1.0, 2.0, 3.0, 1.5, 2.5, 3.5], 3, "frequency")

# Ragged grouped index (must be built manually)
df3 = exprdf.DataFrame()
df3.add_uniform_index("level", [0, 1])
df3.add_grouped_index_groups("number", [[0, 1, 2], [0, 1]])
df3.add_column("val", [10.0, 20.0, 30.0, 40.0, 50.0])
sub = df3.loc(1)   # position 1 in each level group

# get_index_dim returns an IndexDim object (not a dict)
dim = df3.get_index_dim("number")
print(dim.kind)            # "grouped"
print(dim.group_lengths)   # [3, 2]
print(dim.is_regular_grouped())  # False

# set_index infers Uniform + Regular Grouped; ragged data throws
df.set_index("bias", "freq")   # OK
# df3_raw.set_index("level", "number")  # raises: ragged structure
```

## IndexDim API

| Property / Method | Description |
|---|---|
| `dim.kind` | `"uniform"` or `"grouped"` |
| `dim.levels` | Level values (Uniform only; `None` for Grouped) |
| `dim.level_count` | Number of levels (Uniform) or common group size (Regular Grouped) |
| `dim.group_lengths` | Per-group inner counts (Grouped only) |
| `dim.num_groups` | Outer group count |
| `dim.max_group_length` | Largest group size |
| `dim.quantity` | Physical quantity string |
| `dim.is_uniform()` | True for Uniform kind |
| `dim.is_grouped()` | True for Grouped kind |
| `dim.is_regular_grouped()` | True when all group sizes are equal |

## set_index Rules

| Data structure | Result |
|---|---|
| All groups: same run count, same lengths, same values | `Uniform` dim |
| All groups: same run count, same lengths, different values | `Regular Grouped` dim |
| Groups with different run counts or lengths | **throws** — use `add_grouped_index_groups()` |
