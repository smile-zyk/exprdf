# exprdf

A header-only C++ DataFrame library for multi-dimensional parameter sweeps.

## Features

- **4 column types**: `int`, `double`, `string`, `complex<double>`
- **Multi-index**:
	- `add_uniform_index(...)` — Cartesian / fully regular dimensions
	- `add_grouped_index(...)` — equal-sized groups, flat value list
	- `add_grouped_index_groups(...)` — ragged groups (each outer group may have a different number of inner elements)
- **Index kinds**: `Uniform`, `Grouped` (regular or ragged); `is_regular_grouped()` distinguishes the two
- **Index operations**: `set_index`, `reset_index`, `loc`, `sub`, `flat_index`, `multi_index`, `from_product`
- **Mixed index inference**: `set_index(...)` auto-infers Uniform and Regular-Grouped dimensions; ragged structures must be built manually
- **CSV export**: `to_csv()`, `save_csv()`
- **Unit formatting**: auto-scaled display for frequency, voltage, time, etc.
- **pybind11 bindings**: full Python API; `get_index_dim()` returns a typed `IndexDim` object
- **Qt5 viewer**: `DataFrameModel` + `DataFrameView` widget with lazy loading

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
