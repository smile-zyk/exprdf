# exprdf

A header-only C++ DataFrame library for multi-dimensional parameter sweeps.

## Features

- **4 column types**: `int`, `double`, `string`, `complex<double>`
- **Multi-index**:
	- `add_uniform_index(...)` for Cartesian dimensions
	- `add_varying_index(...)` / `add_varying_index_groups(...)` for grouped-varying dimensions
- **Index operations**: `set_index`, `reset_index`, `loc`, `sub`, `from_product`
- **Mixed index inference**: `set_index(...)` can infer uniform/varying dimensions from grouped row layout
- **CSV export**: `to_csv()`, `save_csv()`
- **Unit formatting**: auto-scaled display for frequency, voltage, time, etc.
- **pybind11 bindings**: full Python API with dot/bracket access
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

auto df = std::make_shared<DataFrame>();
df->add_uniform_index<double>("freq", {1e9, 2e9, 3e9}, "Frequency");
df->add_uniform_index<std::string>("port", {"S11", "S21"});
df->add_column<std::complex<double>>("value", ...);

// Slice by index
auto s = df->loc({0});  // fix last dim (port=S11)

// Export
df->save_csv("output.csv");

// Grouped varying index (values differ per outer group)
auto df2 = std::make_shared<DataFrame>();
df2->add_uniform_index<int>("bias", {1, 2});
df2->add_varying_index_groups<double>("freq", {{1.0, 2.0, 3.0}, {1.5, 2.5, 3.5}}, "GHz");

// First index must be uniform; varying index requires an outer group.
```

## Quick Start (Python)

```python
import exprdf

df = exprdf.DataFrame()
df.add_uniform_index("freq", [1e9, 2e9, 3e9], "Frequency")
df.add_uniform_index("port", ["S11", "S21"])
df.add_column("value", [...])

sub = df.loc([0])
df.save_csv("output.csv")

# Grouped varying index
df2 = exprdf.DataFrame()
df2.add_uniform_index("bias", [1, 2])
df2.add_varying_index_groups("freq", [[1.0, 2.0, 3.0], [1.5, 2.5, 3.5]], "GHz")

# set_index can infer mixed uniform/varying dimensions from grouped layout
```
