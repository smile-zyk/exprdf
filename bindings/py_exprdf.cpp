#include <pybind11/pybind11.h>
#include <exprdf/exprdf.hpp>
#include "python_ops.hpp"
#include "py_exprdf_bindings.hpp"

namespace py = pybind11;

PYBIND11_MODULE(exprdf, m) {
    m.doc() = "exprdf: header-only C++ DataFrame library with multi-index support.\n"
              "Column types: int, double, str, complex.\n"
              "Index kinds : Uniform (Cartesian) and Grouped (equal or ragged).";

    bind_types_and_dataframe(m);

    // Setting __array_ufunc__ = None tells numpy to skip its ufunc machinery
    // for DataFrame objects.  When numpy sees `arr OP df` and df has
    // __array_ufunc__ = None it raises TypeError, which causes Python to fall
    // back to df.__radd__ / __rsub__ / __rtruediv__ as intended.
    m.attr("DataFrame").attr("__array_ufunc__") = py::none();

    // Module-level Python operations (centralized in bindings/python_ops.cpp).
    exprdf::pyops::bind_module_functions(m);

    // Special cases not in the centralized operation table (need DataFrame methods):
    m.def("max",
          [](const exprdf::DataFrame& df) { return df.max(); },
          py::arg("df"), "max(df): reduce last independent dim by max of last col");
    m.def("min",
          [](const exprdf::DataFrame& df) { return df.min(); },
          py::arg("df"), "min(df): reduce last independent dim by min of last col");
}
