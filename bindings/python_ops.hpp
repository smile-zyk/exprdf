#ifndef EXPRDF_BINDINGS_PYTHON_OPS_HPP
#define EXPRDF_BINDINGS_PYTHON_OPS_HPP

#include <pybind11/pybind11.h>
#include <string>
#include <vector>

namespace py = pybind11;

namespace exprdf {
namespace pyops {

enum class ArgType {
    DataFrame,
    Int,
    Double,
    Complex,
    String,
    Bool,
    Any
};

struct ArgSpec {
    ArgType type;
    const char* name;
};

struct OpSpec {
    std::string name;
    std::vector<ArgSpec> args;
    std::size_t min_args;
    std::size_t max_args;
    py::object (*impl)(const py::args&);
    const char* doc;
    bool export_to_module;
};

py::object invoke(const std::string& name, const py::tuple& args);
void bind_module_functions(py::module_& m);

} // namespace pyops
} // namespace exprdf

#endif // EXPRDF_BINDINGS_PYTHON_OPS_HPP
