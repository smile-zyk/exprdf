#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <exprdf/exprdf.hpp>

namespace py = pybind11;

PYBIND11_MODULE(exprdf, m) {
    m.doc() = "DataFrame C++ library exposed to Python";

    py::class_<exprdf::DataFrame, std::shared_ptr<exprdf::DataFrame>>(m, "DataFrame")
        .def(py::init<>())

        // add_column: auto-detect type from Python list
        .def("add_column", [](exprdf::DataFrame& self, const std::string& name, py::list data, const std::string& unit) {
            if (data.empty()) {
                self.add_column<double>(name, {}, unit);
                return;
            }
            py::object first = data[0];
            if (py::isinstance<py::str>(first)) {
                std::vector<std::string> v;
                for (auto item : data) v.push_back(item.cast<std::string>());
                self.add_column<std::string>(name, v, unit);
            } else if (py::isinstance<py::int_>(first) && !py::isinstance<py::bool_>(first)) {
                bool has_float = false, has_complex = false;
                for (auto item : data) {
                    if (py::isinstance<py::float_>(item)) has_float = true;
                    if (PyComplex_Check(item.ptr())) has_complex = true;
                }
                if (has_complex) {
                    std::vector<std::complex<double>> v;
                    for (auto item : data) v.push_back(item.cast<std::complex<double>>());
                    self.add_column<std::complex<double>>(name, v, unit);
                } else if (has_float) {
                    std::vector<double> v;
                    for (auto item : data) v.push_back(item.cast<double>());
                    self.add_column<double>(name, v, unit);
                } else {
                    std::vector<int> v;
                    for (auto item : data) v.push_back(item.cast<int>());
                    self.add_column<int>(name, v, unit);
                }
            } else if (PyComplex_Check(first.ptr())) {
                std::vector<std::complex<double>> v;
                for (auto item : data) v.push_back(item.cast<std::complex<double>>());
                self.add_column<std::complex<double>>(name, v, unit);
            } else {
                std::vector<double> v;
                for (auto item : data) v.push_back(item.cast<double>());
                self.add_column<double>(name, v, unit);
            }
        }, py::arg("name"), py::arg("data"), py::arg("unit") = "",
           "Add a column (type auto-detected from data: int, float, str, complex)")

        // insert_column: insert at position with auto-detect type
        .def("insert_column", [](exprdf::DataFrame& self, std::size_t pos, const std::string& name, py::list data, const std::string& unit) {
            if (data.empty()) {
                self.insert_column<double>(pos, name, {}, unit);
                return;
            }
            py::object first = data[0];
            if (py::isinstance<py::str>(first)) {
                std::vector<std::string> v;
                for (auto item : data) v.push_back(item.cast<std::string>());
                self.insert_column<std::string>(pos, name, v, unit);
            } else if (py::isinstance<py::int_>(first) && !py::isinstance<py::bool_>(first)) {
                bool has_float = false, has_complex = false;
                for (auto item : data) {
                    if (py::isinstance<py::float_>(item)) has_float = true;
                    if (PyComplex_Check(item.ptr())) has_complex = true;
                }
                if (has_complex) {
                    std::vector<std::complex<double>> v;
                    for (auto item : data) v.push_back(item.cast<std::complex<double>>());
                    self.insert_column<std::complex<double>>(pos, name, v, unit);
                } else if (has_float) {
                    std::vector<double> v;
                    for (auto item : data) v.push_back(item.cast<double>());
                    self.insert_column<double>(pos, name, v, unit);
                } else {
                    std::vector<int> v;
                    for (auto item : data) v.push_back(item.cast<int>());
                    self.insert_column<int>(pos, name, v, unit);
                }
            } else if (PyComplex_Check(first.ptr())) {
                std::vector<std::complex<double>> v;
                for (auto item : data) v.push_back(item.cast<std::complex<double>>());
                self.insert_column<std::complex<double>>(pos, name, v, unit);
            } else {
                std::vector<double> v;
                for (auto item : data) v.push_back(item.cast<double>());
                self.insert_column<double>(pos, name, v, unit);
            }
        }, py::arg("pos"), py::arg("name"), py::arg("data"), py::arg("unit") = "",
           "Insert a column at position (type auto-detected)")

        // prepend_column: insert at beginning with auto-detect type
        .def("prepend_column", [](exprdf::DataFrame& self, const std::string& name, py::list data, const std::string& unit) {
            if (data.empty()) {
                self.prepend_column<double>(name, {}, unit);
                return;
            }
            py::object first = data[0];
            if (py::isinstance<py::str>(first)) {
                std::vector<std::string> v;
                for (auto item : data) v.push_back(item.cast<std::string>());
                self.prepend_column<std::string>(name, v, unit);
            } else if (py::isinstance<py::int_>(first) && !py::isinstance<py::bool_>(first)) {
                bool has_float = false, has_complex = false;
                for (auto item : data) {
                    if (py::isinstance<py::float_>(item)) has_float = true;
                    if (PyComplex_Check(item.ptr())) has_complex = true;
                }
                if (has_complex) {
                    std::vector<std::complex<double>> v;
                    for (auto item : data) v.push_back(item.cast<std::complex<double>>());
                    self.prepend_column<std::complex<double>>(name, v, unit);
                } else if (has_float) {
                    std::vector<double> v;
                    for (auto item : data) v.push_back(item.cast<double>());
                    self.prepend_column<double>(name, v, unit);
                } else {
                    std::vector<int> v;
                    for (auto item : data) v.push_back(item.cast<int>());
                    self.prepend_column<int>(name, v, unit);
                }
            } else if (PyComplex_Check(first.ptr())) {
                std::vector<std::complex<double>> v;
                for (auto item : data) v.push_back(item.cast<std::complex<double>>());
                self.prepend_column<std::complex<double>>(name, v, unit);
            } else {
                std::vector<double> v;
                for (auto item : data) v.push_back(item.cast<double>());
                self.prepend_column<double>(name, v, unit);
            }
        }, py::arg("name"), py::arg("data"), py::arg("unit") = "",
           "Insert a column at the beginning (type auto-detected)")

        // get_column: dispatch by stored dtype (by name)
        .def("get_column", [](const exprdf::DataFrame& self, const std::string& name) -> py::object {
            std::string dt = self.column_dtype(name);
            if (dt == "int")
                return py::cast(self.get_column_as<int>(name));
            else if (dt == "double")
                return py::cast(self.get_column_as<double>(name));
            else if (dt == "string")
                return py::cast(self.get_column_as<std::string>(name));
            else if (dt == "complex")
                return py::cast(self.get_column_as<std::complex<double>>(name));
            throw std::runtime_error("Unknown dtype: " + dt);
        }, py::arg("name"), "Get column data as a Python list (by name)")

        // get_column by index
        .def("get_column", [](const exprdf::DataFrame& self, std::size_t index) -> py::object {
            const exprdf::Column& col = self.get_column(index);
            switch (col.tag) {
                case exprdf::DType::Int:           return py::cast(col.as<int>());
                case exprdf::DType::Double:        return py::cast(col.as<double>());
                case exprdf::DType::String:        return py::cast(col.as<std::string>());
                case exprdf::DType::Complex:  return py::cast(col.as<std::complex<double>>());
            }
            throw std::runtime_error("Unknown dtype");
        }, py::arg("index"), "Get column data as a Python list (by index)")

        // column_name by index
        .def("column_name", &exprdf::DataFrame::column_name, py::arg("index"),
             "Get column name by index")

        // at: dispatch by stored dtype
        .def("at", [](const exprdf::DataFrame& self, const std::string& col, std::size_t row) -> py::object {
            std::string dt = self.column_dtype(col);
            if (dt == "int")
                return py::cast(self.at<int>(col, row));
            else if (dt == "double")
                return py::cast(self.at<double>(col, row));
            else if (dt == "string")
                return py::cast(self.at<std::string>(col, row));
            else if (dt == "complex")
                return py::cast(self.at<std::complex<double>>(col, row));
            throw std::runtime_error("Unknown dtype: " + dt);
        }, py::arg("col"), py::arg("row"), "Get element at (col, row)")

        // set: dispatch by stored dtype
        .def("set", [](exprdf::DataFrame& self, const std::string& col, std::size_t row, py::object value) {
            std::string dt = self.column_dtype(col);
            if (dt == "int")
                self.set<int>(col, row, value.cast<int>());
            else if (dt == "double")
                self.set<double>(col, row, value.cast<double>());
            else if (dt == "string")
                self.set<std::string>(col, row, value.cast<std::string>());
            else if (dt == "complex")
                self.set<std::complex<double>>(col, row, value.cast<std::complex<double>>());
            else
                throw std::runtime_error("Unknown dtype: " + dt);
        }, py::arg("col"), py::arg("row"), py::arg("value"), "Set element at (col, row)")

        // General methods
        .def("remove_column", &exprdf::DataFrame::remove_column, py::arg("name"))
        .def("has_column", &exprdf::DataFrame::has_column, py::arg("name"))
        .def("column_names", &exprdf::DataFrame::column_names)
        .def("column_dtype", &exprdf::DataFrame::column_dtype, py::arg("name"))
        .def("column_quantity", &exprdf::DataFrame::column_quantity, py::arg("name"),
             "Get column quantity")
        .def("set_column_quantity", &exprdf::DataFrame::set_column_quantity,
             py::arg("name"), py::arg("quantity"), "Set column quantity")
        .def("column_unit", &exprdf::DataFrame::column_quantity, py::arg("name"),
             "Get column quantity (alias for column_quantity)")
        .def("set_column_unit", &exprdf::DataFrame::set_column_quantity,
             py::arg("name"), py::arg("quantity"), "Set column quantity (alias for set_column_quantity)")
        .def("num_rows", &exprdf::DataFrame::num_rows)
        .def("num_columns", &exprdf::DataFrame::num_columns)
        .def("head", &exprdf::DataFrame::head, py::arg("n") = 5)
        .def("tail", &exprdf::DataFrame::tail, py::arg("n") = 5)
        .def("slice", &exprdf::DataFrame::slice, py::arg("start"), py::arg("end"))
        .def("copy", &exprdf::DataFrame::copy)
        .def("to_string", &exprdf::DataFrame::to_string, py::arg("max_rows") = 20)

        // rename column
        .def("rename", [](std::shared_ptr<exprdf::DataFrame> self, const std::string& old_name, const std::string& new_name) {
            self->rename(old_name, new_name);
            return self;
        }, py::arg("old_name"), py::arg("new_name"), "Rename a column by name (returns self)")
        .def("rename", [](std::shared_ptr<exprdf::DataFrame> self, std::size_t index, const std::string& new_name) {
            self->rename(index, new_name);
            return self;
        }, py::arg("index"), py::arg("new_name"), "Rename a column by index (returns self)")
        .def("rename_last", [](std::shared_ptr<exprdf::DataFrame> self, const std::string& new_name) {
            self->rename_last(new_name);
            return self;
        }, py::arg("new_name"), "Rename the last column (returns self)")

        .def("to_csv", &exprdf::DataFrame::to_csv, py::arg("delimiter") = ',',
             "Return CSV string")
        .def("save_csv", &exprdf::DataFrame::save_csv, py::arg("filename"), py::arg("delimiter") = ',',
             "Write CSV to file")

        .def("__repr__", [](const exprdf::DataFrame& self) {
            return self.to_string();
        })
        .def("__getitem__", [](const exprdf::DataFrame& self, const std::string& name) -> py::object {
            if (!self.has_column(name))
                throw py::key_error("Column '" + name + "' not found");
            std::string dt = self.column_dtype(name);
            if (dt == "int")
                return py::cast(self.get_column_as<int>(name));
            else if (dt == "double")
                return py::cast(self.get_column_as<double>(name));
            else if (dt == "string")
                return py::cast(self.get_column_as<std::string>(name));
            else if (dt == "complex")
                return py::cast(self.get_column_as<std::complex<double>>(name));
            throw std::runtime_error("Unknown dtype: " + dt);
        })
        .def("__getattr__", [](const exprdf::DataFrame& self, const std::string& name) -> py::object {
            if (!self.has_column(name))
                throw py::attribute_error("DataFrame has no attribute '" + name + "'");
            return py::cast(self.sub(name));
        })

        // --- Multi-index ---

        // add_index: auto-detect type from Python list
        .def("add_index", [](exprdf::DataFrame& self, const std::string& name, py::list data, const std::string& unit) {
            if (data.empty()) {
                throw std::invalid_argument("Index levels cannot be empty");
            }
            py::object first = data[0];
            if (py::isinstance<py::str>(first)) {
                std::vector<std::string> v;
                for (auto item : data) v.push_back(item.cast<std::string>());
                self.add_index<std::string>(name, v, unit);
            } else if (py::isinstance<py::int_>(first) && !py::isinstance<py::bool_>(first)) {
                bool has_float = false;
                for (auto item : data) {
                    if (py::isinstance<py::float_>(item)) has_float = true;
                }
                if (has_float) {
                    std::vector<double> v;
                    for (auto item : data) v.push_back(item.cast<double>());
                    self.add_index<double>(name, v, unit);
                } else {
                    std::vector<int> v;
                    for (auto item : data) v.push_back(item.cast<int>());
                    self.add_index<int>(name, v, unit);
                }
            } else {
                std::vector<double> v;
                for (auto item : data) v.push_back(item.cast<double>());
                self.add_index<double>(name, v, unit);
            }
        }, py::arg("name"), py::arg("data"), py::arg("unit") = "",
           "Add an independent (index) dimension with auto-detected type")

        .def("num_indices", &exprdf::DataFrame::num_indices,
             "Number of independent (index) dimensions")
        .def("index_names", &exprdf::DataFrame::index_names,
             "Names of independent (index) columns")
        .def("independent_names", &exprdf::DataFrame::independent_names,
             "Names of independent (index) columns (alias for index_names)")
        .def("is_index", &exprdf::DataFrame::is_index, py::arg("name"),
             "Check if a column is an independent (index) column")
        .def("dependent_names", &exprdf::DataFrame::dependent_names,
             "Names of dependent (data) columns")

        // set_index: promote existing columns to index dimensions
        .def("set_index", [](exprdf::DataFrame& self, py::args args) {
            std::vector<std::string> names;
            if (args.size() == 1 && py::isinstance<py::list>(args[0])) {
                for (auto item : args[0].cast<py::list>())
                    names.push_back(item.cast<std::string>());
            } else {
                for (auto a : args) names.push_back(a.cast<std::string>());
            }
            self.set_index(names);
        }, "Promote existing columns to index dimensions (must be in Cartesian product order)")

        // reset_index: demote all index dimensions back to regular columns
        .def("reset_index", &exprdf::DataFrame::reset_index,
             "Demote all index dimensions back to regular columns")

        .def("get_index_levels", [](const exprdf::DataFrame& self, std::size_t dim) -> py::object {
            const exprdf::Column& col = self.get_index_levels(dim);
            switch (col.tag) {
                case exprdf::DType::Int:     return py::cast(col.as<int>());
                case exprdf::DType::Double:  return py::cast(col.as<double>());
                case exprdf::DType::String:  return py::cast(col.as<std::string>());
                case exprdf::DType::Complex: return py::cast(col.as<std::complex<double>>());
            }
            throw std::runtime_error("Unknown dtype");
        }, py::arg("dim"), "Get unique level values for an index dimension (by index)")
        .def("get_index_levels", [](const exprdf::DataFrame& self, const std::string& name) -> py::object {
            const exprdf::Column& col = self.get_index_levels(name);
            switch (col.tag) {
                case exprdf::DType::Int:     return py::cast(col.as<int>());
                case exprdf::DType::Double:  return py::cast(col.as<double>());
                case exprdf::DType::String:  return py::cast(col.as<std::string>());
                case exprdf::DType::Complex: return py::cast(col.as<std::complex<double>>());
            }
            throw std::runtime_error("Unknown dtype");
        }, py::arg("name"), "Get unique level values for an index dimension (by name)")

        .def("strides", &exprdf::DataFrame::strides,
             "Get strides for each index dimension")
        .def("flat_index", &exprdf::DataFrame::flat_index, py::arg("indices"),
             "Compute flat row index from fully-specified multi-index")
        .def("multi_index", &exprdf::DataFrame::multi_index, py::arg("flat"),
             "Decompose flat row index into per-dimension indices (inverse of flat_index)")

        .def("loc", [](const exprdf::DataFrame& self, py::args args) {
            std::vector<std::size_t> indices;
            for (auto a : args) indices.push_back(a.cast<std::size_t>());
            return self.loc(indices);
        }, "Multi-index loc: fix innermost N dimensions (right-aligned), return sub-DataFrame")

        .def("sub", &exprdf::DataFrame::sub, py::arg("name"),
             "Extract sub-DataFrame by column name")

        // --- Metadata ---
        .def_property("path",
            [](const exprdf::DataFrame& self) { return self.path(); },
            [](exprdf::DataFrame& self, const std::string& v) { self.set_path(v); },
            "Source path")
        .def_property("type",
            [](const exprdf::DataFrame& self) { return self.type(); },
            [](exprdf::DataFrame& self, const std::string& v) { self.set_type(v); },
            "Data type tag (inherited by loc/sub)")
        .def_property("df_name",
            [](const exprdf::DataFrame& self) { return self.name(); },
            [](exprdf::DataFrame& self, const std::string& v) { self.set_name(v); },
            "DataFrame name");
}
