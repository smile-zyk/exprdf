#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <exprdf/exprdf.hpp>

namespace py = pybind11;

// ----------------------------------------------------------------
// ColumnGroupProxy: returned by df.col_name when col_name is a
// list or matrix column.  Calling the proxy extracts the element:
//   proxy(i)    -> df.get_list_element(name, i)        [1-based]
//   proxy(i, j) -> df.get_matrix_element(name, i, j)   [1-based]
// ----------------------------------------------------------------
struct ColumnGroupProxy {
    std::shared_ptr<exprdf::DataFrame> df;
    std::string col_name;
};

PYBIND11_MODULE(exprdf, m) {
    m.doc() = "exprdf: header-only C++ DataFrame library with multi-index support.\n"
              "Column types: int, double, str, complex.\n"
              "Index kinds : Uniform (Cartesian) and Grouped (equal or ragged).";

    // ----------------------------------------------------------------
    // ColumnKind enum
    // ----------------------------------------------------------------
    py::enum_<exprdf::DataFrame::ColumnKind>(m, "ColumnKind")
        .value("Independent", exprdf::DataFrame::ColumnKind::Independent)
        .value("Scalar",      exprdf::DataFrame::ColumnKind::Scalar)
        .value("List",        exprdf::DataFrame::ColumnKind::List)
        .value("Matrix",      exprdf::DataFrame::ColumnKind::Matrix)
        .export_values();

    // ----------------------------------------------------------------
    // ColumnGroupProxy
    // ----------------------------------------------------------------
    py::class_<ColumnGroupProxy>(m, "ColumnGroupProxy")
        .def("__call__", [](const ColumnGroupProxy& self, py::args args) -> py::object {
            if (args.size() == 1) {
                std::size_t i = args[0].cast<std::size_t>();
                return py::cast(self.df->get_list_element(self.col_name, i));
            } else if (args.size() == 2) {
                std::size_t i = args[0].cast<std::size_t>();
                std::size_t j = args[1].cast<std::size_t>();
                return py::cast(self.df->get_matrix_element(self.col_name, i, j));
            }
            throw py::value_error(
                "ColumnGroupProxy: expected 1 (list) or 2 (matrix) index arguments");
        })
        .def("__repr__", [](const ColumnGroupProxy& self) {
            return "<ColumnGroupProxy '" + self.col_name + "'>";
        });

    // ----------------------------------------------------------------
    // IndexDim
    // ----------------------------------------------------------------
    py::class_<exprdf::IndexDim>(m, "IndexDim")
        .def_property_readonly("name", [](const exprdf::IndexDim& d) { return d.name; })
        .def_property_readonly("kind", [](const exprdf::IndexDim& d) {
            return d.is_uniform() ? std::string("uniform") : std::string("grouped");
        })
        .def_property_readonly("levels", [](const exprdf::IndexDim& d) -> py::object {
            if (!d.is_uniform()) return py::none();
            switch (d.levels.tag) {
                case exprdf::DType::Int:     return py::cast(d.levels.as<int>());
                case exprdf::DType::Double:  return py::cast(d.levels.as<double>());
                case exprdf::DType::String:  return py::cast(d.levels.as<std::string>());
                case exprdf::DType::Complex: return py::cast(d.levels.as<std::complex<double>>());
            }
            return py::none();
        })
        .def_property_readonly("quantity", [](const exprdf::IndexDim& d) { return d.levels.quantity; })
        .def_property_readonly("group_lengths", [](const exprdf::IndexDim& d) { return d.group_lengths; })
        .def_property_readonly("num_outer", [](const exprdf::IndexDim& d) { return d.num_outer; })
        .def_property_readonly("level_count", [](const exprdf::IndexDim& d) { return d.level_count(); })
        .def_property_readonly("num_groups",   [](const exprdf::IndexDim& d) { return d.num_groups(); })
        .def_property_readonly("max_group_length", [](const exprdf::IndexDim& d) { return d.max_group_length(); })
        .def("is_uniform",         &exprdf::IndexDim::is_uniform)
        .def("is_grouped",         &exprdf::IndexDim::is_grouped)
        .def("is_regular_grouped", &exprdf::IndexDim::is_regular_grouped)
        .def("__repr__", [](const exprdf::IndexDim& d) {
            std::string s = "IndexDim(name='" + d.name + "', kind=" +
                            (d.is_uniform() ? "uniform" : "grouped");
            if (d.is_uniform())
                s += ", level_count=" + std::to_string(d.level_count());
            else
                s += ", num_groups=" + std::to_string(d.num_groups());
            s += ")";
            return s;
        });

    py::class_<exprdf::DataFrame, std::shared_ptr<exprdf::DataFrame>>(m, "DataFrame")
        .def(py::init<>())

        // ----------------------------------------------------------------
        // Column management
        // ----------------------------------------------------------------

        // add_column: auto-detect type from Python list
        .def("add_column", [](exprdf::DataFrame& self, const std::string& name, py::list data, const std::string& quantity) {
            if (data.empty()) {
                self.add_column<double>(name, {}, quantity);
                return;
            }
            py::object first = data[0];
            if (py::isinstance<py::str>(first)) {
                std::vector<std::string> v;
                for (auto item : data) v.push_back(item.cast<std::string>());
                self.add_column<std::string>(name, v, quantity);
            } else if (py::isinstance<py::int_>(first) && !py::isinstance<py::bool_>(first)) {
                bool has_float = false, has_complex = false;
                for (auto item : data) {
                    if (py::isinstance<py::float_>(item)) has_float = true;
                    if (PyComplex_Check(item.ptr())) has_complex = true;
                }
                if (has_complex) {
                    std::vector<std::complex<double>> v;
                    for (auto item : data) v.push_back(item.cast<std::complex<double>>());
                    self.add_column<std::complex<double>>(name, v, quantity);
                } else if (has_float) {
                    std::vector<double> v;
                    for (auto item : data) v.push_back(item.cast<double>());
                    self.add_column<double>(name, v, quantity);
                } else {
                    std::vector<int> v;
                    for (auto item : data) v.push_back(item.cast<int>());
                    self.add_column<int>(name, v, quantity);
                }
            } else if (PyComplex_Check(first.ptr())) {
                std::vector<std::complex<double>> v;
                for (auto item : data) v.push_back(item.cast<std::complex<double>>());
                self.add_column<std::complex<double>>(name, v, quantity);
            } else {
                std::vector<double> v;
                for (auto item : data) v.push_back(item.cast<double>());
                self.add_column<double>(name, v, quantity);
            }
        }, py::arg("name"), py::arg("data"), py::arg("quantity") = "",
           "Add a column (type auto-detected from data: int, float, str, complex)")

        // insert_column: insert at position with auto-detect type
        .def("insert_column", [](exprdf::DataFrame& self, std::size_t pos, const std::string& name, py::list data, const std::string& quantity) {
            if (data.empty()) {
                self.insert_column<double>(pos, name, {}, quantity);
                return;
            }
            py::object first = data[0];
            if (py::isinstance<py::str>(first)) {
                std::vector<std::string> v;
                for (auto item : data) v.push_back(item.cast<std::string>());
                self.insert_column<std::string>(pos, name, v, quantity);
            } else if (py::isinstance<py::int_>(first) && !py::isinstance<py::bool_>(first)) {
                bool has_float = false, has_complex = false;
                for (auto item : data) {
                    if (py::isinstance<py::float_>(item)) has_float = true;
                    if (PyComplex_Check(item.ptr())) has_complex = true;
                }
                if (has_complex) {
                    std::vector<std::complex<double>> v;
                    for (auto item : data) v.push_back(item.cast<std::complex<double>>());
                    self.insert_column<std::complex<double>>(pos, name, v, quantity);
                } else if (has_float) {
                    std::vector<double> v;
                    for (auto item : data) v.push_back(item.cast<double>());
                    self.insert_column<double>(pos, name, v, quantity);
                } else {
                    std::vector<int> v;
                    for (auto item : data) v.push_back(item.cast<int>());
                    self.insert_column<int>(pos, name, v, quantity);
                }
            } else if (PyComplex_Check(first.ptr())) {
                std::vector<std::complex<double>> v;
                for (auto item : data) v.push_back(item.cast<std::complex<double>>());
                self.insert_column<std::complex<double>>(pos, name, v, quantity);
            } else {
                std::vector<double> v;
                for (auto item : data) v.push_back(item.cast<double>());
                self.insert_column<double>(pos, name, v, quantity);
            }
        }, py::arg("pos"), py::arg("name"), py::arg("data"), py::arg("quantity") = "",
           "Insert a column at position (type auto-detected)")

        // prepend_column: insert at beginning with auto-detect type
        .def("prepend_column", [](exprdf::DataFrame& self, const std::string& name, py::list data, const std::string& quantity) {
            if (data.empty()) {
                self.prepend_column<double>(name, {}, quantity);
                return;
            }
            py::object first = data[0];
            if (py::isinstance<py::str>(first)) {
                std::vector<std::string> v;
                for (auto item : data) v.push_back(item.cast<std::string>());
                self.prepend_column<std::string>(name, v, quantity);
            } else if (py::isinstance<py::int_>(first) && !py::isinstance<py::bool_>(first)) {
                bool has_float = false, has_complex = false;
                for (auto item : data) {
                    if (py::isinstance<py::float_>(item)) has_float = true;
                    if (PyComplex_Check(item.ptr())) has_complex = true;
                }
                if (has_complex) {
                    std::vector<std::complex<double>> v;
                    for (auto item : data) v.push_back(item.cast<std::complex<double>>());
                    self.prepend_column<std::complex<double>>(name, v, quantity);
                } else if (has_float) {
                    std::vector<double> v;
                    for (auto item : data) v.push_back(item.cast<double>());
                    self.prepend_column<double>(name, v, quantity);
                } else {
                    std::vector<int> v;
                    for (auto item : data) v.push_back(item.cast<int>());
                    self.prepend_column<int>(name, v, quantity);
                }
            } else if (PyComplex_Check(first.ptr())) {
                std::vector<std::complex<double>> v;
                for (auto item : data) v.push_back(item.cast<std::complex<double>>());
                self.prepend_column<std::complex<double>>(name, v, quantity);
            } else {
                std::vector<double> v;
                for (auto item : data) v.push_back(item.cast<double>());
                self.prepend_column<double>(name, v, quantity);
            }
        }, py::arg("name"), py::arg("data"), py::arg("quantity") = "",
           "Insert a column at the beginning (type auto-detected)")

        // ----------------------------------------------------------------
        // Element access
        // ----------------------------------------------------------------

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

        // ----------------------------------------------------------------
        // General DataFrame methods
        // ----------------------------------------------------------------
        .def("remove_column", &exprdf::DataFrame::remove_column, py::arg("name"))
        .def("has_column", &exprdf::DataFrame::has_column, py::arg("name"))
        .def("column_names", &exprdf::DataFrame::column_names)
        .def("column_dtype", &exprdf::DataFrame::column_dtype, py::arg("name"))
        .def("column_quantity", &exprdf::DataFrame::column_quantity, py::arg("name"),
             "Get column quantity")
        .def("set_column_quantity", &exprdf::DataFrame::set_column_quantity,
             py::arg("name"), py::arg("quantity"), "Set column quantity")
        .def("column_shape", &exprdf::DataFrame::column_shape, py::arg("name"),
             "Get shape of a column: [] = scalar, [n] = list, [m,n] = matrix")
        .def("column_kind", &exprdf::DataFrame::column_kind, py::arg("name"),
             "Classify column: Independent / Scalar / List / Matrix")
        .def("is_independent", &exprdf::DataFrame::is_independent, py::arg("name"),
             "True if column is an index (independent) dimension")
        .def("is_dependent", &exprdf::DataFrame::is_dependent, py::arg("name"),
             "True if column is a dependent (data) column")
        .def("is_scalar", &exprdf::DataFrame::is_scalar, py::arg("name"),
             "True if column has no shape (ordinary scalar column)")
        .def("is_list", &exprdf::DataFrame::is_list, py::arg("name"),
             "True if column is a 1-D list column")
        .def("is_matrix", &exprdf::DataFrame::is_matrix, py::arg("name"),
             "True if column is a 2-D matrix column")
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

        // ----------------------------------------------------------------
        // Python operator overloads
        // ----------------------------------------------------------------
        .def("__repr__", [](const exprdf::DataFrame& self) {
            return self.to_string();
        })
        .def("__getitem__", [](const exprdf::DataFrame& self, py::object key) -> py::object {
            if (py::isinstance<py::str>(key)) {
                std::string name = key.cast<std::string>();
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
            } else if (py::isinstance<py::int_>(key)) {
                // Integer row index: return value at that row from the last column.
                std::size_t row = key.cast<std::size_t>();
                if (self.num_columns() == 0)
                    throw py::index_error("DataFrame has no columns");
                std::string last = self.column_name(self.num_columns() - 1);
                std::string dt = self.column_dtype(last);
                if (dt == "int")    return py::cast(self.at<int>(last, row));
                if (dt == "double") return py::cast(self.at<double>(last, row));
                if (dt == "string") return py::cast(self.at<std::string>(last, row));
                if (dt == "complex") return py::cast(self.at<std::complex<double>>(last, row));
                throw std::runtime_error("Unknown dtype: " + dt);
            }
            throw py::type_error("Expected str column name or int row index");
        })
        .def("__getattr__", [](std::shared_ptr<exprdf::DataFrame> self, const std::string& name) -> py::object {
            if (!self->has_column(name))
                throw py::attribute_error("DataFrame has no attribute '" + name + "'");
            // List / matrix columns return a callable proxy.
            auto sh = self->column_shape(name);
            if (!sh.empty())
                return py::cast(ColumnGroupProxy{self, name});
            return py::cast(self->sub(name));
        })

        // ----------------------------------------------------------------
        // Multi-index construction
        // ----------------------------------------------------------------

        // add_uniform_index: auto-detect type from Python list
        .def("add_uniform_index", [](exprdf::DataFrame& self, const std::string& name, py::list data, const std::string& quantity) {
            if (data.empty()) {
                throw std::invalid_argument("Index levels cannot be empty");
            }
            py::object first = data[0];
            if (py::isinstance<py::str>(first)) {
                std::vector<std::string> v;
                for (auto item : data) v.push_back(item.cast<std::string>());
                self.add_uniform_index<std::string>(name, v, quantity);
            } else if (py::isinstance<py::int_>(first) && !py::isinstance<py::bool_>(first)) {
                bool has_float = false;
                for (auto item : data) {
                    if (py::isinstance<py::float_>(item)) has_float = true;
                }
                if (has_float) {
                    std::vector<double> v;
                    for (auto item : data) v.push_back(item.cast<double>());
                    self.add_uniform_index<double>(name, v, quantity);
                } else {
                    std::vector<int> v;
                    for (auto item : data) v.push_back(item.cast<int>());
                    self.add_uniform_index<int>(name, v, quantity);
                }
            } else {
                std::vector<double> v;
                for (auto item : data) v.push_back(item.cast<double>());
                self.add_uniform_index<double>(name, v, quantity);
            }
        }, py::arg("name"), py::arg("data"), py::arg("quantity") = "",
           "Add an independent (index) dimension with auto-detected type")

        // add_grouped_index: auto-detect type from Python list
        .def("add_grouped_index", [](exprdf::DataFrame& self, const std::string& name, py::list data, std::size_t group_size, const std::string& quantity) {
            if (data.empty()) {
                throw std::invalid_argument("Values cannot be empty");
            }
            py::object first = data[0];
            if (py::isinstance<py::str>(first)) {
                std::vector<std::string> v;
                for (auto item : data) v.push_back(item.cast<std::string>());
                self.add_grouped_index<std::string>(name, v, group_size, quantity);
            } else if (py::isinstance<py::int_>(first) && !py::isinstance<py::bool_>(first)) {
                bool has_float = false;
                for (auto item : data) {
                    if (py::isinstance<py::float_>(item)) has_float = true;
                }
                if (has_float) {
                    std::vector<double> v;
                    for (auto item : data) v.push_back(item.cast<double>());
                    self.add_grouped_index<double>(name, v, group_size, quantity);
                } else {
                    std::vector<int> v;
                    for (auto item : data) v.push_back(item.cast<int>());
                    self.add_grouped_index<int>(name, v, group_size, quantity);
                }
            } else {
                std::vector<double> v;
                for (auto item : data) v.push_back(item.cast<double>());
                self.add_grouped_index<double>(name, v, group_size, quantity);
            }
        }, py::arg("name"), py::arg("data"), py::arg("group_size"), py::arg("quantity") = "",
           "Add a grouped index dimension (equal-sized groups, flat input)")

        // add_grouped_index_groups: grouped input, inner lists may differ in size
        .def("add_grouped_index_groups", [](exprdf::DataFrame& self, const std::string& name, py::list groups, const std::string& quantity) {
            if (groups.empty())
                throw std::invalid_argument("groups cannot be empty");
            py::list first_group = groups[0].cast<py::list>();
            if (first_group.empty())
                throw std::invalid_argument("Each grouped group must have at least one element");

            py::object first = first_group[0];
            auto to_vv_str = [&]() {
                std::vector<std::vector<std::string>> vv;
                for (auto g : groups) {
                    py::list gl = g.cast<py::list>();
                    std::vector<std::string> row;
                    for (auto item : gl) row.push_back(item.cast<std::string>());
                    vv.push_back(std::move(row));
                }
                return vv;
            };
            auto to_vv_double = [&]() {
                std::vector<std::vector<double>> vv;
                for (auto g : groups) {
                    py::list gl = g.cast<py::list>();
                    std::vector<double> row;
                    for (auto item : gl) row.push_back(item.cast<double>());
                    vv.push_back(std::move(row));
                }
                return vv;
            };
            auto to_vv_int = [&]() {
                std::vector<std::vector<int>> vv;
                for (auto g : groups) {
                    py::list gl = g.cast<py::list>();
                    std::vector<int> row;
                    for (auto item : gl) row.push_back(item.cast<int>());
                    vv.push_back(std::move(row));
                }
                return vv;
            };

            if (py::isinstance<py::str>(first)) {
                self.add_grouped_index_groups<std::string>(name, to_vv_str(), quantity);
            } else if (py::isinstance<py::int_>(first) && !py::isinstance<py::bool_>(first)) {
                // check if any value is float
                bool has_float = false;
                for (auto g : groups) {
                    py::list gl = g.cast<py::list>();
                    for (auto item : gl)
                        if (py::isinstance<py::float_>(item)) { has_float = true; break; }
                    if (has_float) break;
                }
                if (has_float)
                    self.add_grouped_index_groups<double>(name, to_vv_double(), quantity);
                else
                    self.add_grouped_index_groups<int>(name, to_vv_int(), quantity);
            } else {
                self.add_grouped_index_groups<double>(name, to_vv_double(), quantity);
            }
        }, py::arg("name"), py::arg("groups"), py::arg("quantity") = "",
           "Add a grouped index dimension where inner group sizes may differ")

        // add_grouped_index_groups (flat overload): flat data + explicit per-group sizes
        .def("add_grouped_index_groups", [](exprdf::DataFrame& self, const std::string& name, py::list data, py::list group_sizes, const std::string& quantity) {
            if (data.empty())
                throw std::invalid_argument("data cannot be empty");
            std::vector<std::size_t> sizes;
            for (auto s : group_sizes) sizes.push_back(s.cast<std::size_t>());

            py::object first = data[0];
            if (py::isinstance<py::str>(first)) {
                std::vector<std::string> v;
                for (auto item : data) v.push_back(item.cast<std::string>());
                self.add_grouped_index_groups<std::string>(name, v, sizes, quantity);
            } else if (py::isinstance<py::int_>(first) && !py::isinstance<py::bool_>(first)) {
                bool has_float = false;
                for (auto item : data)
                    if (py::isinstance<py::float_>(item)) { has_float = true; break; }
                if (has_float) {
                    std::vector<double> v;
                    for (auto item : data) v.push_back(item.cast<double>());
                    self.add_grouped_index_groups<double>(name, v, sizes, quantity);
                } else {
                    std::vector<int> v;
                    for (auto item : data) v.push_back(item.cast<int>());
                    self.add_grouped_index_groups<int>(name, v, sizes, quantity);
                }
            } else {
                std::vector<double> v;
                for (auto item : data) v.push_back(item.cast<double>());
                self.add_grouped_index_groups<double>(name, v, sizes, quantity);
            }
        }, py::arg("name"), py::arg("data"), py::arg("group_sizes"), py::arg("quantity") = "",
           "Add a grouped index dimension from flat data + explicit per-group sizes")

        // ----------------------------------------------------------------
        // Multi-index queries & addressing
        // ----------------------------------------------------------------
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

        .def("get_index_dim", [](const exprdf::DataFrame& self, std::size_t dim) -> exprdf::IndexDim {
            return self.get_index_dim(dim);
        }, py::arg("dim"), "Get index dimension (by position)")
        .def("get_index_dim", [](const exprdf::DataFrame& self, const std::string& name) -> exprdf::IndexDim {
            return self.get_index_dim(name);
        }, py::arg("name"), "Get index dimension (by name)")

        .def("strides", &exprdf::DataFrame::strides,
             "Get strides for each index dimension")
        .def("flat_index", &exprdf::DataFrame::flat_index, py::arg("indices"),
             "Compute flat row index from fully-specified multi-index")
        .def("multi_index", &exprdf::DataFrame::multi_index, py::arg("flat"),
             "Decompose flat row index into per-dimension indices (inverse of flat_index)")

        .def("loc", [](const exprdf::DataFrame& self, py::args args) {
            std::vector<int64_t> indices;
            for (auto a : args) indices.push_back(a.cast<int64_t>());
            return self.loc(indices);
        }, "Multi-index loc: fix innermost N dimensions (right-aligned), -1 = wildcard")

        .def("sub", &exprdf::DataFrame::sub, py::arg("name"),
             "Extract sub-DataFrame by column name")

        // ----------------------------------------------------------------
        // List / matrix column construction and element extraction
        // ----------------------------------------------------------------

        // add_list_column: data is a list-of-lists; all inner lists must be same length.
        // Indexing via get_list_element(name, k) or df.name(k), both 1-based.
        .def("add_list_column", [](exprdf::DataFrame& self, const std::string& name,
                                   py::list data, const std::string& quantity) {
            if (data.empty()) {
                self.add_list_column<double>(name, {}, quantity);
                return;
            }
            py::list first_row = data[0].cast<py::list>();
            if (first_row.empty())
                throw std::invalid_argument("List column: inner lists cannot be empty");
            py::object first = first_row[0];
            bool has_float = false, has_complex = false;
            for (auto row_item : data) {
                py::list rl = row_item.cast<py::list>();
                for (auto item : rl) {
                    if (py::isinstance<py::float_>(item)) has_float = true;
                    if (PyComplex_Check(item.ptr())) has_complex = true;
                }
            }
            auto build_vv_dbl = [&]() {
                std::vector<std::vector<double>> vv;
                for (auto row_item : data) {
                    py::list rl = row_item.cast<py::list>();
                    std::vector<double> row;
                    for (auto item : rl) row.push_back(item.cast<double>());
                    vv.push_back(std::move(row));
                }
                return vv;
            };
            if (has_complex) {
                std::vector<std::vector<std::complex<double>>> vv;
                for (auto row_item : data) {
                    py::list rl = row_item.cast<py::list>();
                    std::vector<std::complex<double>> row;
                    for (auto item : rl) row.push_back(item.cast<std::complex<double>>());
                    vv.push_back(std::move(row));
                }
                self.add_list_column<std::complex<double>>(name, vv, quantity);
            } else if (has_float || !py::isinstance<py::int_>(first) ||
                       py::isinstance<py::bool_>(first)) {
                self.add_list_column<double>(name, build_vv_dbl(), quantity);
            } else {
                std::vector<std::vector<int>> vv;
                for (auto row_item : data) {
                    py::list rl = row_item.cast<py::list>();
                    std::vector<int> row;
                    for (auto item : rl) row.push_back(item.cast<int>());
                    vv.push_back(std::move(row));
                }
                self.add_list_column<int>(name, vv, quantity);
            }
        }, py::arg("name"), py::arg("data"), py::arg("quantity") = "",
           "Add a list column (data = list of same-length lists); index via df.name(k) [1-based]")

        // add_matrix_column: data is a list of m×n matrices (list-of-list-of-lists).
        .def("add_matrix_column", [](exprdf::DataFrame& self, const std::string& name,
                                     py::list data, const std::string& quantity) {
            if (data.empty()) {
                self.add_matrix_column<double>(name, {}, quantity);
                return;
            }
            bool has_float = false, has_complex = false;
            for (auto mat_item : data) {
                py::list mat_rows = mat_item.cast<py::list>();
                for (auto mat_row : mat_rows) {
                    py::list mat_cols = mat_row.cast<py::list>();
                    for (auto item : mat_cols) {
                        if (py::isinstance<py::float_>(item)) has_float = true;
                        if (PyComplex_Check(item.ptr())) has_complex = true;
                    }
                }
            }
            auto build_vvv_dbl = [&]() {
                std::vector<std::vector<std::vector<double>>> vvv;
                for (auto mat_item : data) {
                    py::list mat_rows = mat_item.cast<py::list>();
                    std::vector<std::vector<double>> mat;
                    for (auto mat_row : mat_rows) {
                        py::list mat_cols = mat_row.cast<py::list>();
                        std::vector<double> row;
                        for (auto item : mat_cols) row.push_back(item.cast<double>());
                        mat.push_back(std::move(row));
                    }
                    vvv.push_back(std::move(mat));
                }
                return vvv;
            };
            if (has_complex) {
                std::vector<std::vector<std::vector<std::complex<double>>>> vvv;
                for (auto mat_item : data) {
                    py::list mat_rows = mat_item.cast<py::list>();
                    std::vector<std::vector<std::complex<double>>> mat;
                    for (auto mat_row : mat_rows) {
                        py::list mat_cols = mat_row.cast<py::list>();
                        std::vector<std::complex<double>> row;
                        for (auto item : mat_cols) row.push_back(item.cast<std::complex<double>>());
                        mat.push_back(std::move(row));
                    }
                    vvv.push_back(std::move(mat));
                }
                self.add_matrix_column<std::complex<double>>(name, vvv, quantity);
            } else if (has_float) {
                self.add_matrix_column<double>(name, build_vvv_dbl(), quantity);
            } else {
                // Check if int
                py::list first_mat = data[0].cast<py::list>();
                py::object first = first_mat[0].cast<py::list>()[0];
                if (py::isinstance<py::int_>(first) && !py::isinstance<py::bool_>(first)) {
                    std::vector<std::vector<std::vector<int>>> vvv;
                    for (auto mat_item : data) {
                        py::list mat_rows = mat_item.cast<py::list>();
                        std::vector<std::vector<int>> mat;
                        for (auto mat_row : mat_rows) {
                            py::list mat_cols = mat_row.cast<py::list>();
                            std::vector<int> row;
                            for (auto item : mat_cols) row.push_back(item.cast<int>());
                            mat.push_back(std::move(row));
                        }
                        vvv.push_back(std::move(mat));
                    }
                    self.add_matrix_column<int>(name, vvv, quantity);
                } else {
                    self.add_matrix_column<double>(name, build_vvv_dbl(), quantity);
                }
            }
        }, py::arg("name"), py::arg("data"), py::arg("quantity") = "",
           "Add a matrix column (data = list of m×n matrices); index via df.name(i,j) [1-based]")

        // get_list_element / get_matrix_element: explicit extraction methods.
        .def("get_list_element", &exprdf::DataFrame::get_list_element,
             py::arg("name"), py::arg("idx"),
             "Extract element idx (1-based) from list column -> scalar DataFrame")
        .def("get_matrix_element", &exprdf::DataFrame::get_matrix_element,
             py::arg("name"), py::arg("row_idx"), py::arg("col_idx"),
             "Extract element (row_idx, col_idx) (1-based) from matrix column -> scalar DataFrame")

        // ----------------------------------------------------------------
        // Metadata
        // ----------------------------------------------------------------
        .def_property("path",
            [](const exprdf::DataFrame& self) { return self.path(); },
            [](exprdf::DataFrame& self, const std::string& v) { self.set_path(v); },
            "Source path")
        .def_property("type",
            [](const exprdf::DataFrame& self) { return self.type(); },
            [](exprdf::DataFrame& self, const std::string& v) { self.set_type(v); },
            "Data type tag (inherited by loc/sub)")
        .def_property("name",
            [](const exprdf::DataFrame& self) { return self.name(); },
            [](exprdf::DataFrame& self, const std::string& v) { self.set_name(v); },
            "DataFrame name")

        // ----------------------------------------------------------------
        // Arithmetic operators (operate on the last column)
        // ----------------------------------------------------------------
        .def("__add__", [](const exprdf::DataFrame& self, const exprdf::DataFrame& other) {
            return self + other;
        }, py::arg("other"), "Element-wise add of last columns (df + df)")
        .def("__sub__", [](const exprdf::DataFrame& self, const exprdf::DataFrame& other) {
            return self - other;
        }, py::arg("other"), "Element-wise subtract of last columns (df - df)")
        .def("__mul__", [](const exprdf::DataFrame& self, const exprdf::DataFrame& other) {
            return self * other;
        }, py::arg("other"), "Element-wise multiply of last columns (df * df)")
        .def("__truediv__", [](const exprdf::DataFrame& self, const exprdf::DataFrame& other) {
            return self / other;
        }, py::arg("other"), "Element-wise divide of last columns (df / df)")
        // scalar on right
        .def("__add__", [](const exprdf::DataFrame& self, double s) {
            return self + s;
        }, py::arg("scalar"), "Add scalar to last column (df + scalar)")
        .def("__sub__", [](const exprdf::DataFrame& self, double s) {
            return self - s;
        }, py::arg("scalar"), "Subtract scalar from last column (df - scalar)")
        .def("__mul__", [](const exprdf::DataFrame& self, double s) {
            return self * s;
        }, py::arg("scalar"), "Multiply last column by scalar (df * scalar)")
        .def("__truediv__", [](const exprdf::DataFrame& self, double s) {
            return self / s;
        }, py::arg("scalar"), "Divide last column by scalar (df / scalar)")
        // scalar on left
        .def("__radd__", [](const exprdf::DataFrame& self, double s) {
            return self + s;
        }, py::arg("scalar"), "Add scalar to last column (scalar + df)")
        .def("__rmul__", [](const exprdf::DataFrame& self, double s) {
            return self * s;
        }, py::arg("scalar"), "Multiply last column by scalar (scalar * df)")
        .def("__rsub__", [](const exprdf::DataFrame& self, double s) {
            return s - self;
        }, py::arg("scalar"), "scalar - df on last column")
        .def("__rtruediv__", [](const exprdf::DataFrame& self, double s) {
            return s / self;
        }, py::arg("scalar"), "scalar / df on last column");

    // ----------------------------------------------------------------
    // Module-level math functions (operate on the last column)
    // ----------------------------------------------------------------
    m.def("abs",    [](const exprdf::DataFrame& df) { return df.math_abs();    }, py::arg("df"), "abs(df): |last col|, int→int, double→double, complex→double");
    m.def("mag",    [](const exprdf::DataFrame& df) { return df.math_mag();    }, py::arg("df"), "mag(df): magnitude of last col (same as abs)");
    m.def("real",   [](const exprdf::DataFrame& df) { return df.math_real();   }, py::arg("df"), "real(df): real part of last col → double");
    m.def("imag",   [](const exprdf::DataFrame& df) { return df.math_imag();   }, py::arg("df"), "imag(df): imaginary part of last col → double");
    m.def("phase",  [](const exprdf::DataFrame& df) { return df.math_phase();  }, py::arg("df"), "phase(df): phase angle (radians) of last col → double");
    m.def("dB",     [](const exprdf::DataFrame& df) { return df.math_dB();     }, py::arg("df"), "dB(df): 20·log10(|last col|) → double");
    m.def("dBm",    [](const exprdf::DataFrame& df) { return df.math_dBm();    }, py::arg("df"), "dBm(df): 10·log10(|last col|·1000) → double");
    m.def("wtodBm", [](const exprdf::DataFrame& df) { return df.math_wtodBm(); }, py::arg("df"), "wtodBm(df): 10·log10(last col·1000), real input only → double");
    m.def("sqr",    [](const exprdf::DataFrame& df) { return df.math_sqr();    }, py::arg("df"), "sqr(df): x² on last col, preserves type");
    m.def("sqrt",   [](const exprdf::DataFrame& df) { return df.math_sqrt();   }, py::arg("df"), "sqrt(df): √last col, int→double, complex→complex");
    m.def("exp",    [](const exprdf::DataFrame& df) { return df.math_exp();    }, py::arg("df"), "exp(df): e^(last col), int→double, complex→complex");
    m.def("ln",     [](const exprdf::DataFrame& df) { return df.math_ln();     }, py::arg("df"), "ln(df): natural log of last col, int→double, complex→complex");
    m.def("log10",  [](const exprdf::DataFrame& df) { return df.math_log10();  }, py::arg("df"), "log10(df): log₁₀ of last col, int→double, complex→complex");
}
