cmake_minimum_required(VERSION 3.10)

project(Template LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Header-only DataFrame library
add_library(dataframe INTERFACE)
target_include_directories(dataframe INTERFACE ${CMAKE_SOURCE_DIR}/include)

# Original executable
add_executable(Template main.cc)

# C++ DataFrame test
enable_testing()
add_executable(test_dataframe test_dataframe.cc)
target_link_libraries(test_dataframe PRIVATE dataframe)
add_test(NAME test_dataframe COMMAND test_dataframe)

# Python bindings (FetchContent since vcpkg pybind11 has broken dep chain)
include(FetchContent)
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG        v2.11.1
)
FetchContent_MakeAvailable(pybind11)
pybind11_add_module(exprdf bindings/py_exprdf.cpp)
target_link_libraries(exprdf PRIVATE dataframe)

# Qt5 DataFrame widget library
set(CMAKE_AUTOMOC ON)
find_package(Qt5 COMPONENTS Widgets REQUIRED)
add_library(dataframe_qt
    src/qt/DataFrameModel.cpp
    src/qt/DataFrameView.cpp
    include/exprdf/qt/DataFrameModel.h
    include/exprdf/qt/DataFrameView.h
)
target_link_libraries(dataframe_qt PUBLIC dataframe Qt5::Widgets)

# Example viewer executable
add_executable(example_view example_view.cpp)
target_link_libraries(example_view PRIVATE dataframe_qt)

# Deploy Qt DLLs next to example_view
get_target_property(_qmake_path Qt5::qmake IMPORTED_LOCATION)
get_filename_component(_qt_bin_dir "${_qmake_path}" DIRECTORY)
add_custom_command(TARGET example_view POST_BUILD
    COMMAND "${_qt_bin_dir}/windeployqt.exe" "$<TARGET_FILE:example_view>"
    COMMENT "Running windeployqt..."
)