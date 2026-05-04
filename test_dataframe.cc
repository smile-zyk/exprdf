#include <iostream>
#include <cassert>
#include <complex>
#include <string>
#include <vector>
#include <fstream>
#include <cstdio>
#include <exprdf/exprdf.hpp>
#include <exprdf/UnitFormat.hpp>

int main() {
    using namespace exprdf;

    // === Test 1: Basic construction and add columns ===
    std::cout << "=== Test 1: Add columns ===" << std::endl;
    DataFrame frame;
    frame.add_column<int>("id", {1, 2, 3, 4, 5});
    frame.add_column<double>("value", {1.1, 2.2, 3.3, 4.4, 5.5});
    frame.add_column<std::string>("name", {"Alice", "Bob", "Charlie", "Dave", "Eve"});
    frame.add_column<std::complex<double>>("signal",
        {{1.0, 0.5}, {2.0, -1.0}, {0.0, 3.0}, {-1.0, 0.0}, {0.5, 0.5}});


    assert(frame.num_rows() == 5);
    assert(frame.num_columns() == 4);
    std::cout << "PASSED: 5 rows, 4 columns" << std::endl;

    // === Test 2: column_names and has_column ===
    std::cout << "\n=== Test 2: column_names / has_column ===" << std::endl;
    auto names = frame.column_names();
    assert(names.size() == 4);
    assert(frame.has_column("id"));
    assert(!frame.has_column("nonexistent"));
    std::cout << "PASSED" << std::endl;

    // === Test 3: get_column (type-erased) ===
    std::cout << "\n=== Test 3: get_column (type-erased) ===" << std::endl;
    {
        const exprdf::Column& col_id = frame.get_column("id");
        assert(col_id.tag == exprdf::DType::Int);
        assert(col_id.size() == 5);
        assert(col_id.as<int>()[0] == 1);
        assert(col_id.as<int>()[4] == 5);
        assert(col_id.dtype_name() == std::string("int"));

        const exprdf::Column& col_val = frame.get_column("value");
        assert(col_val.tag == exprdf::DType::Double);
        assert(col_val.as<double>()[1] == 2.2);

        const exprdf::Column& col_name = frame.get_column("name");
        assert(col_name.tag == exprdf::DType::String);
        assert(col_name.as<std::string>()[2] == "Charlie");

        const exprdf::Column& col_sig = frame.get_column("signal");
        assert(col_sig.tag == exprdf::DType::Complex);
        assert(col_sig.as<std::complex<double>>()[0] == std::complex<double>(1.0, 0.5));

        // Non-const: modify via get_column
        exprdf::Column& col_mut = frame.get_column("id");
        col_mut.as<int>()[0] = 100;
        assert(frame.at<int>("id", 0) == 100);
        col_mut.as<int>()[0] = 1; // restore
    }
    std::cout << "PASSED" << std::endl;

    // === Test 3b: get_column_as (typed) ===
    std::cout << "\n=== Test 3b: get_column_as (typed) ===" << std::endl;
    const auto& ids = frame.get_column_as<int>("id");
    assert(ids[0] == 1 && ids[4] == 5);
    const auto& vals = frame.get_column_as<double>("value");
    assert(vals[1] == 2.2);
    const auto& strs = frame.get_column_as<std::string>("name");
    assert(strs[2] == "Charlie");
    const auto& sigs = frame.get_column_as<std::complex<double>>("signal");
    assert(sigs[0] == std::complex<double>(1.0, 0.5));
    std::cout << "PASSED" << std::endl;

    // === Test 3c: access by index ===
    std::cout << "\n=== Test 3c: access by index ===" << std::endl;
    assert(frame.column_name(0) == "id");
    assert(frame.column_name(1) == "value");
    assert(frame.column_name(2) == "name");
    assert(frame.column_name(3) == "signal");
    // get_column by index
    const auto& col0 = frame.get_column(static_cast<std::size_t>(0));
    assert(col0.tag == exprdf::DType::Int);
    assert(col0.as<int>()[0] == 1);
    // get_column_as by index
    const auto& vals_by_idx = frame.get_column_as<double>(static_cast<std::size_t>(1));
    assert(vals_by_idx[1] == 2.2);
    // out of range
    {
        bool caught = false;
        try { frame.column_name(99); } catch (const std::out_of_range&) { caught = true; }
        assert(caught);
    }
    std::cout << "PASSED" << std::endl;

    // === Test 4: at / set ===
    std::cout << "\n=== Test 4: at / set ===" << std::endl;
    assert(frame.at<int>("id", 2) == 3);
    frame.set<double>("value", 0, 99.9);
    assert(frame.at<double>("value", 0) == 99.9);
    std::cout << "PASSED" << std::endl;

    // === Test 5: column_dtype ===
    std::cout << "\n=== Test 5: column_dtype ===" << std::endl;
    assert(frame.column_dtype("id") == "int");
    assert(frame.column_dtype("value") == "double");
    assert(frame.column_dtype("name") == "string");
    assert(frame.column_dtype("signal") == "complex");
    std::cout << "PASSED" << std::endl;

    // === Test 6: slice / head / tail ===
    std::cout << "\n=== Test 6: slice / head / tail ===" << std::endl;
    auto sliced = frame.slice(1, 3);
    assert(sliced->num_rows() == 2);
    assert(sliced->at<int>("id", 0) == 2);
    assert(sliced->at<int>("id", 1) == 3);

    auto h = frame.head(2);
    assert(h->num_rows() == 2);

    auto t = frame.tail(2);
    assert(t->num_rows() == 2);
    assert(t->at<int>("id", 0) == 4);
    std::cout << "PASSED" << std::endl;

    // === Test 7: remove_column ===
    std::cout << "\n=== Test 7: remove_column ===" << std::endl;
    auto frame2 = frame.copy();
    frame2->remove_column("name");
    assert(frame2->num_columns() == 3);
    assert(!frame2->has_column("name"));
    std::cout << "PASSED" << std::endl;

    // === Test 8: copy independence ===
    std::cout << "\n=== Test 8: copy independence ===" << std::endl;
    auto frame3 = frame.copy();
    frame3->set<int>("id", 0, 999);
    assert(frame.at<int>("id", 0) == 1);   // original unchanged
    assert(frame3->at<int>("id", 0) == 999);
    std::cout << "PASSED" << std::endl;

    // === Test 9: error handling ===
    std::cout << "\n=== Test 9: error handling ===" << std::endl;
    bool caught = false;
    try { frame.add_column<int>("id", {1, 2}); } catch (const std::invalid_argument&) { caught = true; }
    assert(caught);

    caught = false;
    try { frame.get_column_as<double>("id"); } catch (const std::invalid_argument&) { caught = true; }
    assert(caught);

    caught = false;
    try { frame.add_column<int>("short_col", {1, 2}); } catch (const std::invalid_argument&) { caught = true; }
    assert(caught);
    std::cout << "PASSED" << std::endl;

    // === Test 10: to_string ===
    std::cout << "\n=== Test 10: to_string ===" << std::endl;
    std::cout << frame.to_string() << std::endl;

    // === Test 11: column quantities ===
    std::cout << "\n=== Test 11: column quantities ===" << std::endl;
    DataFrame frame_u;
    frame_u.add_column<double>("voltage", {1.0, 2.0, 3.0}, "voltage");
    frame_u.add_column<double>("current", {0.1, 0.2, 0.3}, "current");
    frame_u.add_column<int>("index", {1, 2, 3});  // no quantity
    assert(frame_u.column_quantity("voltage") == "voltage");
    assert(frame_u.column_quantity("current") == "current");
    assert(frame_u.column_quantity("index") == "");
    frame_u.set_column_quantity("index", "temperature");
    assert(frame_u.column_quantity("index") == "temperature");
    std::cout << frame_u.to_string() << std::endl;
    // slice preserves quantities
    auto slice_u = frame_u.slice(0, 2);
    assert(slice_u->column_quantity("voltage") == "voltage");
    // copy preserves quantities
    auto copy_u = frame_u.copy();
    assert(copy_u->column_quantity("current") == "current");
    std::cout << "PASSED" << std::endl;

        // === Test 11b: unit_format public API ===
        std::cout << "\n=== Test 11b: unit_format public API ===" << std::endl;
        assert(unit_format::format(unit_format::quantity::frequency, 2.5e9) == "2.500 GHz");
        assert(unit_format::Format(unit_format::quantity::frequency, 1500.0) ==
            unit_format::format(unit_format::quantity::frequency, 1500.0));
        auto freq_scale = unit_format::scale_for_range(unit_format::quantity::frequency,
                                  1.0e6, 20.0e6);
        assert(freq_scale.unit_symbol == "MHz");
        assert(unit_format::format(unit_format::quantity::unitless, 25.0) == "25.00");
        unit_format::register_units("temperature", {{"C", 1.0}});
        assert(unit_format::format("temperature", 25.0) == "25.00 C");

        // Test base_scale and 0 value formatting
        auto base_freq = unit_format::base_scale(unit_format::quantity::frequency);
        assert(base_freq.unit_symbol == "Hz");
        assert(base_freq.scale_factor == 1.0);
        
        auto base_unknown = unit_format::base_scale("unknown_quantity");
        assert(base_unknown.unit_symbol == "?");
        assert(unit_format::format(unit_format::quantity::frequency, 0.0) == "0 Hz");

        std::cout << "PASSED" << std::endl;

    // === Test 12: insert_column / prepend_column ===
    std::cout << "\n=== Test 12: insert_column / prepend_column ===" << std::endl;
    {
        DataFrame df_ins;
        df_ins.add_column<int>("a", {1, 2, 3});
        df_ins.add_column<int>("c", {7, 8, 9});
        // Insert "b" at position 1 (between a and c)
        df_ins.insert_column<int>(1, "b", {4, 5, 6});
        assert(df_ins.column_names()[0] == "a");
        assert(df_ins.column_names()[1] == "b");
        assert(df_ins.column_names()[2] == "c");
        assert(df_ins.at<int>("b", 0) == 4);
        // Prepend "z" at the beginning
        df_ins.prepend_column<double>("z", {0.1, 0.2, 0.3});
        assert(df_ins.column_names()[0] == "z");
        assert(df_ins.column_names()[1] == "a");
        assert(df_ins.num_columns() == 4);
        // Insert at end (same as add_column)
        df_ins.insert_column<std::string>(4, "end", {"x", "y", "z"});
        assert(df_ins.column_names()[4] == "end");
        // Out of range should throw
        bool caught = false;
        try { df_ins.insert_column<int>(99, "bad", {1, 2, 3}); }
        catch (const std::out_of_range&) { caught = true; }
        assert(caught);
        // Duplicate name should throw
        caught = false;
        try { df_ins.insert_column<int>(0, "a", {1, 2, 3}); }
        catch (const std::invalid_argument&) { caught = true; }
        assert(caught);
    }
    std::cout << "PASSED" << std::endl;

    // === Test 13: Multi-index — add_uniform_index & Cartesian product ===
    std::cout << "\n=== Test 13: Multi-index add_uniform_index ===" << std::endl;
    {
        DataFrame mi;
        mi.add_uniform_index<int>("a", {1, 2});
        mi.add_uniform_index<int>("b", {10, 20, 30});
        assert(mi.num_rows() == 6);
        assert(mi.num_columns() == 2);
        assert(mi.num_indices() == 2);

        // Verify expanded Cartesian product: a repeats each 3x, b tiles 2x
        auto& a_data = mi.get_column_as<int>("a");
        assert(a_data[0] == 1 && a_data[1] == 1 && a_data[2] == 1);
        assert(a_data[3] == 2 && a_data[4] == 2 && a_data[5] == 2);
        auto& b_data = mi.get_column_as<int>("b");
        assert(b_data[0] == 10 && b_data[1] == 20 && b_data[2] == 30);
        assert(b_data[3] == 10 && b_data[4] == 20 && b_data[5] == 30);

        // Add dependent column
        mi.add_column<double>("value", {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
        assert(mi.num_columns() == 3);
        assert(mi.is_index("a"));
        assert(mi.is_index("b"));
        assert(!mi.is_index("value"));
        assert(mi.index_names() == (std::vector<std::string>{"a", "b"}));
        assert(mi.dependent_names() == (std::vector<std::string>{"value"}));

        std::cout << mi.to_string() << std::endl;
    }
    std::cout << "PASSED" << std::endl;

    // === Test 14: Multi-index — strides & flat_index ===
    std::cout << "\n=== Test 14: strides & flat_index ===" << std::endl;
    {
        DataFrame mi;
        mi.add_uniform_index<int>("a", {1, 2});
        mi.add_uniform_index<int>("b", {10, 20, 30});
        mi.add_column<double>("v", {1, 2, 3, 4, 5, 6});

        auto s = mi.strides();
        assert(s.size() == 2);
        assert(s[0] == 3); // stride for a: size(b) = 3
        assert(s[1] == 1); // stride for b: 1

        // flat_index({0,0}) = row 0, ({0,2}) = row 2, ({1,0}) = row 3
        assert(mi.flat_index({0, 0}) == 0);
        assert(mi.flat_index({0, 2}) == 2);
        assert(mi.flat_index({1, 0}) == 3);
        assert(mi.flat_index({1, 2}) == 5);
    }
    std::cout << "PASSED" << std::endl;

    // === Test 15: Multi-index — loc ===
    std::cout << "\n=== Test 15: loc ===" << std::endl;
    {
        DataFrame mi;
        mi.add_uniform_index<int>("a", {1, 2});
        mi.add_uniform_index<int>("b", {10, 20, 30});
        mi.add_column<double>("v", {1, 2, 3, 4, 5, 6});

        // loc({0}): fix b=10 (innermost at index 0), a varies → 2 rows
        auto sub = mi.loc({0});
        assert(sub->num_rows() == 2);
        assert(sub->num_indices() == 1); // only a remains
        assert(sub->at<double>("v", 0) == 1.0);
        assert(sub->at<double>("v", 1) == 4.0);

        // loc({1}): fix b=20 → rows 1,4
        auto sub2 = mi.loc({1});
        assert(sub2->num_rows() == 2);
        assert(sub2->at<double>("v", 0) == 2.0);

        // loc({0, 1}): right-aligned: 0→a=1, 1→b=20 → 1 row
        auto sub3 = mi.loc({0, 1});
        assert(sub3->num_rows() == 1);
        assert(sub3->num_indices() == 0);
        assert(sub3->at<double>("v", 0) == 2.0);

        // Chained loc: fix b=30 first, then a=2
        auto sub4 = mi.loc({2})->loc({1}); // b=30, then a=2
        assert(sub4->num_rows() == 1);
        assert(sub4->at<double>("v", 0) == 6.0);

        std::cout << "sub (b=10):\n" << sub->to_string() << std::endl;
    }
    std::cout << "PASSED" << std::endl;

    // === Test 16: Multi-index — from_product ===
    std::cout << "\n=== Test 16: from_product ===" << std::endl;
    {
        auto mi = DataFrame::from_product({
            {"x", Column::from_string({"A", "B"})},
            {"y", Column::from_int({1, 2, 3})}
        });
        assert(mi->num_rows() == 6);
        assert(mi->num_indices() == 2);
        assert(mi->get_column_as<std::string>("x")[0] == "A");
        assert(mi->get_column_as<std::string>("x")[3] == "B");
        assert(mi->get_column_as<int>("y")[0] == 1);
        assert(mi->get_column_as<int>("y")[2] == 3);
    }
    std::cout << "PASSED" << std::endl;

    // === Test 17: Multi-index — get_index_dim ===
    std::cout << "\n=== Test 17: get_index_dim ===" << std::endl;
    {
        DataFrame mi;
        mi.add_uniform_index<int>("a", {1, 2});
        mi.add_uniform_index<std::string>("b", {"x", "y", "z"});

        auto& dim_a = mi.get_index_dim(0);
        assert(dim_a.levels.tag == DType::Int);
        assert(dim_a.level_count() == 2);
        assert(dim_a.is_uniform());

        auto& dim_b = mi.get_index_dim("b");
        assert(dim_b.levels.tag == DType::String);
        assert(dim_b.levels.as<std::string>()[1] == "y");
        assert(dim_b.level_count() == 3);
    }
    std::cout << "PASSED" << std::endl;

    // === Test 19: Multi-index — copy preserves index info ===
    std::cout << "\n=== Test 19: copy preserves index ===" << std::endl;
    {
        DataFrame mi;
        mi.add_uniform_index<int>("a", {1, 2});
        mi.add_uniform_index<int>("b", {10, 20});
        mi.add_column<double>("v", {1, 2, 3, 4});

        auto mi2 = mi.copy();
        assert(mi2->num_indices() == 2);
        assert(mi2->is_index("a"));
        assert(mi2->is_index("b"));
        assert(mi2->index_names() == mi.index_names());
    }
    std::cout << "PASSED" << std::endl;

    // === Test 20: Multi-index — 3 dimensions ===
    std::cout << "\n=== Test 20: 3D multi-index ===" << std::endl;
    {
        DataFrame mi;
        mi.add_uniform_index<int>("a", {1, 2});       // 2
        mi.add_uniform_index<int>("b", {10, 20, 30});  // 3
        mi.add_uniform_index<int>("c", {100, 200});    // 2
        assert(mi.num_rows() == 12); // 2*3*2

        mi.add_column<double>("v", {1,2,3,4,5,6,7,8,9,10,11,12});

        auto s = mi.strides();
        assert(s[0] == 6); // 3*2
        assert(s[1] == 2); // 2
        assert(s[2] == 1); // 1

        // loc({1}) -> fix c=200 (innermost), 6 rows
        auto sub = mi.loc({1});
        assert(sub->num_rows() == 6);
        assert(sub->num_indices() == 2); // a, b remain
        assert(sub->at<double>("v", 0) == 2.0);
        assert(sub->at<double>("v", 5) == 12.0);

        // loc({1, 0}) -> fix b=20, c=100, 2 rows
        auto sub2 = mi.loc({1, 0});
        assert(sub2->num_rows() == 2);
        assert(sub2->num_indices() == 1); // only a remains
        assert(sub2->at<double>("v", 0) == 3.0);
        assert(sub2->at<double>("v", 1) == 9.0);

        // loc({1, 0, 1}) -> a=2, b=10, c=200 → row 7
        auto sub3 = mi.loc({1, 0, 1});
        assert(sub3->num_rows() == 1);
        assert(sub3->at<double>("v", 0) == 8.0);
    }
    std::cout << "PASSED" << std::endl;

    // === Test 21: set_index — promote columns to index ===
    std::cout << "\n=== Test 21: set_index ===" << std::endl;
    {
        DataFrame df;
        df.add_column<int>("a", {1, 1, 1, 2, 2, 2});
        df.add_column<int>("b", {10, 20, 30, 10, 20, 30});
        df.add_column<double>("v", {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});

        assert(df.num_indices() == 0);
        df.set_index({"a", "b"});
        assert(df.num_indices() == 2);
        assert(df.is_index("a"));
        assert(df.is_index("b"));
        assert(!df.is_index("v"));
        assert(df.index_names() == (std::vector<std::string>{"a", "b"}));
        assert(df.dependent_names() == (std::vector<std::string>{"v"}));

        // Index columns should be first in col_order
        assert(df.column_names()[0] == "a");
        assert(df.column_names()[1] == "b");
        assert(df.column_names()[2] == "v");

        // get_index_dim should work
        auto& dim_a = df.get_index_dim("a");
        assert(dim_a.level_count() == 2);
        assert(dim_a.levels.as<int>()[0] == 1 && dim_a.levels.as<int>()[1] == 2);
        auto& dim_b = df.get_index_dim("b");
        assert(dim_b.level_count() == 3);

        // loc should work after set_index
        auto sub = df.loc({0}); // fix b=10 (innermost)
        assert(sub->num_rows() == 2);
        assert(sub->at<double>("v", 0) == 1.0);
        assert(sub->at<double>("v", 1) == 4.0);

        auto sub2 = df.loc({0, 1}); // a=1, b=20
        assert(sub2->num_rows() == 1);
        assert(sub2->at<double>("v", 0) == 2.0);
    }
    std::cout << "PASSED" << std::endl;

    // === Test 22: set_index — validation errors ===
    std::cout << "\n=== Test 22: set_index validation ===" << std::endl;
    {
        // a=[1,1,1,2], b=[10,20,30,10]: a=1 has 3 b-values, a=2 has 1 → ragged → throws
        DataFrame df;
        df.add_column<int>("a", {1, 1, 1, 2});
        df.add_column<int>("b", {10, 20, 30, 10});
        df.add_column<double>("v", {1, 2, 3, 4});
        bool caught1 = false;
        try { df.set_index({"a", "b"}); }
        catch (const std::invalid_argument&) { caught1 = true; }
        assert(caught1);

        // a=[1,1,2,2,3] (group sizes 2,2,1), b=[10,20,10,20,10]: ragged → throws
        DataFrame df2;
        df2.add_column<int>("a", {1, 1, 2, 2, 3});
        df2.add_column<int>("b", {10, 20, 10, 20, 10});
        df2.add_column<double>("v", {1, 2, 3, 4, 5});
        bool caught2 = false;
        try { df2.set_index({"a", "b"}); }
        catch (const std::invalid_argument&) { caught2 = true; }
        assert(caught2);

        // Nonexistent column still throws
        DataFrame df3;
        df3.add_column<int>("a", {1, 2});
        bool caught3 = false;
        try { df3.set_index({"nonexistent"}); }
        catch (const std::invalid_argument&) { caught3 = true; }
        assert(caught3);
    }
    std::cout << "PASSED" << std::endl;

    // === Test 23: reset_index ===
    std::cout << "\n=== Test 23: reset_index ===" << std::endl;
    {
        DataFrame df;
        df.add_column<int>("a", {1, 1, 2, 2});
        df.add_column<int>("b", {10, 20, 10, 20});
        df.add_column<double>("v", {1, 2, 3, 4});
        df.set_index({"a", "b"});
        assert(df.num_indices() == 2);

        df.reset_index();
        assert(df.num_indices() == 0);
        assert(!df.is_index("a"));
        assert(!df.is_index("b"));
        // Data is preserved
        assert(df.num_rows() == 4);
        assert(df.at<double>("v", 0) == 1.0);
        assert(df.at<int>("a", 2) == 2);
    }
    std::cout << "PASSED" << std::endl;

    // === Test 24: set_index with single column ===
    std::cout << "\n=== Test 24: set_index single column ===" << std::endl;
    {
        DataFrame df;
        df.add_column<std::string>("freq", {"1GHz", "2GHz", "3GHz"});
        df.add_column<double>("s11", {-10.0, -15.0, -20.0});
        df.set_index("freq");
        assert(df.num_indices() == 1);
        assert(df.is_index("freq"));
        assert(df.column_names()[0] == "freq");

        auto sub = df.loc({1}); // freq="2GHz"
        assert(sub->num_rows() == 1);
        assert(sub->at<double>("s11", 0) == -15.0);
    }
    std::cout << "PASSED" << std::endl;

    // === Test 25: set_index then re-set after reset ===
    std::cout << "\n=== Test 25: set_index after reset ===" << std::endl;
    {
        DataFrame df;
        df.add_column<int>("x", {1, 1, 2, 2});
        df.add_column<int>("y", {10, 20, 10, 20});
        df.add_column<double>("z", {1, 2, 3, 4});

        df.set_index({"x", "y"});
        assert(df.num_indices() == 2);
        df.reset_index();
        assert(df.num_indices() == 0);

        // Can re-set index
        df.set_index({"x", "y"});
        assert(df.num_indices() == 2);
        auto sub = df.loc({0}); // y=10
        assert(sub->num_rows() == 2);
        assert(sub->at<double>("z", 0) == 1.0);
        assert(sub->at<double>("z", 1) == 3.0);
    }
    std::cout << "PASSED" << std::endl;

    // === Test 26: set_index — unordered data (auto-reorder) ===
    std::cout << "\n=== Test 26: set_index unordered ==" << std::endl;
    {
        // Data rows are shuffled: (2,30), (1,10), (2,10), (1,30), (1,20), (2,20)
        DataFrame df;
        df.add_column<int>("a", {2, 1, 2, 1, 1, 2});
        df.add_column<int>("b", {30, 10, 10, 30, 20, 20});
        df.add_column<double>("v", {6.0, 1.0, 4.0, 3.0, 2.0, 5.0});

        df.set_index({"a", "b"});
        assert(df.num_indices() == 2);

        // Unique levels (first-appearance): a=[2,1], b=[30,10,20]
        auto& dim_a = df.get_index_dim("a");
        assert(dim_a.levels.as<int>()[0] == 2 && dim_a.levels.as<int>()[1] == 1);
        auto& dim_b = df.get_index_dim("b");
        assert(dim_b.levels.as<int>()[0] == 30 && dim_b.levels.as<int>()[1] == 10 && dim_b.levels.as<int>()[2] == 20);

        // After reorder, data should be in Cartesian order:
        //   (2,30)=6, (2,10)=4, (2,20)=5, (1,30)=3, (1,10)=1, (1,20)=2
        auto& a_data = df.get_column_as<int>("a");
        assert(a_data[0] == 2 && a_data[1] == 2 && a_data[2] == 2);
        assert(a_data[3] == 1 && a_data[4] == 1 && a_data[5] == 1);
        auto& b_data = df.get_column_as<int>("b");
        assert(b_data[0] == 30 && b_data[1] == 10 && b_data[2] == 20);
        assert(b_data[3] == 30 && b_data[4] == 10 && b_data[5] == 20);
        auto& v_data = df.get_column_as<double>("v");
        assert(v_data[0] == 6.0 && v_data[1] == 4.0 && v_data[2] == 5.0);
        assert(v_data[3] == 3.0 && v_data[4] == 1.0 && v_data[5] == 2.0);

        // loc should work correctly on reordered data
        auto sub = df.loc({1}); // fix b=10 (index 1 in levels [30,10,20])
        assert(sub->num_rows() == 2);
        assert(sub->at<double>("v", 0) == 4.0); // a=2, b=10
        assert(sub->at<double>("v", 1) == 1.0); // a=1, b=10

        std::cout << "Reordered:\n" << df.to_string() << std::endl;
    }
    std::cout << "PASSED" << std::endl;

    // === Test 27: set_index — already sorted (no-op reorder) ===
    std::cout << "\n=== Test 27: set_index already sorted ==" << std::endl;
    {
        DataFrame df;
        df.add_column<int>("a", {1, 1, 2, 2});
        df.add_column<int>("b", {10, 20, 10, 20});
        df.add_column<double>("v", {1.0, 2.0, 3.0, 4.0});
        df.set_index({"a", "b"});
        // Should remain unchanged
        auto& v = df.get_column_as<double>("v");
        assert(v[0] == 1.0 && v[1] == 2.0 && v[2] == 3.0 && v[3] == 4.0);
    }
    std::cout << "PASSED" << std::endl;

    // === Test 28: set_index — 3D unordered ===
    std::cout << "\n=== Test 28: set_index 3D unordered ==" << std::endl;
    {
        // 2 x 2 x 2 = 8 rows, fully shuffled
        // Canonical order for a=[1,2], b=[10,20], c=[100,200]:
        //   (1,10,100)=1, (1,10,200)=2, (1,20,100)=3, (1,20,200)=4,
        //   (2,10,100)=5, (2,10,200)=6, (2,20,100)=7, (2,20,200)=8
        DataFrame df;
        df.add_column<int>("a", {2, 1, 2, 1, 1, 2, 1, 2});
        df.add_column<int>("b", {20, 10, 10, 20, 10, 20, 20, 10});
        df.add_column<int>("c", {200, 100, 100, 200, 200, 100, 100, 200});
        df.add_column<double>("v", {8.0, 1.0, 5.0, 4.0, 2.0, 7.0, 3.0, 6.0});

        df.set_index({"a", "b", "c"});
        assert(df.num_indices() == 3);

        // Unique levels: a=[2,1], b=[20,10], c=[200,100]
        // Cartesian order:
        //   (2,20,200)=8, (2,20,100)=7, (2,10,200)=6, (2,10,100)=5,
        //   (1,20,200)=4, (1,20,100)=3, (1,10,200)=2, (1,10,100)=1
        auto& v = df.get_column_as<double>("v");
        assert(v[0] == 8.0 && v[1] == 7.0 && v[2] == 6.0 && v[3] == 5.0);
        assert(v[4] == 4.0 && v[5] == 3.0 && v[6] == 2.0 && v[7] == 1.0);

        // loc({0}) -> fix c=200 (innermost, index 0)
        auto sub = df.loc({0});
        assert(sub->num_rows() == 4);
        assert(sub->at<double>("v", 0) == 8.0);
        assert(sub->at<double>("v", 1) == 6.0);
    }
    std::cout << "PASSED" << std::endl;

    // === Test 29: set_index — duplicate rows should fail ===
    std::cout << "\n=== Test 29: set_index duplicate rows ==" << std::endl;
    {
        DataFrame df;
        df.add_column<int>("a", {1, 1, 1, 2});
        df.add_column<int>("b", {10, 10, 20, 10}); // (1,10) appears twice
        df.add_column<double>("v", {1, 2, 3, 4});
        bool caught = false;
        try { df.set_index({"a", "b"}); }
        catch (const std::invalid_argument&) { caught = true; }
        assert(caught);
    }
    std::cout << "PASSED" << std::endl;

    // === Test 30: independent_names alias ===
    std::cout << "\n=== Test 30: independent_names alias ==" << std::endl;
    {
        DataFrame df;
        df.add_uniform_index<int>("freq", {1, 2, 3});
        df.add_uniform_index<std::string>("port", {"p1", "p2"});
        df.add_column<double>("val", {1, 2, 3, 4, 5, 6});
        auto inames = df.independent_names();
        auto idx_names = df.index_names();
        assert(inames == idx_names);
        assert(inames.size() == 2);
        assert(inames[0] == "freq");
        assert(inames[1] == "port");
    }
    std::cout << "PASSED" << std::endl;

    // === Test 31: sub — dependent column ===
    std::cout << "\n=== Test 31: sub dependent column ==" << std::endl;
    {
        DataFrame df;
        df.add_uniform_index<int>("a", {1, 2});
        df.add_uniform_index<int>("b", {10, 20, 30});
        df.add_column<double>("x", {1, 2, 3, 4, 5, 6});
        df.add_column<double>("y", {7, 8, 9, 10, 11, 12});
        df.set_type("sparam");

        auto s = df.sub("x");
        assert(s->num_columns() == 3); // a, b, x
        assert(s->has_column("a"));
        assert(s->has_column("b"));
        assert(s->has_column("x"));
        assert(!s->has_column("y"));
        assert(s->num_rows() == 6);
        assert(s->num_indices() == 2);
        assert(s->type() == "sparam");
    }
    std::cout << "PASSED" << std::endl;

    // === Test 32: sub — independent column ===
    std::cout << "\n=== Test 32: sub independent column ==" << std::endl;
    {
        DataFrame df;
        df.add_uniform_index<int>("a", {1, 2});
        df.add_uniform_index<int>("b", {10, 20, 30});
        df.add_uniform_index<int>("c", {100, 200});
        df.add_column<double>("v", std::vector<double>(12, 0.0));
        df.set_type("zparam");

        // sub("b") should return dims a,b only
        auto s = df.sub("b");
        assert(s->num_columns() == 2); // a, b
        assert(s->has_column("a"));
        assert(s->has_column("b"));
        assert(!s->has_column("c"));
        assert(!s->has_column("v"));
        // rows = |a| * |b| = 2 * 3 = 6
        assert(s->num_rows() == 6);
        assert(s->num_indices() == 2);
        assert(s->type() == "zparam");

        // sub("a") should return only dim a
        auto s2 = df.sub("a");
        assert(s2->num_columns() == 1);
        assert(s2->num_rows() == 2);
        assert(s2->num_indices() == 1);
    }
    std::cout << "PASSED" << std::endl;

    // === Test 33: path/type/name metadata ===
    std::cout << "\n=== Test 33: path/type/name metadata ==" << std::endl;
    {
        DataFrame df;
        df.add_uniform_index<int>("f", {1, 2});
        df.add_column<double>("v", {10, 20});

        // Default empty
        assert(df.path().empty());
        assert(df.type().empty());
        assert(df.name().empty());

        df.set_path("/data/test.csv");
        df.set_type("sparam");
        df.set_name("S21");

        assert(df.path() == "/data/test.csv");
        assert(df.type() == "sparam");
        assert(df.name() == "S21");

        // loc inherits type
        auto l = df.loc({0});
        assert(l->type() == "sparam");

        // sub inherits type
        auto s = df.sub("v");
        assert(s->type() == "sparam");
    }
    std::cout << "PASSED" << std::endl;

    // === Test 34: rename column by name ===
    std::cout << "\n=== Test 34: rename by name ==" << std::endl;
    {
        DataFrame df;
        df.add_column<int>("a", {1, 2, 3});
        df.add_column<double>("b", {4.0, 5.0, 6.0});
        df.rename("b", "x");
        assert(!df.has_column("b"));
        assert(df.has_column("x"));
        assert(df.column_names()[1] == "x");
        assert(df.at<double>("x", 0) == 4.0);
        // rename index column (via set_index)
        DataFrame df2;
        df2.add_column<int>("i", {1, 1, 2, 2});
        df2.add_column<int>("j", {10, 20, 10, 20});
        df2.add_column<double>("v", {1, 2, 3, 4});
        df2.set_index({"i", "j"});
        df2.rename("i", "freq");
        assert(df2.is_index("freq"));
        assert(!df2.is_index("i"));
        assert(df2.column_names()[0] == "freq");
    }
    std::cout << "PASSED" << std::endl;

    // === Test 35: rename column by index ===
    std::cout << "\n=== Test 35: rename by index ==" << std::endl;
    {
        DataFrame df;
        df.add_column<int>("a", {1, 2});
        df.add_column<double>("b", {3.0, 4.0});
        df.add_column<std::string>("c", {"x", "y"});
        df.rename(static_cast<std::size_t>(1), "beta");
        assert(df.column_names()[1] == "beta");
        assert(df.at<double>("beta", 0) == 3.0);
        // out of range
        bool caught = false;
        try { df.rename(static_cast<std::size_t>(99), "z"); }
        catch (const std::out_of_range&) { caught = true; }
        assert(caught);
        // duplicate name
        caught = false;
        try { df.rename("beta", "a"); }
        catch (const std::invalid_argument&) { caught = true; }
        assert(caught);
    }
    std::cout << "PASSED" << std::endl;

    // === Test 36: rename_last ===
    std::cout << "\n=== Test 36: rename_last ==" << std::endl;
    {
        DataFrame df;
        df.add_uniform_index<int>("a", {1, 2});
        df.add_uniform_index<int>("b", {10, 20, 30});
        df.add_column<double>("t", {1, 2, 3, 4, 5, 6});
        // chain: sub("t") then rename_last
        auto s = df.sub("t");
        s->rename_last("result");
        assert(s->column_names().back() == "result");
        assert(s->has_column("result"));
        assert(!s->has_column("t"));
        assert(s->at<double>("result", 0) == 1.0);
        // rename same name is no-op
        s->rename("result", "result");
        assert(s->has_column("result"));
    }
    std::cout << "PASSED" << std::endl;

    // === Test 37: to_csv basic ===
    std::cout << "\n=== Test 37: to_csv basic ===" << std::endl;
    {
        DataFrame df;
        df.add_column<int>("id", {1, 2, 3});
        df.add_column<double>("value", {1.1, 2.2, 3.3});
        df.add_column<std::string>("name", {"Alice", "Bob", "Charlie"});
        std::string csv = df.to_csv();
        // Check header
        assert(csv.find("id,value,name\n") == 0);
        // Check a data row
        assert(csv.find("1,1.1,Alice\n") != std::string::npos);
        assert(csv.find("3,3.3,Charlie\n") != std::string::npos);
    }
    std::cout << "PASSED" << std::endl;

    // === Test 38: to_csv with special characters ===
    std::cout << "\n=== Test 38: to_csv escape ===" << std::endl;
    {
        DataFrame df;
        df.add_column<std::string>("text", {"hello", "has,comma", "has\"quote", "has\nnewline"});
        std::string csv = df.to_csv();
        assert(csv.find("\"has,comma\"") != std::string::npos);
        assert(csv.find("\"has\"\"quote\"") != std::string::npos);
        assert(csv.find("\"has\nnewline\"") != std::string::npos);
    }
    std::cout << "PASSED" << std::endl;

    // === Test 39: to_csv custom delimiter ===
    std::cout << "\n=== Test 39: to_csv custom delimiter ===" << std::endl;
    {
        DataFrame df;
        df.add_column<int>("a", {1, 2});
        df.add_column<int>("b", {3, 4});
        std::string csv = df.to_csv('\t');
        assert(csv.find("a\tb\n") == 0);
        assert(csv.find("1\t3\n") != std::string::npos);
    }
    std::cout << "PASSED" << std::endl;

    // === Test 40: save_csv ===
    std::cout << "\n=== Test 40: save_csv ===" << std::endl;
    {
        DataFrame df;
        df.add_column<int>("x", {10, 20});
        df.add_column<double>("y", {1.5, 2.5});
        df.save_csv("test_output.csv");
        // Read back and verify
        std::ifstream ifs("test_output.csv");
        assert(ifs.is_open());
        std::string content((std::istreambuf_iterator<char>(ifs)),
                            std::istreambuf_iterator<char>());
        assert(content == df.to_csv());
        ifs.close();
        std::remove("test_output.csv");
    }
    std::cout << "PASSED" << std::endl;

    // === Test 41: to_csv with complex ===
    std::cout << "\n=== Test 41: to_csv complex ===" << std::endl;
    {
        DataFrame df;
        df.add_column<std::complex<double>>("z", {{1.0, 2.0}, {-1.0, -0.5}});
        std::string csv = df.to_csv();
        assert(csv.find("z\n") == 0);
        assert(csv.find("(1+2j)") != std::string::npos);
    }
    std::cout << "PASSED" << std::endl;

    // === Test 42: to_csv empty DataFrame ===
    std::cout << "\n=== Test 42: to_csv empty ===" << std::endl;
    {
        DataFrame df;
        assert(df.to_csv().empty());
    }
    std::cout << "PASSED" << std::endl;

    // === Test 43: add_grouped_index basic ===
    std::cout << "\n=== Test 43: add_grouped_index basic ===" << std::endl;
    {
        // bias (Uniform): [1, 2]
        // freq (Grouped, uniform_grouped with group_lengths all =3): different per bias group
        DataFrame df;
        df.add_uniform_index<int>("bias", {1, 2});
        df.add_grouped_index<double>("freq",
            {1.0, 2.0, 3.0,   // bias=1
             1.5, 2.5, 3.5},  // bias=2
            3, unit_format::quantity::frequency);

        assert(df.num_rows() == 6);
        assert(df.num_indices() == 2);

        // Check dim info
        auto dim_bias = df.get_index_dim("bias");
        assert(dim_bias.is_uniform());
        assert(dim_bias.level_count() == 2);

        auto dim_freq = df.get_index_dim("freq");
        assert(dim_freq.is_grouped());
        assert(dim_freq.is_regular_grouped());
        assert(dim_freq.num_groups() == 2);
        assert(dim_freq.group_lengths[0] == 3);

        // Verify data layout
        auto bias_col = df.get_column_as<int>("bias");
        assert(bias_col[0] == 1 && bias_col[1] == 1 && bias_col[2] == 1);
        assert(bias_col[3] == 2 && bias_col[4] == 2 && bias_col[5] == 2);

        auto freq_col = df.get_column_as<double>("freq");
        assert(approx_equal(freq_col[0], 1.0));
        assert(approx_equal(freq_col[3], 1.5));

        // Add dependent column
        df.add_column<double>("S11", {-10, -20, -30, -15, -25, -35});
        assert(df.num_rows() == 6);

        std::cout << df.to_string() << std::endl;
    }
    std::cout << "PASSED" << std::endl;

    // === Test 44: add_grouped_index validation ===
    std::cout << "\n=== Test 44: add_grouped_index validation ===" << std::endl;
    {
        // First index cannot be varying
        bool caught = false;
        try {
            DataFrame first;
            first.add_grouped_index<double>("freq", {1.0, 2.0}, 2);
        } catch (const std::invalid_argument&) { caught = true; }
        assert(caught);

        DataFrame df;
        df.add_uniform_index<int>("bias", {1, 2});

        // Wrong number of values
        caught = false;
        try {
            df.add_grouped_index<double>("freq", {1.0, 2.0}, 3);
        } catch (const std::invalid_argument&) { caught = true; }
        assert(caught);

        // group_size = 0
        caught = false;
        try {
            df.add_grouped_index<double>("freq", {}, 0);
        } catch (const std::invalid_argument&) { caught = true; }
        assert(caught);
    }
    std::cout << "PASSED" << std::endl;

    // === Test 45: add_grouped_index with loc ===
    std::cout << "\n=== Test 45: add_grouped_index loc ===" << std::endl;
    {
        DataFrame df;
        df.add_uniform_index<int>("bias", {1, 2});
        df.add_grouped_index<double>("freq",
            {1.0, 2.0, 3.0, 1.5, 2.5, 3.5}, 3);
        df.add_column<double>("S11", {-10, -20, -30, -15, -25, -35});

        // loc({1}) — fix freq dim at index 1 within each group
        auto sub = df.loc({1});
        assert(sub->num_rows() == 2);
        auto s11 = sub->get_column_as<double>("S11");
        assert(approx_equal(s11[0], -20.0));
        assert(approx_equal(s11[1], -25.0));
    }
    std::cout << "PASSED" << std::endl;

    // === Test 46: add_grouped_index basic ===
    std::cout << "\n=== Test 46: add_grouped_index basic ===" << std::endl;
    {
        DataFrame df;
        df.add_uniform_index<int>("bias", {1, 2});
        df.add_grouped_index<double>("freq", {1.0, 2.0, 3.0, 1.5, 2.5, 3.5}, 3,
                                            unit_format::quantity::frequency);
        assert(df.num_rows() == 6);
        auto freq_col = df.get_column_as<double>("freq");
        assert(approx_equal(freq_col[0], 1.0));
        assert(approx_equal(freq_col[3], 1.5));

        // Add an inner uniform index after varying
        df.add_uniform_index<std::string>("port", {"S11", "S21"});
        assert(df.num_rows() == 12);
        auto bias_col = df.get_column_as<int>("bias");
        auto port_col = df.get_column_as<std::string>("port");
        assert(bias_col[0] == 1 && bias_col[1] == 1);
        assert(port_col[0] == "S11" && port_col[1] == "S21");
    }
    std::cout << "PASSED" << std::endl;

    // === Test 48: set_index auto-infer varying in middle ===
    std::cout << "\n=== Test 48: set_index auto infer varying ===" << std::endl;
    {
        DataFrame df;
        df.add_column<int>("bias",  {1,1,1,1,1,1,2,2,2,2,2,2});
        df.add_column<double>("freq", {1,1,2,2,3,3,1.5,1.5,2.5,2.5,3.5,3.5});
        df.add_column<std::string>("port", {"S11","S21","S11","S21","S11","S21",
                                              "S11","S21","S11","S21","S11","S21"});
        df.add_column<double>("S", {-10,-11,-20,-21,-30,-31,-15,-16,-25,-26,-35,-36});

        df.set_index({"bias", "freq", "port"});

        auto d0 = df.get_index_dim("bias");
        auto d1 = df.get_index_dim("freq");
        auto d2 = df.get_index_dim("port");
        assert(d0.is_uniform() && d0.level_count() == 2);
        assert(d1.is_grouped() && d1.is_regular_grouped());
        assert(d1.group_lengths.size() == 6 && d1.group_lengths[0] == 2);
        assert(d2.is_uniform() && d2.level_count() == 2);
    }
    std::cout << "PASSED" << std::endl;

    // === Test 49: set_index keeps unordered uniform behavior ===
    std::cout << "\n=== Test 49: set_index unordered uniform ===" << std::endl;
    {
        DataFrame df;
        df.add_column<int>("a", {2,2,2,1,1,1});
        df.add_column<int>("b", {30,10,20,30,10,20});
        df.add_column<int>("v", {6,4,5,3,1,2});
        df.set_index({"a", "b"});

        auto a = df.get_column_as<int>("a");
        auto b = df.get_column_as<int>("b");
        assert(a[0] == 2 && b[0] == 30);
        assert(a[5] == 1 && b[5] == 20);
    }
    std::cout << "PASSED" << std::endl;

    // === Test 50: set_index infer two varying dims ===
    std::cout << "\n=== Test 50: set_index infer two varying dims ===" << std::endl;
    {
        DataFrame df;
        // Layout by bias(2) x freq(varying=2) x port(varying=2)
        df.add_column<int>("bias",  {1,1,1,1,2,2,2,2});
        df.add_column<double>("freq", {1,1,2,2,1.5,1.5,2.5,2.5});
        df.add_column<std::string>("port", {"S11","S21","S11","S21","S11","S21","S11","S21"});
        df.add_column<double>("S", {-10,-11,-20,-21,-15,-16,-25,-26});

        df.set_index({"bias", "freq", "port"});
        auto d0 = df.get_index_dim("bias");
        auto d1 = df.get_index_dim("freq");
        auto d2 = df.get_index_dim("port");
        assert(d0.is_uniform() && d0.level_count() == 2);
        assert(d1.is_grouped() && d1.is_regular_grouped());
        assert(d1.group_lengths.size() == 4 && d1.group_lengths[0] == 2);
        assert(d2.is_uniform() && d2.level_count() == 2);
    }
    std::cout << "PASSED" << std::endl;

    // === Test 51: add two varying indices and inner uniform ===
    std::cout << "\n=== Test 51: add two varying + inner uniform ===" << std::endl;
    {
        DataFrame df;
        df.add_uniform_index<int>("bias", {1, 2});
        df.add_grouped_index<double>("freq", {1.0, 2.0, 1.5, 2.5}, 2);
        // old_rows=4, group_size=2 => values=8
        df.add_grouped_index<double>("temp", {25,30,25,30,26,31,26,31}, 2, "C");
        df.add_uniform_index<std::string>("port", {"S11", "S21"});

        assert(df.num_rows() == 16);
        auto d_freq = df.get_index_dim("freq");
        auto d_temp = df.get_index_dim("temp");
        auto d_port = df.get_index_dim("port");
        assert(d_freq.is_grouped() && d_freq.is_regular_grouped() && d_freq.group_lengths[0] == 2);
        assert(d_temp.is_grouped() && d_temp.is_regular_grouped() && d_temp.group_lengths[0] == 2);
        assert(d_port.is_uniform() && d_port.level_count() == 2);
    }
    std::cout << "PASSED" << std::endl;

    // === Test 52: set_index infer fails on non-constant blocks ===
    std::cout << "\n=== Test 52: set_index invalid mixed blocks ===" << std::endl;
    {
        DataFrame df;
        // freq run lengths are inconsistent (2 then 1 then 1), not a valid grouped layout.
        df.add_column<int>("bias", {1,1,2,2});
        df.add_column<int>("freq", {10,10,20,10});
        df.add_column<double>("v", {1,2,3,4});
        bool caught = false;
        try { df.set_index({"bias", "freq"}); }
        catch (const std::invalid_argument&) { caught = true; }
        assert(caught);
    }
    std::cout << "PASSED" << std::endl;

    // === Test 53: reset then set_index infer mixed ===
    std::cout << "\n=== Test 53: reset then set_index infer mixed ===" << std::endl;
    {
        DataFrame df;
        df.add_uniform_index<int>("bias", {1, 2});
        df.add_grouped_index<double>("freq", {1.0, 2.0, 3.0, 1.5, 2.5, 3.5}, 3);
        df.add_uniform_index<std::string>("port", {"S11", "S21"});
        df.add_column<double>("S", {-10,-11,-20,-21,-30,-31,-15,-16,-25,-26,-35,-36});

        df.reset_index();
        df.set_index({"bias", "freq", "port"});
        auto d1 = df.get_index_dim("freq");
        assert(d1.is_grouped() && d1.is_regular_grouped());
        assert(d1.group_lengths.size() == 6 && d1.group_lengths[0] == 2);
    }
    std::cout << "PASSED" << std::endl;

    // === Test 54: set_index with 1/3/4 columns, keep 2nd as dependent ===
    std::cout << "\n=== Test 54: set_index selected columns layout ===" << std::endl;
    {
        DataFrame df;
        // Add all as regular columns first (order: 1=bias, 2=val, 3=freq, 4=port)
        df.add_column<int>("bias", {2, 1, 2, 1, 1, 2, 1, 2});
        df.add_column<int>("val",  {222,111,211,122,112,221,121,212});
        df.add_column<int>("freq", {20, 10, 10, 20, 10, 20, 20, 10});
        df.add_column<std::string>("port", {"S21","S11","S11","S21","S21","S11","S11","S21"});

        // Promote columns 1/3/4 to index; keep column 2 as dependent
        df.set_index({"bias", "freq", "port"});

        assert(df.num_indices() == 3);
        assert(df.dependent_names() == (std::vector<std::string>{"val"}));

        // Index columns should be placed first in the requested order.
        auto names = df.column_names();
        assert(names[0] == "bias");
        assert(names[1] == "freq");
        assert(names[2] == "port");
        assert(names[3] == "val");

        // First-appearance levels are bias=[2,1], freq=[20,10], port=[S21,S11],
        // so dependent values should be reordered into this Cartesian layout.
        auto v = df.get_column_as<int>("val");
        assert(v[0] == 222);
        assert(v[1] == 221);
        assert(v[2] == 212);
        assert(v[3] == 211);
        assert(v[4] == 122);
        assert(v[5] == 121);
        assert(v[6] == 112);
        assert(v[7] == 111);
    }
    std::cout << "PASSED" << std::endl;

    // === Test 55: loc with -1 wildcard ===
    std::cout << "\n=== Test 55: loc wildcard (-1) ===" << std::endl;
    {
        DataFrame df;
        df.add_uniform_index<int>("a", {1, 2});
        df.add_uniform_index<int>("b", {10, 20, 30});
        df.add_column<double>("v", {1, 2, 3, 4, 5, 6});

        // loc(-1): wildcard on last dim → keep all b values, column b visible
        auto s1 = df.loc({-1});
        assert(s1->num_rows() == 6);  // all rows kept
        assert(s1->num_indices() == 2); // both a and b remain
        assert(s1->column_names().size() == 3); // a, b, v all present

        // loc(0, -1): fix a=1, wildcard on b
        auto s2 = df.loc({0, -1});
        assert(s2->num_rows() == 3);  // a=1: rows 0,1,2
        assert(s2->num_indices() == 1); // only b remains
        auto v2 = s2->get_column_as<double>("v");
        assert(v2[0] == 1.0);
        assert(v2[1] == 2.0);
        assert(v2[2] == 3.0);
        // b column should still be present
        auto b2 = s2->get_column_as<int>("b");
        assert(b2[0] == 10);
        assert(b2[1] == 20);
        assert(b2[2] == 30);

        // loc(-1, 0): wildcard on a, fix b=10
        auto s3 = df.loc({-1, 0});
        assert(s3->num_rows() == 2);  // b=10: rows 0,3
        assert(s3->num_indices() == 1); // only a remains
        auto v3 = s3->get_column_as<double>("v");
        assert(v3[0] == 1.0);
        assert(v3[1] == 4.0);
        // a column should be present, b should be excluded
        auto a3 = s3->get_column_as<int>("a");
        assert(a3[0] == 1);
        assert(a3[1] == 2);
        assert(s3->column_names().size() == 2); // a, v (no b)
    }
    std::cout << "PASSED" << std::endl;

    // === Test 56: loc wildcard with 3 dimensions ===
    std::cout << "\n=== Test 56: loc wildcard 3 dims ===" << std::endl;
    {
        DataFrame df;
        df.add_uniform_index<std::string>("x", {"A", "B"});
        df.add_uniform_index<int>("y", {1, 2});
        df.add_uniform_index<int>("z", {10, 20});
        // 2*2*2 = 8 rows
        df.add_column<int>("val", {1,2,3,4,5,6,7,8});

        // loc(-1, -1): wildcard on y and z → same as loc on x only is impossible
        // Actually: right-aligned, so -1 maps to y, -1 maps to z
        // All y and z values kept, columns y and z visible
        auto s1 = df.loc({-1, -1});
        assert(s1->num_rows() == 8);
        assert(s1->num_indices() == 3); // x, y, z all remain
        assert(s1->column_names().size() == 4); // x, y, z, val

        // loc(0, -1, 1): x=A(0), wildcard on y, z=20(1)
        auto s2 = df.loc({0, -1, 1});
        assert(s2->num_rows() == 2); // x=A, z=20: rows (A,1,20)=2, (A,2,20)=4
        assert(s2->num_indices() == 1); // only y remains
        auto v2 = s2->get_column_as<int>("val");
        assert(v2[0] == 2);
        assert(v2[1] == 4);
        // y column visible, x and z excluded
        assert(s2->column_names().size() == 2); // y, val

        // loc(-1, 0, -1): wildcard x, y=1(0), wildcard z
        auto s3 = df.loc({-1, 0, -1});
        assert(s3->num_rows() == 4); // y=1: (A,1,10),(A,1,20),(B,1,10),(B,1,20)
        assert(s3->num_indices() == 2); // x and z remain
        auto v3 = s3->get_column_as<int>("val");
        assert(v3[0] == 1);
        assert(v3[1] == 2);
        assert(v3[2] == 5);
        assert(v3[3] == 6);
    }
    std::cout << "PASSED" << std::endl;

    // === Test 57: add_grouped_index_groups basic ===
    std::cout << "\n=== Test 57: add_grouped_index_groups basic ===" << std::endl;
    {
        // level(uniform 2) x number(ragged): level 0 has 3 contours, level 1 has 2 contours
        DataFrame df;
        df.add_uniform_index<int>("level", {0, 1});
        df.add_grouped_index_groups<int>("number", {{0,1,2}, {0,1}});
        // Total rows = 3 + 2 = 5
        assert(df.num_rows() == 5);
        assert(df.num_indices() == 2);
        assert(df.get_index_dim("level").is_uniform());
        assert(df.get_index_dim("number").is_grouped());
        assert(df.get_index_dim("number").group_lengths.size() == 2);
        assert(df.get_index_dim("number").group_lengths[0] == 3);
        assert(df.get_index_dim("number").group_lengths[1] == 2);
        // level column: [0,0,0,1,1]
        auto lv = df.get_column_as<int>("level");
        assert(lv[0]==0 && lv[1]==0 && lv[2]==0 && lv[3]==1 && lv[4]==1);
        // number column: [0,1,2,0,1]
        auto nv = df.get_column_as<int>("number");
        assert(nv[0]==0 && nv[1]==1 && nv[2]==2 && nv[3]==0 && nv[4]==1);
    }
    std::cout << "PASSED" << std::endl;

    // === Test 58: loc on ragged dimension ===
    std::cout << "\n=== Test 58: loc on ragged dimension ===" << std::endl;
    {
        // level(uniform 2) x number(ragged 3,2) x point(ragged: 2,3,1, 4,2)
        DataFrame df;
        df.add_uniform_index<int>("level", {0, 1});
        df.add_grouped_index_groups<int>("number", {{0,1,2},{0,1}});
        // Points per contour: contour(L0,N0)=2, (L0,N1)=3, (L0,N2)=1, (L1,N0)=4, (L1,N1)=2
        df.add_grouped_index_groups<int>("point", {{0,1},{0,1,2},{0},{0,1,2,3},{0,1}});
        // Total rows = 2+3+1+4+2 = 12
        assert(df.num_rows() == 12);

        // Add a data column
        std::vector<double> vals(12);
        for (int i = 0; i < 12; ++i) vals[i] = i * 1.0;
        df.add_column<double>("x", vals);

        // loc(0): fix point=0 → every contour has pt0: 5 contours → 5 rows
        auto s0 = df.loc({0});
        assert(s0->num_rows() == 5);
        assert(s0->num_indices() == 2); // level + number remain

        // loc(1): fix point=1 → contours with ≥2 pts: (L0,N0)✓3pt,(L0,N1)✓,(L1,N0)✓,(L1,N1)✓; (L0,N2)✗1pt → 4 rows
        auto s1 = df.loc({1});
        assert(s1->num_rows() == 4);

        // loc(2): fix point=2 → contours with ≥3 pts: (L0,N1)✓3; (L1,N0)✓4 → 2 rows
        auto s2 = df.loc({2});
        assert(s2->num_rows() == 2);

        // loc(0, 0): fix number=0, point=0 → (L0,N0) pt0 + (L1,N0) pt0 → 2 rows
        auto s00 = df.loc({0, 0});
        assert(s00->num_rows() == 2);
        assert(s00->num_indices() == 1); // only level remains

        // loc(2, 0): number=2, point=0 → only (L0,N2) pt0, L1 has no N2 → 1 row
        auto s20 = df.loc({2, 0});
        assert(s20->num_rows() == 1);
    }
    std::cout << "PASSED" << std::endl;

    // === Test 59: set_index rejects ragged; manual construction works ===
    std::cout << "\n=== Test 59: set_index rejects ragged ===" << std::endl;
    {
        // level=[0,0,0,1,1], number=[0,1,2,0,1]: group0 has 3 elements, group1 has 2 → ragged
        // set_index should throw; users must use add_grouped_index_groups for ragged
        DataFrame df;
        df.add_column<int>("level",  {0,0,0,1,1});
        df.add_column<int>("number", {0,1,2,0,1});
        df.add_column<double>("val", {10,20,30,40,50});
        bool caught = false;
        try { df.set_index({"level","number"}); }
        catch (const std::invalid_argument&) { caught = true; }
        assert(caught);

        // Manual construction of the ragged index
        DataFrame df2;
        df2.add_uniform_index<int>("level", {0, 1});
        df2.add_grouped_index_groups<int>("number", {{0,1,2},{0,1}});
        df2.add_column<double>("val", {10,20,30,40,50});
        assert(df2.num_indices() == 2);
        assert(df2.get_index_dim("level").is_uniform());
        assert(df2.get_index_dim("number").is_grouped());
        // loc({1}): number=1 in each level group
        // level=0 group: [0,1,2] → position 1 = row1 (val=20)
        // level=1 group: [0,1]   → position 1 = row4 (val=50)
        auto s = df2.loc({1});
        assert(s->num_rows() == 2);
        auto vv = s->get_column_as<double>("val");
        assert(vv[0] == 20.0 && vv[1] == 50.0);
        // loc({2}): only level=0 group has number=2 (row 2, val=30)
        auto s2 = df2.loc({2});
        assert(s2->num_rows() == 1);
        assert(s2->get_column_as<double>("val")[0] == 30.0);
        std::cout << df2.to_string() << std::endl;
    }
    std::cout << "PASSED" << std::endl;

    // === Test 60: set_index rejects ragged (same_lens failure) and accepts regular grouped ===
    std::cout << "\n=== Test 60: set_index ragged(same_lens) + regular grouped ===" << std::endl;
    {
        // Negative: level=[0,0,0,1,1], number=[1,1,1,2,2]
        // group0 has 1 run of length 3; group1 has 1 run of length 2
        // same_count=true, same_lens=false → ragged → throw
        DataFrame df_neg;
        df_neg.add_column<int>("level",  {0,0,0,1,1});
        df_neg.add_column<int>("number", {1,1,1,2,2});
        df_neg.add_column<double>("val", {10,20,30,40,50});
        bool caught = false;
        try { df_neg.set_index({"level","number"}); }
        catch (const std::invalid_argument&) { caught = true; }
        assert(caught);

        // Positive: level=[0,0,1,1], number=[1,2,3,4]
        // group0=[1,2] → runs[(0,1),(1,1)]; group1=[3,4] → runs[(2,1),(3,1)]
        // same_count=true(2 each), same_lens=true(all 1), same_vals=false → Regular Grouped
        DataFrame df_pos;
        df_pos.add_column<int>("level",  {0,0,1,1});
        df_pos.add_column<int>("number", {1,2,3,4});
        df_pos.add_column<double>("val", {10,20,30,40});
        df_pos.set_index({"level","number"});
        assert(df_pos.num_indices() == 2);
        assert(df_pos.get_index_dim("level").is_uniform());
        assert(df_pos.get_index_dim("number").is_grouped());
        assert(df_pos.get_index_dim("number").is_regular_grouped());
        std::cout << df_pos.to_string() << std::endl;
    }
    std::cout << "PASSED" << std::endl;

    // === Test 56: flat_index / multi_index with all-Uniform ==
    std::cout << "\n=== Test 56: flat_index/multi_index Uniform ===" << std::endl;
    {
        // bias(2) x freq(3): 6 rows, row-major
        DataFrame df;
        df.add_uniform_index<int>("bias", {1, 2});
        df.add_uniform_index<int>("freq", {10, 20, 30});
        df.add_column<int>("v", {0,1,2,3,4,5});
        // flat_index
        assert(df.flat_index({0, 0}) == 0);
        assert(df.flat_index({0, 2}) == 2);
        assert(df.flat_index({1, 0}) == 3);
        assert(df.flat_index({1, 2}) == 5);
        // multi_index
        assert((df.multi_index(0) == std::vector<std::size_t>{0,0}));
        assert((df.multi_index(2) == std::vector<std::size_t>{0,2}));
        assert((df.multi_index(3) == std::vector<std::size_t>{1,0}));
        assert((df.multi_index(5) == std::vector<std::size_t>{1,2}));
        // round-trip
        for (std::size_t r = 0; r < 6; ++r)
            assert(df.flat_index(df.multi_index(r)) == r);
    }
    std::cout << "PASSED" << std::endl;

    // === Test 57: flat_index / multi_index with Grouped dim ===
    std::cout << "\n=== Test 57: flat_index/multi_index Grouped ===" << std::endl;
    {
        // level(Uniform,2) x number(Grouped): level=0 → [0,1,2], level=1 → [0,1]
        // rows: (L0,N0),(L0,N1),(L0,N2),(L1,N0),(L1,N1)  →  0,1,2,3,4
        DataFrame df;
        df.add_uniform_index<int>("level",  {0, 1});
        df.add_grouped_index_groups<int>("number", {{0,1,2},{0,1}});

        // flat_index
        assert(df.flat_index({0, 0}) == 0);
        assert(df.flat_index({0, 1}) == 1);
        assert(df.flat_index({0, 2}) == 2);
        assert(df.flat_index({1, 0}) == 3);
        assert(df.flat_index({1, 1}) == 4);
        // level=1 only has 2 numbers → index 2 should throw
        bool caught = false;
        try { df.flat_index({1, 2}); } catch (const std::out_of_range&) { caught = true; }
        assert(caught);
        // multi_index
        assert((df.multi_index(0) == std::vector<std::size_t>{0,0}));
        assert((df.multi_index(2) == std::vector<std::size_t>{0,2}));
        assert((df.multi_index(3) == std::vector<std::size_t>{1,0}));
        assert((df.multi_index(4) == std::vector<std::size_t>{1,1}));
        // round-trip
        for (std::size_t r = 0; r < 5; ++r)
            assert(df.flat_index(df.multi_index(r)) == r);
    }
    std::cout << "PASSED" << std::endl;

    // === Test 58: flat_index / multi_index with mixed Uniform+Grouped ===
    std::cout << "\n=== Test 58: flat_index/multi_index mixed ===" << std::endl;
    {
        // bias(Uniform,2) x freq(Grouped,uniform_grouped 3/group) x port(Uniform,2)
        // rows: bias=1 → freq=1.0,2.0,3.0 each with port=S11,S21 → 6
        //       bias=2 → freq=1.5,2.5,3.5 each with port=S11,S21 → 6
        // total 12 rows
        DataFrame df;
        df.add_uniform_index<int>("bias", {1, 2});
        df.add_grouped_index<double>("freq", {1.0,2.0,3.0,1.5,2.5,3.5}, 3);
        df.add_uniform_index<std::string>("port", {"S11","S21"});
        df.add_column<int>("v", {0,1,2,3,4,5,6,7,8,9,10,11});
        assert(df.num_rows() == 12);

        // flat_index: (bias=0,freq=0,port=0) → row 0
        assert(df.flat_index({0,0,0}) == 0);
        assert(df.flat_index({0,0,1}) == 1);
        assert(df.flat_index({0,1,0}) == 2);
        assert(df.flat_index({0,2,1}) == 5);
        assert(df.flat_index({1,0,0}) == 6);
        assert(df.flat_index({1,2,1}) == 11);

        // multi_index
        assert((df.multi_index(0)  == std::vector<std::size_t>{0,0,0}));
        assert((df.multi_index(5)  == std::vector<std::size_t>{0,2,1}));
        assert((df.multi_index(6)  == std::vector<std::size_t>{1,0,0}));
        assert((df.multi_index(11) == std::vector<std::size_t>{1,2,1}));

        // round-trip for all rows
        for (std::size_t r = 0; r < 12; ++r)
            assert(df.flat_index(df.multi_index(r)) == r);
    }
    std::cout << "PASSED" << std::endl;

    std::cout << "\n=== Test A1: Arithmetic operators — double columns ===" << std::endl;
    {
        DataFrame df1, df2;
        df1.add_column<double>("a", {1.0, 2.0, 3.0});
        df1.add_column<double>("v", {10.0, 20.0, 30.0});
        df2.add_column<double>("a", {1.0, 2.0, 3.0});
        df2.add_column<double>("v", {1.0, 2.0, 3.0});

        // df + df
        auto s = df1 + df2;
        assert(approx_equal(s->at<double>("v", 0), 11.0));
        assert(approx_equal(s->at<double>("v", 1), 22.0));
        assert(approx_equal(s->at<double>("v", 2), 33.0));
        // original unchanged
        assert(approx_equal(df1.at<double>("v", 0), 10.0));

        // df - df
        auto d = df1 - df2;
        assert(approx_equal(d->at<double>("v", 0), 9.0));
        assert(approx_equal(d->at<double>("v", 2), 27.0));

        // df * df
        auto m = df1 * df2;
        assert(approx_equal(m->at<double>("v", 0), 10.0));
        assert(approx_equal(m->at<double>("v", 1), 40.0));

        // df / df
        auto q = df1 / df2;
        assert(approx_equal(q->at<double>("v", 0), 10.0));
        assert(approx_equal(q->at<double>("v", 2), 10.0));

        // scalar ops: df op scalar
        auto sp = df1 + 5.0;
        assert(approx_equal(sp->at<double>("v", 0), 15.0));

        auto sm = df1 - 5.0;
        assert(approx_equal(sm->at<double>("v", 0), 5.0));

        auto sc = df1 * 2.0;
        assert(approx_equal(sc->at<double>("v", 0), 20.0));
        assert(approx_equal(sc->at<double>("v", 2), 60.0));

        auto sd = df1 / 2.0;
        assert(approx_equal(sd->at<double>("v", 0), 5.0));

        // scalar on left
        auto rs = 100.0 - df1;
        assert(approx_equal(rs->at<double>("v", 0), 90.0));
        assert(approx_equal(rs->at<double>("v", 2), 70.0));

        auto rm = 3.0 * df1;
        assert(approx_equal(rm->at<double>("v", 0), 30.0));

        auto ra = 1.0 + df1;
        assert(approx_equal(ra->at<double>("v", 1), 21.0));

        auto rd = 120.0 / df1;
        assert(approx_equal(rd->at<double>("v", 0), 12.0));
        assert(approx_equal(rd->at<double>("v", 2), 4.0));
    }
    std::cout << "PASSED" << std::endl;

    std::cout << "\n=== Test A2: Arithmetic operators — int column with scalar → double ===" << std::endl;
    {
        DataFrame df;
        df.add_column<int>("x", {2, 4, 6});
        df.add_column<int>("v", {10, 20, 30});

        // int * double scalar → double column (promoted)
        auto sc = df * 2.0;
        assert(sc->column_dtype("v") == "double");
        assert(approx_equal(sc->at<double>("v", 0), 20.0));
        assert(approx_equal(sc->at<double>("v", 2), 60.0));

        auto sd = df / 4.0;
        assert(sd->column_dtype("v") == "double");
        assert(approx_equal(sd->at<double>("v", 0), 2.5));

        auto sp = df + 0.5;
        assert(sp->column_dtype("v") == "double");
        assert(approx_equal(sp->at<double>("v", 1), 20.5));

        // scalar - int_df → double
        auto rs = 100.0 - df;
        assert(rs->column_dtype("v") == "double");
        assert(approx_equal(rs->at<double>("v", 0), 90.0));
    }
    std::cout << "PASSED" << std::endl;

    std::cout << "\n=== Test A3: Arithmetic operators — complex column ===" << std::endl;
    {
        using C = std::complex<double>;
        DataFrame df1, df2;
        df1.add_column<C>("v", {C(1,2), C(3,4)});
        df2.add_column<C>("v", {C(1,0), C(0,1)});

        auto s = df1 + df2;
        assert(approx_equal(s->at<C>("v", 0), C(2,2)));
        assert(approx_equal(s->at<C>("v", 1), C(3,5)));

        auto sc = df1 * 2.0;
        assert(approx_equal(sc->at<C>("v", 0), C(2,4)));
    }
    std::cout << "PASSED" << std::endl;

    std::cout << "\n=== Test A5: Arithmetic — mixed types (int+double, int+complex, double+complex) ===" << std::endl;
    {
        using C = std::complex<double>;

        // int + double → double
        DataFrame di, dd;
        di.add_column<int>("v", {1, 2, 3});
        dd.add_column<double>("v", {0.5, 1.5, 2.5});
        auto r1 = di + dd;
        assert(r1->column_dtype("v") == "double");
        assert(approx_equal(r1->at<double>("v", 0), 1.5));
        assert(approx_equal(r1->at<double>("v", 2), 5.5));

        // double + int → double (order reversed)
        auto r2 = dd + di;
        assert(r2->column_dtype("v") == "double");
        assert(approx_equal(r2->at<double>("v", 1), 3.5));

        // int + complex → complex
        DataFrame dc;
        dc.add_column<C>("v", {C(0,1), C(0,2), C(0,3)});
        auto r3 = di + dc;
        assert(r3->column_dtype("v") == "complex");
        assert(approx_equal(r3->at<C>("v", 0), C(1,1)));
        assert(approx_equal(r3->at<C>("v", 2), C(3,3)));

        // double + complex → complex
        auto r4 = dd + dc;
        assert(r4->column_dtype("v") == "complex");
        assert(approx_equal(r4->at<C>("v", 0), C(0.5,1)));

        // int * double_df → double
        auto r5 = di * dd;
        assert(r5->column_dtype("v") == "double");
        assert(approx_equal(r5->at<double>("v", 1), 3.0));

        // string column throws
        DataFrame ds;
        ds.add_column<std::string>("v", {"a", "b", "c"});
        bool caught = false;
        try { auto bad = di + ds; } catch (const std::invalid_argument&) { caught = true; }
        assert(caught);
    }
    std::cout << "PASSED" << std::endl;

    std::cout << "\n=== Test A4: Arithmetic operators — with multi-index ===" << std::endl;
    {
        DataFrame df1, df2;
        df1.add_uniform_index<int>("f", {1, 2, 3});
        df1.add_column<double>("s11", {-10.0, -20.0, -30.0});
        df2.add_uniform_index<int>("f", {1, 2, 3});
        df2.add_column<double>("s11", {1.0, 2.0, 3.0});

        auto r = df1 + df2;
        assert(approx_equal(r->at<double>("s11", 0), -9.0));
        assert(approx_equal(r->at<double>("s11", 2), -27.0));
        // index dims preserved
        assert(r->num_indices() == 1);
        assert(r->is_index("f"));
    }
    std::cout << "PASSED" << std::endl;

    std::cout << "\n=== ALL TESTS PASSED ===" << std::endl;
    return 0;
}
