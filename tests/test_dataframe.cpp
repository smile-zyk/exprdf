#include "exprdf/exprdf.hpp"

#include <cassert>
#include <functional>
#include <iostream>
#include <memory>
#include <stdexcept>

int main() {
    using exprdf::DataFrame;
    using exprdf::Column;

    auto make_col_ptrs = [](std::initializer_list<Column> cols) {
        std::vector<std::unique_ptr<Column>> out;
        out.reserve(cols.size());
        for (const auto& c : cols)
            out.emplace_back(new Column(c));
        return out;
    };

    // 1) sub() should capture a live parent when parent is shared-owned.
    std::shared_ptr<DataFrame> parent = std::make_shared<DataFrame>();
    parent->add_column<int>("x", std::vector<int>{1, 2, 3});
    std::shared_ptr<DataFrame> child = parent->sub("x");

    assert(child->has_parent());
    std::shared_ptr<const DataFrame> locked_parent = child->parent();
    assert(static_cast<bool>(locked_parent));
    assert(locked_parent.get() == parent.get());
    locked_parent.reset();

    // 2) After parent destruction, weak parent in child should expire safely.
    parent.reset();
    assert(!child->has_parent());
    assert(!child->parent());

    // 3) For non-shared (stack) parent, sub() should not throw and parent remains empty.
    DataFrame stack_df;
    stack_df.add_column<int>("y", std::vector<int>{7, 8});
    std::shared_ptr<DataFrame> stack_child = stack_df.sub("y");
    assert(!stack_child->has_parent());
    assert(!stack_child->parent());

    // 4) Keep Column-reference APIs compatible after DataFrame internal unique_ptr migration.
    DataFrame df_ref;
    df_ref.add_column<int>("a", std::vector<int>{1, 2, 3});
    df_ref.add_column<double>("b", std::vector<double>{10.0, 20.0, 30.0});

    // const last_column() returns a stable const reference.
    const DataFrame& cdf_ref = df_ref;
    const auto& last_const = cdf_ref.last_column();
    assert(last_const.tag == exprdf::DType::Double);
    assert(last_const.as<double>().size() == 3);
    assert(last_const.as<double>()[1] == 20.0);

    // mutable last_column() returns a writable reference.
    auto& last_mut = df_ref.last_column();
    last_mut.as<double>()[1] = 25.0;
    assert(df_ref.at<double>("b", 1) == 25.0);

    // get_column(index) keeps reference semantics.
    const auto& col_a_const = cdf_ref.get_column(static_cast<std::size_t>(0));
    assert(col_a_const.tag == exprdf::DType::Int);
    auto& col_a_mut = df_ref.get_column(static_cast<std::size_t>(0));
    col_a_mut.as<int>()[2] = 42;
    assert(df_ref.at<int>("a", 2) == 42);

    // get_column(name) keeps reference semantics.
    const auto& col_b_const = cdf_ref.get_column("b");
    assert(col_b_const.as<double>()[0] == 10.0);
    auto& col_b_mut = df_ref.get_column("b");
    col_b_mut.as<double>()[0] = 11.0;
    assert(df_ref.at<double>("b", 0) == 11.0);

    // 5) DataFrame deep copy compatibility (copy ctor / copy assignment).
    DataFrame copied_ctor(df_ref);
    copied_ctor.set<double>("b", 0, -1.0);
    assert(df_ref.at<double>("b", 0) == 11.0);
    assert(copied_ctor.at<double>("b", 0) == -1.0);

    DataFrame copied_assign;
    copied_assign = df_ref;
    copied_assign.set<int>("a", 0, -7);
    assert(df_ref.at<int>("a", 0) == 1);
    assert(copied_assign.at<int>("a", 0) == -7);

    // 6) add_uniform_index_column: scalar-column input path and expansion behavior.
    DataFrame df_ui;
    auto f_levels = Column::from_scalar<double>({1.0, 2.0, 3.0});
    f_levels->quantity = "Hz";
    df_ui.add_uniform_index_column("f", std::move(f_levels));
    assert(df_ui.num_indices() == 1);
    assert(df_ui.num_rows() == 3);
    assert(df_ui.at<double>("f", 0) == 1.0);
    assert(df_ui.at<double>("f", 1) == 2.0);
    assert(df_ui.at<double>("f", 2) == 3.0);
    assert(df_ui.column_quantity("f") == "Hz");

    auto t_levels = Column::from_scalar<int>({10, 20});
    df_ui.add_uniform_index_column("t", std::move(t_levels), "C");
    assert(df_ui.num_indices() == 2);
    assert(df_ui.num_rows() == 6);
    assert(df_ui.at<double>("f", 0) == 1.0);
    assert(df_ui.at<double>("f", 1) == 1.0);
    assert(df_ui.at<double>("f", 2) == 2.0);
    assert(df_ui.at<double>("f", 3) == 2.0);
    assert(df_ui.at<double>("f", 4) == 3.0);
    assert(df_ui.at<double>("f", 5) == 3.0);
    assert(df_ui.at<int>("t", 0) == 10);
    assert(df_ui.at<int>("t", 1) == 20);
    assert(df_ui.at<int>("t", 2) == 10);
    assert(df_ui.at<int>("t", 3) == 20);
    assert(df_ui.at<int>("t", 4) == 10);
    assert(df_ui.at<int>("t", 5) == 20);
    assert(df_ui.column_quantity("t") == "C");

    {
        DataFrame df_bad;
        bool thrown = false;
        try {
            std::unique_ptr<Column> null_levels;
            df_bad.add_uniform_index_column("u", std::move(null_levels));
        } catch (const std::invalid_argument&) {
            thrown = true;
        }
        assert(thrown);
    }

    {
        DataFrame df_bad;
        bool thrown = false;
        try {
            auto list_levels = Column::from_list_flat<int>({1, 2, 3, 4}, 2);
            df_bad.add_uniform_index_column("u", std::move(list_levels));
        } catch (const std::invalid_argument&) {
            thrown = true;
        }
        assert(thrown);
    }

    // 7) grouped index APIs: allow grouped as first index dimension.
    DataFrame df_g_first;
    df_g_first.add_grouped_index<int>("g", std::vector<int>{10, 20, 30}, 3, "u");
    assert(df_g_first.num_indices() == 1);
    assert(df_g_first.num_rows() == 3);
    assert(df_g_first.at<int>("g", 0) == 10);
    assert(df_g_first.at<int>("g", 1) == 20);
    assert(df_g_first.at<int>("g", 2) == 30);
    assert(df_g_first.column_quantity("g") == "u");

    DataFrame df_g_first_groups;
    df_g_first_groups.add_grouped_index_groups<int>("gg", std::vector<std::vector<int>>{{7, 8, 9}});
    assert(df_g_first_groups.num_indices() == 1);
    assert(df_g_first_groups.num_rows() == 3);
    assert(df_g_first_groups.at<int>("gg", 0) == 7);
    assert(df_g_first_groups.at<int>("gg", 1) == 8);
    assert(df_g_first_groups.at<int>("gg", 2) == 9);

    {
        DataFrame df_bad;
        bool thrown = false;
        try {
            df_bad.add_grouped_index<int>("g", std::vector<int>{1, 2}, 3);
        } catch (const std::invalid_argument&) {
            thrown = true;
        }
        assert(thrown);
    }

    {
        DataFrame df_bad;
        bool thrown = false;
        try {
            df_bad.add_grouped_index_groups<int>("g", std::vector<std::vector<int>>{{1}, {2}});
        } catch (const std::invalid_argument&) {
            thrown = true;
        }
        assert(thrown);
    }

    // 8) Column combine helpers: scalar -> list
    Column s1 = *Column::from_scalar<int>({1, 2});
    Column s2 = *Column::from_scalar<int>({10, 20});
    Column s3 = *Column::from_scalar<int>({100, 200});
    Column list_col = *Column::combine_scalars_to_list(make_col_ptrs({s1, s2, s3}));
    assert(list_col.shape.size() == 1);
    assert(list_col.shape[0] == 3);
    assert(list_col.num_rows() == 2);
    assert(list_col.get<int>(0, 1) == 1);
    assert(list_col.get<int>(0, 2) == 10);
    assert(list_col.get<int>(0, 3) == 100);
    assert(list_col.get<int>(1, 1) == 2);
    assert(list_col.get<int>(1, 2) == 20);
    assert(list_col.get<int>(1, 3) == 200);

    // 9) Column combine helpers: list -> matrix
    Column l1 = *Column::from_list_flat<int>({1, 2, 3, 4}, 2);
    Column l2 = *Column::from_list_flat<int>({5, 6, 7, 8}, 2);
    Column matrix_from_lists = *Column::combine_lists_to_matrix(make_col_ptrs({l1, l2}));
    assert(matrix_from_lists.shape.size() == 2);
    assert(matrix_from_lists.shape[0] == 2);
    assert(matrix_from_lists.shape[1] == 2);
    assert(matrix_from_lists.num_rows() == 2);
    assert(matrix_from_lists.get<int>(0, 1, 1) == 1);
    assert(matrix_from_lists.get<int>(0, 1, 2) == 2);
    assert(matrix_from_lists.get<int>(0, 2, 1) == 5);
    assert(matrix_from_lists.get<int>(0, 2, 2) == 6);
    assert(matrix_from_lists.get<int>(1, 1, 1) == 3);
    assert(matrix_from_lists.get<int>(1, 1, 2) == 4);
    assert(matrix_from_lists.get<int>(1, 2, 1) == 7);
    assert(matrix_from_lists.get<int>(1, 2, 2) == 8);

    // 10) Column combine helpers: scalar -> matrix (with explicit shape)
    Column m11 = *Column::from_scalar<int>({1, 2});
    Column m12 = *Column::from_scalar<int>({3, 4});
    Column m21 = *Column::from_scalar<int>({5, 6});
    Column m22 = *Column::from_scalar<int>({7, 8});
    Column matrix_from_scalars = *Column::combine_scalars_to_matrix(make_col_ptrs({m11, m12, m21, m22}), 2, 2);
    assert(matrix_from_scalars.shape.size() == 2);
    assert(matrix_from_scalars.shape[0] == 2);
    assert(matrix_from_scalars.shape[1] == 2);
    assert(matrix_from_scalars.num_rows() == 2);
    assert(matrix_from_scalars.get<int>(0, 1, 1) == 1);
    assert(matrix_from_scalars.get<int>(0, 1, 2) == 3);
    assert(matrix_from_scalars.get<int>(0, 2, 1) == 5);
    assert(matrix_from_scalars.get<int>(0, 2, 2) == 7);
    assert(matrix_from_scalars.get<int>(1, 1, 1) == 2);
    assert(matrix_from_scalars.get<int>(1, 1, 2) == 4);
    assert(matrix_from_scalars.get<int>(1, 2, 1) == 6);
    assert(matrix_from_scalars.get<int>(1, 2, 2) == 8);

    // 11) Column combine helpers: negative paths should throw invalid_argument.
    auto expect_invalid_argument = [](const std::function<void()>& fn) {
        bool thrown = false;
        try {
            fn();
        } catch (const std::invalid_argument&) {
            thrown = true;
        }
        assert(thrown);
    };

    // scalar -> list: dtype mismatch / non-scalar input / row mismatch
    expect_invalid_argument([&]() {
        Column::combine_scalars_to_list(
            make_col_ptrs({*Column::from_scalar<int>({1, 2}), *Column::from_scalar<double>({1.0, 2.0})}));
    });
    expect_invalid_argument([&]() {
        Column::combine_scalars_to_list(
            make_col_ptrs({*Column::from_scalar<int>({1, 2}), *Column::from_list_flat<int>({3, 4, 5, 6}, 2)}));
    });
    expect_invalid_argument([&]() {
        Column::combine_scalars_to_list(
            make_col_ptrs({*Column::from_scalar<int>({1, 2}), *Column::from_scalar<int>({3, 4, 5})}));
    });

    // list -> matrix: non-list input / list-size mismatch / row mismatch
    expect_invalid_argument([&]() {
        Column::combine_lists_to_matrix(
            make_col_ptrs({*Column::from_list_flat<int>({1, 2, 3, 4}, 2), *Column::from_scalar<int>({5, 6})}));
    });
    expect_invalid_argument([&]() {
        Column::combine_lists_to_matrix(
            make_col_ptrs({*Column::from_list_flat<int>({1, 2, 3, 4}, 2), *Column::from_list_flat<int>({5, 6, 7, 8, 9, 10}, 3)}));
    });
    expect_invalid_argument([&]() {
        Column::combine_lists_to_matrix(
            make_col_ptrs({*Column::from_list_flat<int>({1, 2, 3, 4}, 2), *Column::from_list_flat<int>({5, 6, 7, 8, 9, 10, 11, 12}, 2)}));
    });

    // scalar -> matrix: dimension mismatch / non-scalar input / dtype mismatch / row mismatch
    expect_invalid_argument([&]() {
        Column::combine_scalars_to_matrix(
            make_col_ptrs({*Column::from_scalar<int>({1, 2}), *Column::from_scalar<int>({3, 4}), *Column::from_scalar<int>({5, 6})}),
            2, 2);
    });
    expect_invalid_argument([&]() {
        Column::combine_scalars_to_matrix(
            make_col_ptrs({*Column::from_scalar<int>({1, 2}), *Column::from_list_flat<int>({3, 4, 5, 6}, 2),
             *Column::from_scalar<int>({7, 8}), *Column::from_scalar<int>({9, 10})}),
            2, 2);
    });
    expect_invalid_argument([&]() {
        Column::combine_scalars_to_matrix(
            make_col_ptrs({*Column::from_scalar<int>({1, 2}), *Column::from_scalar<double>({3.0, 4.0}),
             *Column::from_scalar<int>({5, 6}), *Column::from_scalar<int>({7, 8})}),
            2, 2);
    });
    expect_invalid_argument([&]() {
        Column::combine_scalars_to_matrix(
            make_col_ptrs({*Column::from_scalar<int>({1, 2}), *Column::from_scalar<int>({3, 4, 5}),
             *Column::from_scalar<int>({6, 7}), *Column::from_scalar<int>({8, 9})}),
            2, 2);
    });

    std::cout << "test_dataframe: all checks passed\n";
    return 0;
}
