#include "exprdf/exprdf.hpp"

#include <cassert>
#include <iostream>
#include <memory>

int main() {
    using exprdf::DataFrame;

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

    std::cout << "test_dataframe: all checks passed\n";
    return 0;
}
