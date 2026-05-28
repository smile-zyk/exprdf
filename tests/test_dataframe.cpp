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

    std::cout << "test_dataframe: all checks passed\n";
    return 0;
}
