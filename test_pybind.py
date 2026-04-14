import sys
sys.path.insert(0, r"c:\dev\cpp_cmake_template\build\Debug")

import exprdf as pdf

print("=== Python Test 1: Create DataFrame & add columns ===")
df = pdf.DataFrame()
df.add_column("id", [1, 2, 3, 4, 5])
df.add_column("value", [1.1, 2.2, 3.3, 4.4, 5.5])
df.add_column("name", ["Alice", "Bob", "Charlie", "Dave", "Eve"])
df.add_column("signal", [1+0.5j, 2-1j, 3j, -1+0j, 0.5+0.5j])
assert df.num_rows() == 5
assert df.num_columns() == 4
print("PASSED")

print("\n=== Python Test 2: column_names / has_column ===")
names = df.column_names()
assert len(names) == 4
assert df.has_column("id")
assert not df.has_column("nope")
print(f"Columns: {names}")
print("PASSED")

print("\n=== Python Test 3: get_column ===")
ids = df.get_column("id")
assert ids == [1, 2, 3, 4, 5]
vals = df.get_column("value")
assert abs(vals[1] - 2.2) < 1e-9
strs = df.get_column("name")
assert strs[2] == "Charlie"
sigs = df.get_column("signal")
assert abs(sigs[0] - (1+0.5j)) < 1e-9
print("PASSED")

print("\n=== Python Test 4: at / set ===")
assert df.at("id", 0) == 1
df.set("value", 0, 99.9)
assert abs(df.at("value", 0) - 99.9) < 1e-9
print("PASSED")

print("\n=== Python Test 5: column_dtype ===")
assert df.column_dtype("id") == "int"
assert df.column_dtype("value") == "double"
assert df.column_dtype("name") == "string"
assert df.column_dtype("signal") == "complex"
print("PASSED")

print("\n=== Python Test 6: slice / head / tail ===")
s = df.slice(1, 3)
assert s.num_rows() == 2
assert s.at("id", 0) == 2

h = df.head(2)
assert h.num_rows() == 2

t = df.tail(2)
assert t.num_rows() == 2
assert t.at("id", 0) == 4
print("PASSED")

print("\n=== Python Test 7: remove_column ===")
df2 = df.copy()
df2.remove_column("name")
assert df2.num_columns() == 3
assert not df2.has_column("name")
print("PASSED")

print("\n=== Python Test 8: copy independence ===")
df3 = df.copy()
df3.set("id", 0, 999)
assert df.at("id", 0) == 1
assert df3.at("id", 0) == 999
print("PASSED")

print("\n=== Python Test 9: error handling ===")
try:
    df.add_column("id", [1, 2])
    assert False, "Should have raised"
except (RuntimeError, ValueError):
    pass

try:
    df.add_column("short_col", [1, 2])
    assert False, "Should have raised"
except (RuntimeError, ValueError):
    pass
print("PASSED")

print("\n=== Python Test 10: to_string / repr ===")
print(repr(df))
print("PASSED")

print("\n=== Python Test 11: bracket access (df['xxx']) and dot access (df.xxx) ===")
# bracket access returns column data directly
assert df["id"] == [1, 2, 3, 4, 5]
assert df["name"] == ["Alice", "Bob", "Charlie", "Dave", "Eve"]
assert abs(df["value"][1] - 2.2) < 1e-9
assert abs(df["signal"][0] - (1+0.5j)) < 1e-9
try:
    _ = df["nonexistent"]
    assert False, "Should have raised KeyError"
except KeyError:
    pass
# dot access returns sub-DataFrame
sub_id = df.id
assert isinstance(sub_id, pdf.DataFrame)
assert sub_id.has_column("id")
assert sub_id["id"] == [1, 2, 3, 4, 5]
try:
    _ = df.nonexistent
    assert False, "Should have raised AttributeError"
except AttributeError:
    pass
print("PASSED")

print("\n=== Python Test 12: access by index ===")
assert df.column_name(0) == "id"
assert df.column_name(1) == "value"
assert df.column_name(2) == "name"
assert df.column_name(3) == "signal"
# get_column by index
assert df.get_column(0) == [1, 2, 3, 4, 5]
assert abs(df.get_column(1)[1] - 2.2) < 1e-9
assert df.get_column(2) == ["Alice", "Bob", "Charlie", "Dave", "Eve"]
assert abs(df.get_column(3)[0] - (1+0.5j)) < 1e-9
# out of range
try:
    df.column_name(99)
    assert False, "Should have raised"
except (RuntimeError, IndexError):
    pass
print("PASSED")

print("\n=== Python Test 13: column units ===")
df_u = pdf.DataFrame()
df_u.add_column("voltage", [1.0, 2.0, 3.0], unit="V")
df_u.add_column("current", [0.1, 0.2, 0.3], unit="A")
df_u.add_column("index", [1, 2, 3])  # no unit
assert df_u.column_quantity("voltage") == "V"
assert df_u.column_quantity("current") == "A"
assert df_u.column_quantity("index") == ""
df_u.set_column_quantity("index", "n")
assert df_u.column_quantity("index") == "n"
print(repr(df_u))
# slice preserves units
s_u = df_u.slice(0, 2)
assert s_u.column_quantity("voltage") == "V"
# copy preserves units
c_u = df_u.copy()
assert c_u.column_quantity("current") == "A"
print("PASSED")

print("\n=== Python Test 14: insert_column / prepend_column ===")
df_ins = pdf.DataFrame()
df_ins.add_column("a", [1, 2, 3])
df_ins.add_column("c", [7, 8, 9])
df_ins.insert_column(1, "b", [4, 5, 6])
assert df_ins.column_names() == ["a", "b", "c"]
assert df_ins["b"] == [4, 5, 6]
df_ins.prepend_column("z", [0.1, 0.2, 0.3])
assert df_ins.column_names()[0] == "z"
assert df_ins.num_columns() == 4
# Insert at end
df_ins.insert_column(4, "end", ["x", "y", "z"])
assert df_ins.column_names()[-1] == "end"
# Out of range
try:
    df_ins.insert_column(99, "bad", [1, 2, 3])
    assert False, "Should have raised"
except (RuntimeError, IndexError):
    pass
# Duplicate name
try:
    df_ins.insert_column(0, "a", [1, 2, 3])
    assert False, "Should have raised"
except (RuntimeError, ValueError):
    pass
print("PASSED")

print("\n=== Python Test 15: Multi-index add_index ===")
mi = pdf.DataFrame()
mi.add_index("a", [1, 2])
mi.add_index("b", [10, 20, 30])
assert mi.num_rows() == 6
assert mi.num_columns() == 2
assert mi.num_indices() == 2
assert mi["a"] == [1, 1, 1, 2, 2, 2]
assert mi["b"] == [10, 20, 30, 10, 20, 30]
mi.add_column("v", [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
assert mi.num_columns() == 3
assert mi.is_index("a")
assert mi.is_index("b")
assert not mi.is_index("v")
assert mi.index_names() == ["a", "b"]
assert mi.dependent_names() == ["v"]
print(repr(mi))
print("PASSED")

print("\n=== Python Test 16: Multi-index strides & flat_index ===")
assert mi.strides() == [3, 1]
assert mi.flat_index([0, 0]) == 0
assert mi.flat_index([0, 2]) == 2
assert mi.flat_index([1, 0]) == 3
assert mi.flat_index([1, 2]) == 5
print("PASSED")

print("\n=== Python Test 17: Multi-index loc ===")
sub = mi.loc(0)  # fix b=10 (innermost at index 0)
assert sub.num_rows() == 2
assert sub.num_indices() == 1
assert sub["v"] == [1.0, 4.0]
print(repr(sub))

sub2 = mi.loc(1)  # fix b=20
assert sub2.num_rows() == 2
assert sub2["v"] == [2.0, 5.0]

sub3 = mi.loc(0, 1)  # right-aligned: 0->a, 1->b=20
assert sub3.num_rows() == 1
assert sub3.num_indices() == 0
assert sub3.at("v", 0) == 2.0

# chained loc: fix b=30, then a=2
sub4 = mi.loc(2).loc(1)
assert sub4.num_rows() == 1
assert sub4.at("v", 0) == 6.0
print("PASSED")

print("\n=== Python Test 18: Multi-index get_index_levels ===")
assert mi.get_index_levels(0) == [1, 2]
assert mi.get_index_levels("b") == [10, 20, 30]
print("PASSED")

print("\n=== Python Test 19: Multi-index copy preserves index ===")
mi2 = mi.copy()
assert mi2.num_indices() == 2
assert mi2.is_index("a")
assert mi2.index_names() == ["a", "b"]
print("PASSED")

print("\n=== Python Test 20: 3D multi-index ===")
mi3 = pdf.DataFrame()
mi3.add_index("a", [1, 2])
mi3.add_index("b", [10, 20, 30])
mi3.add_index("c", [100, 200])
assert mi3.num_rows() == 12
mi3.add_column("v", list(range(1, 13)))

sub = mi3.loc(1)  # fix c=200 (innermost)
assert sub.num_rows() == 6
assert sub.num_indices() == 2
assert sub.at("v", 0) == 2

sub2 = mi3.loc(1, 0)  # fix b=20, c=100
assert sub2.num_rows() == 2
assert sub2.num_indices() == 1

sub3 = mi3.loc(1, 0, 1)  # a=2, b=10, c=200
assert sub3.num_rows() == 1
assert sub3.at("v", 0) == 8
print("PASSED")

print("\n=== Python Test 21: set_index ===")
df_si = pdf.DataFrame()
df_si.add_column("a", [1, 1, 1, 2, 2, 2])
df_si.add_column("b", [10, 20, 30, 10, 20, 30])
df_si.add_column("v", [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
assert df_si.num_indices() == 0
df_si.set_index("a", "b")
assert df_si.num_indices() == 2
assert df_si.is_index("a")
assert df_si.is_index("b")
assert not df_si.is_index("v")
assert df_si.index_names() == ["a", "b"]
assert df_si.dependent_names() == ["v"]
# loc works after set_index
sub = df_si.loc(0)  # fix b=10
assert sub.num_rows() == 2
assert sub["v"] == [1.0, 4.0]
sub2 = df_si.loc(0, 1)  # a=1, b=20
assert sub2.num_rows() == 1
assert sub2.at("v", 0) == 2.0
print("PASSED")

print("\n=== Python Test 22: set_index validation ===")
# Non-Cartesian data
df_bad = pdf.DataFrame()
df_bad.add_column("a", [1, 1, 2, 2])
df_bad.add_column("b", [10, 20, 10, 30])
df_bad.add_column("v", [1, 2, 3, 4])
try:
    df_bad.set_index("a", "b")
    assert False, "Should have raised"
except (RuntimeError, ValueError):
    pass
print("PASSED")

print("\n=== Python Test 23: reset_index ===")
df_ri = pdf.DataFrame()
df_ri.add_column("a", [1, 1, 2, 2])
df_ri.add_column("b", [10, 20, 10, 20])
df_ri.add_column("v", [1.0, 2.0, 3.0, 4.0])
df_ri.set_index("a", "b")
assert df_ri.num_indices() == 2
df_ri.reset_index()
assert df_ri.num_indices() == 0
assert not df_ri.is_index("a")
assert df_ri.num_rows() == 4
assert df_ri.at("v", 0) == 1.0
# Can re-set
df_ri.set_index("a", "b")
assert df_ri.num_indices() == 2
sub = df_ri.loc(0)  # fix b=10
assert sub["v"] == [1.0, 3.0]
print("PASSED")

print("\n=== Python Test 24: set_index with list ===")
df_list = pdf.DataFrame()
df_list.add_column("x", [1, 1, 2, 2])
df_list.add_column("y", [10, 20, 10, 20])
df_list.add_column("z", [1.0, 2.0, 3.0, 4.0])
df_list.set_index(["x", "y"])  # pass as list
assert df_list.num_indices() == 2
assert df_list.is_index("x")
print("PASSED")

print("\n=== Python Test 25: set_index unordered ===")
df_uo = pdf.DataFrame()
df_uo.add_column("a", [2, 1, 2, 1, 1, 2])
df_uo.add_column("b", [30, 10, 10, 30, 20, 20])
df_uo.add_column("v", [6.0, 1.0, 4.0, 3.0, 2.0, 5.0])
df_uo.set_index("a", "b")
assert df_uo.num_indices() == 2
# After reorder: a=[2,2,2,1,1,1], b=[30,10,20,30,10,20]
# v=[6,4,5,3,1,2]
assert df_uo["a"] == [2, 2, 2, 1, 1, 1]
assert df_uo["b"] == [30, 10, 20, 30, 10, 20]
assert df_uo["v"] == [6.0, 4.0, 5.0, 3.0, 1.0, 2.0]
# loc works on reordered data
sub = df_uo.loc(1)  # fix b=10
assert sub.num_rows() == 2
assert sub["v"] == [4.0, 1.0]
print("PASSED")

print("\n=== Python Test 26: set_index already sorted ===")
df_sorted = pdf.DataFrame()
df_sorted.add_column("a", [1, 1, 2, 2])
df_sorted.add_column("b", [10, 20, 10, 20])
df_sorted.add_column("v", [1.0, 2.0, 3.0, 4.0])
df_sorted.set_index("a", "b")
assert df_sorted["v"] == [1.0, 2.0, 3.0, 4.0]  # unchanged
print("PASSED")

print("\n=== Python Test 27: set_index duplicate rows fails ===")
df_dup = pdf.DataFrame()
df_dup.add_column("a", [1, 1, 1, 2])
df_dup.add_column("b", [10, 10, 20, 10])
df_dup.add_column("v", [1, 2, 3, 4])
try:
    df_dup.set_index("a", "b")
    assert False, "Should have raised"
except (RuntimeError, ValueError):
    pass
print("PASSED")

print("\n=== Python Test 28: independent_names alias ===")
df30 = pdf.DataFrame()
df30.add_index("freq", [1, 2, 3])
df30.add_index("port", ["p1", "p2"])
df30.add_column("val", [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
assert df30.independent_names() == df30.index_names()
assert df30.independent_names() == ["freq", "port"]
print("PASSED")

print("\n=== Python Test 29: column_quantity alias ===")
df31 = pdf.DataFrame()
df31.add_column("voltage", [1.0, 2.0], "V")
assert df31.column_quantity("voltage") == "V"
df31.set_column_quantity("voltage", "mV")
assert df31.column_quantity("voltage") == "mV"
# Old name still works
assert df31.column_unit("voltage") == "mV"  # old alias still works
print("PASSED")

print("\n=== Python Test 30: sub dependent column ===")
df32 = pdf.DataFrame()
df32.add_index("a", [1, 2])
df32.add_index("b", [10, 20, 30])
df32.add_column("x", [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
df32.add_column("y", [7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
df32.type = "sparam"
s32 = df32.sub("x")
assert s32.num_columns() == 3  # a, b, x
assert s32.has_column("a")
assert s32.has_column("b")
assert s32.has_column("x")
assert not s32.has_column("y")
assert s32.num_rows() == 6
assert s32.type == "sparam"
print("PASSED")

print("\n=== Python Test 31: sub independent column ===")
df33 = pdf.DataFrame()
df33.add_index("a", [1, 2])
df33.add_index("b", [10, 20, 30])
df33.add_index("c", [100, 200])
df33.add_column("v", [0.0] * 12)
df33.type = "zparam"
s33 = df33.sub("b")
assert s33.num_columns() == 2  # a, b
assert s33.num_rows() == 6
assert s33.type == "zparam"
s33a = df33.sub("a")
assert s33a.num_columns() == 1
assert s33a.num_rows() == 2
print("PASSED")

print("\n=== Python Test 32: path/type/name metadata ===")
df34 = pdf.DataFrame()
df34.add_index("f", [1, 2])
df34.add_column("v", [10.0, 20.0])
assert df34.path == ""
assert df34.type == ""
assert df34.df_name == ""
df34.path = "/data/test.csv"
df34.type = "sparam"
df34.df_name = "S21"
assert df34.path == "/data/test.csv"
assert df34.type == "sparam"
assert df34.df_name == "S21"
# loc inherits type
l34 = df34.loc(0)
assert l34.type == "sparam"
# sub inherits type
s34 = df34.sub("v")
assert s34.type == "sparam"
print("PASSED")

print("\n=== Python Test 33: rename by name ===")
df35 = pdf.DataFrame()
df35.add_column("a", [1, 2, 3])
df35.add_column("b", [4.0, 5.0, 6.0])
df35.rename("b", "x")
assert not df35.has_column("b")
assert df35.has_column("x")
assert df35.column_names()[1] == "x"
assert df35.at("x", 0) == 4.0
# rename by index
df35.rename(0, "alpha")
assert df35.column_names()[0] == "alpha"
# duplicate name error
try:
    df35.rename("alpha", "x")
    assert False
except Exception:
    pass
print("PASSED")

print("\n=== Python Test 34: rename_last (chain with sub) ===")
df36 = pdf.DataFrame()
df36.add_index("a", [1, 2])
df36.add_index("b", [10, 20, 30])
df36.add_column("t", [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
# chain: sub then rename_last
result = df36.sub("t").rename_last("my_var")
assert result.column_names()[-1] == "my_var"
assert result.has_column("my_var")
assert not result.has_column("t")
assert result.at("my_var", 0) == 1.0
# rename returns self (same object)
df37 = pdf.DataFrame()
df37.add_column("x", [1, 2])
ret = df37.rename("x", "y")
assert ret is df37
print("PASSED")

# === Python Test 35: to_csv basic ===
print("\n=== Python Test 35: to_csv basic ===")
df_csv = pdf.DataFrame()
df_csv.add_column("id", [1, 2, 3])
df_csv.add_column("value", [1.1, 2.2, 3.3])
df_csv.add_column("name", ["Alice", "Bob", "Charlie"])
csv = df_csv.to_csv()
assert csv.startswith("id,value,name\n")
assert "1,1.1,Alice\n" in csv
assert "3,3.3,Charlie\n" in csv
print("PASSED")

# === Python Test 36: to_csv custom delimiter ===
print("\n=== Python Test 36: to_csv custom delimiter ===")
csv_tab = df_csv.to_csv('\t')
assert csv_tab.startswith("id\tvalue\tname\n")
assert "1\t1.1\tAlice\n" in csv_tab
print("PASSED")

# === Python Test 37: save_csv ===
print("\n=== Python Test 37: save_csv ===")
import os
df_csv.save_csv("test_output_py.csv")
with open("test_output_py.csv", "r") as f:
    content = f.read()
assert content == df_csv.to_csv()
os.remove("test_output_py.csv")
print("PASSED")

# === Python Test 38: to_csv empty ===
print("\n=== Python Test 38: to_csv empty ===")
df_empty = pdf.DataFrame()
assert df_empty.to_csv() == ""
print("PASSED")

print("\n=== ALL PYTHON TESTS PASSED ===")
