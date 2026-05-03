import argparse
import os
import sys
from pathlib import Path


def _candidate_module_dirs(script_dir: Path):
    module_dir = os.environ.get("EXPRDF_MODULE_DIR")
    if module_dir:
        yield Path(module_dir)

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--module-dir")
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    if args.module_dir:
        yield Path(args.module_dir)

    build_dir = script_dir / "build"
    if build_dir.exists():
        yield build_dir
        for config_dir in ("Debug", "Release", "RelWithDebInfo", "MinSizeRel"):
            yield build_dir / config_dir


def _configure_import_path():
    script_dir = Path(__file__).resolve().parent
    candidates = []
    seen = set()

    for candidate in _candidate_module_dirs(script_dir):
        resolved = candidate.resolve()
        key = str(resolved)
        if key in seen:
            continue
        seen.add(key)
        candidates.append(resolved)

    for candidate in candidates:
        if any(candidate.glob("exprdf*.pyd")) or any(candidate.glob("exprdf*.so")):
            sys.path.insert(0, str(candidate))
            return

    searched = "\n".join(str(path) for path in candidates) or "<none>"
    raise ImportError(
        "Unable to locate the exprdf Python module.\n"
        "Pass --module-dir, set EXPRDF_MODULE_DIR, or build the module with CMake first.\n"
        f"Searched:\n{searched}"
    )


_configure_import_path()

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

print("\n=== Python Test 13: column quantities ===")
df_u = pdf.DataFrame()
df_u.add_column("voltage", [1.0, 2.0, 3.0], quantity="voltage")
df_u.add_column("current", [0.1, 0.2, 0.3], quantity="current")
df_u.add_column("index", [1, 2, 3])  # no quantity
assert df_u.column_quantity("voltage") == "voltage"
assert df_u.column_quantity("current") == "current"
assert df_u.column_quantity("index") == ""
df_u.set_column_quantity("index", "temperature")
assert df_u.column_quantity("index") == "temperature"
print(repr(df_u))
# slice preserves quantities
s_u = df_u.slice(0, 2)
assert s_u.column_quantity("voltage") == "voltage"
# copy preserves quantities
c_u = df_u.copy()
assert c_u.column_quantity("current") == "current"
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

print("\n=== Python Test 15: Multi-index add_uniform_index ===")
mi = pdf.DataFrame()
mi.add_uniform_index("a", [1, 2])
mi.add_uniform_index("b", [10, 20, 30])
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

print("\n=== Python Test 18: Multi-index get_index_dim ===")
dim_a = mi.get_index_dim(0)
assert dim_a.levels == [1, 2]
assert dim_a.level_count == 2
assert dim_a.kind == "uniform"
dim_b = mi.get_index_dim("b")
assert dim_b.levels == [10, 20, 30]
assert dim_b.level_count == 3
print("PASSED")

print("\n=== Python Test 19: Multi-index copy preserves index ===")
mi2 = mi.copy()
assert mi2.num_indices() == 2
assert mi2.is_index("a")
assert mi2.index_names() == ["a", "b"]
print("PASSED")

print("\n=== Python Test 20: 3D multi-index ===")
mi3 = pdf.DataFrame()
mi3.add_uniform_index("a", [1, 2])
mi3.add_uniform_index("b", [10, 20, 30])
mi3.add_uniform_index("c", [100, 200])
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
# a=[1,1,1,2], b=[10,20,30,10]: ragged → should throw with new set_index
df_v22 = pdf.DataFrame()
df_v22.add_column("a", [1, 1, 1, 2])
df_v22.add_column("b", [10, 20, 30, 10])
df_v22.add_column("v", [1, 2, 3, 4])
caught_v22 = False
try:
    df_v22.set_index("a", "b")
except Exception:
    caught_v22 = True
assert caught_v22

# Nonexistent column still raises
df_bad2 = pdf.DataFrame()
df_bad2.add_column("a", [1, 2])
try:
    df_bad2.set_index("nonexistent")
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
df30.add_uniform_index("freq", [1, 2, 3])
df30.add_uniform_index("port", ["p1", "p2"])
df30.add_column("val", [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
assert df30.independent_names() == df30.index_names()
assert df30.independent_names() == ["freq", "port"]
print("PASSED")

print("\n=== Python Test 29: column_quantity API ===")
df31 = pdf.DataFrame()
df31.add_column("voltage", [1.0, 2.0], "voltage")
assert df31.column_quantity("voltage") == "voltage"
df31.set_column_quantity("voltage", "current")
assert df31.column_quantity("voltage") == "current"
print("PASSED")

print("\n=== Python Test 30: sub dependent column ===")
df32 = pdf.DataFrame()
df32.add_uniform_index("a", [1, 2])
df32.add_uniform_index("b", [10, 20, 30])
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
df33.add_uniform_index("a", [1, 2])
df33.add_uniform_index("b", [10, 20, 30])
df33.add_uniform_index("c", [100, 200])
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
df34.add_uniform_index("f", [1, 2])
df34.add_column("v", [10.0, 20.0])
assert df34.path == ""
assert df34.type == ""
assert df34.name == ""
df34.path = "/data/test.csv"
df34.type = "sparam"
df34.name = "S21"
assert df34.path == "/data/test.csv"
assert df34.type == "sparam"
assert df34.name == "S21"
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
df36.add_uniform_index("a", [1, 2])
df36.add_uniform_index("b", [10, 20, 30])
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

# === Python Test 39: add_grouped_index basic ===
print("\n=== Python Test 39: add_grouped_index basic ===")
df_var = pdf.DataFrame()
df_var.add_uniform_index("bias", [1, 2])
df_var.add_grouped_index("freq", [1.0, 2.0, 3.0, 1.5, 2.5, 3.5], 3, "frequency")
assert df_var.num_rows() == 6
assert df_var.num_indices() == 2
dim_freq = df_var.get_index_dim("freq")
assert dim_freq.name == "freq"
assert dim_freq.kind == "grouped"
assert dim_freq.group_lengths[0] == 3
df_var.add_column("S11", [-10.0, -20.0, -30.0, -15.0, -25.0, -35.0])
print(df_var)
print("PASSED")

# === Python Test 40: add_grouped_index loc ===
print("\n=== Python Test 40: add_grouped_index loc ===")
sub_var = df_var.loc(1)  # fix freq at index 1
assert sub_var.num_rows() == 2
s11 = sub_var["S11"]
assert abs(s11[0] - (-20.0)) < 1e-12
assert abs(s11[1] - (-25.0)) < 1e-12
print("PASSED")

# === Python Test 41: first index cannot be varying ===
print("\n=== Python Test 41: first index cannot be varying ===")
df_first = pdf.DataFrame()
caught = False
try:
    df_first.add_grouped_index("freq", [1.0, 2.0], 2)
except Exception:
    caught = True
assert caught
print("PASSED")

# === Python Test 42: add_grouped_index basic ===
print("\n=== Python Test 42: add_grouped_index basic ===")
df_groups = pdf.DataFrame()
df_groups.add_uniform_index("bias", [1, 2])
df_groups.add_grouped_index("freq", [1.0, 2.0, 3.0, 1.5, 2.5, 3.5], 3, "frequency")
assert df_groups.num_rows() == 6
df_groups.add_uniform_index("port", ["S11", "S21"])
assert df_groups.num_rows() == 12
print("PASSED")

# === Python Test 44: set_index auto infer varying in middle ===
print("\n=== Python Test 44: set_index auto infer varying ===")
df_mix = pdf.DataFrame()
df_mix.add_column("bias", [1,1,1,1,1,1,2,2,2,2,2,2])
df_mix.add_column("freq", [1.0,1.0,2.0,2.0,3.0,3.0,1.5,1.5,2.5,2.5,3.5,3.5])
df_mix.add_column("port", ["S11","S21","S11","S21","S11","S21",
                            "S11","S21","S11","S21","S11","S21"])
df_mix.add_column("S", [-10,-11,-20,-21,-30,-31,-15,-16,-25,-26,-35,-36])
df_mix.set_index("bias", "freq", "port")

d0 = df_mix.get_index_dim("bias")
d1 = df_mix.get_index_dim("freq")
d2 = df_mix.get_index_dim("port")
assert d0.kind == "uniform" and d0.level_count == 2
assert d1.kind == "grouped" and len(d1.group_lengths) == 6 and d1.group_lengths[0] == 2
assert d2.kind == "uniform" and d2.level_count == 2
print("PASSED")

# === Python Test 45: set_index keeps unordered uniform behavior ===
print("\n=== Python Test 45: set_index unordered uniform ===")
df_unordered = pdf.DataFrame()
df_unordered.add_column("a", [2,2,2,1,1,1])
df_unordered.add_column("b", [30,10,20,30,10,20])
df_unordered.add_column("v", [6,4,5,3,1,2])
df_unordered.set_index("a", "b")
assert df_unordered["a"][0] == 2 and df_unordered["b"][0] == 30
assert df_unordered["a"][5] == 1 and df_unordered["b"][5] == 20
print("PASSED")

# === Python Test 46: set_index infer two varying dims ===
print("\n=== Python Test 46: set_index infer two varying dims ===")
df_mix2 = pdf.DataFrame()
df_mix2.add_column("bias", [1,1,1,1,2,2,2,2])
df_mix2.add_column("freq", [1.0,1.0,2.0,2.0,1.5,1.5,2.5,2.5])
df_mix2.add_column("port", ["S11","S21","S11","S21","S11","S21","S11","S21"])
df_mix2.add_column("S", [-10,-11,-20,-21,-15,-16,-25,-26])
df_mix2.set_index("bias", "freq", "port")
assert df_mix2.get_index_dim("bias").kind == "uniform"
assert df_mix2.get_index_dim("freq").kind == "grouped"
assert len(df_mix2.get_index_dim("freq").group_lengths) == 4 and df_mix2.get_index_dim("freq").group_lengths[0] == 2
assert df_mix2.get_index_dim("port").kind == "uniform"
print("PASSED")

# === Python Test 47: add two varying plus inner uniform ===
print("\n=== Python Test 47: add two varying + inner uniform ===")
df_v2 = pdf.DataFrame()
df_v2.add_uniform_index("bias", [1, 2])
df_v2.add_grouped_index("freq", [1.0, 2.0, 1.5, 2.5], 2)
df_v2.add_grouped_index("temp", [25,30,25,30,26,31,26,31], 2, "temperature")
df_v2.add_uniform_index("port", ["S11", "S21"])
assert df_v2.num_rows() == 16
assert df_v2.get_index_dim("freq").kind == "grouped"
assert df_v2.get_index_dim("temp").kind == "grouped"
assert df_v2.get_index_dim("port").kind == "uniform"
print("PASSED")

# === Python Test 48: set_index invalid mixed blocks ===
print("\n=== Python Test 48: set_index invalid mixed blocks ===")
df_bad2 = pdf.DataFrame()
df_bad2.add_column("bias", [1,1,2,2])
df_bad2.add_column("freq", [10,10,20,10])
df_bad2.add_column("v", [1,2,3,4])
caught = False
try:
    df_bad2.set_index("bias", "freq")
except Exception:
    caught = True
assert caught
print("PASSED")

# === Python Test 49: reset then set_index infer mixed ===
print("\n=== Python Test 49: reset then set_index infer mixed ===")
df_reset = pdf.DataFrame()
df_reset.add_uniform_index("bias", [1, 2])
df_reset.add_grouped_index("freq", [1.0, 2.0, 3.0, 1.5, 2.5, 3.5], 3)
df_reset.add_uniform_index("port", ["S11", "S21"])
df_reset.add_column("S", [-10,-11,-20,-21,-30,-31,-15,-16,-25,-26,-35,-36])
df_reset.reset_index()
df_reset.set_index("bias", "freq", "port")
assert df_reset.get_index_dim("freq").kind == "grouped"
assert len(df_reset.get_index_dim("freq").group_lengths) == 6 and df_reset.get_index_dim("freq").group_lengths[0] == 2
print("PASSED")

# === Python Test 50: set_index with 1/3/4 columns, keep 2nd as dependent ===
print("\n=== Python Test 50: set_index selected columns layout ===")
df_sel = pdf.DataFrame()
df_sel.add_column("bias", [2,1,2,1,1,2,1,2])
df_sel.add_column("val",  [222,111,211,122,112,221,121,212])
df_sel.add_column("freq", [20,10,10,20,10,20,20,10])
df_sel.add_column("port", ["S21","S11","S11","S21","S21","S11","S11","S21"])

df_sel.set_index("bias", "freq", "port")
assert df_sel.num_indices() == 3
assert df_sel.dependent_names() == ["val"]

cols = df_sel.column_names()
assert cols[0] == "bias"
assert cols[1] == "freq"
assert cols[2] == "port"
assert cols[3] == "val"

v = df_sel["val"]
assert v == [222,221,212,211,122,121,112,111]
print("PASSED")

# === Python Test 51: loc with -1 wildcard ===
print("\n=== Python Test 51: loc wildcard (-1) ===")
df_w = pdf.DataFrame()
df_w.add_column("a", [1,1,1,2,2,2])
df_w.add_column("b", [10,20,30,10,20,30])
df_w.add_column("v", [1.0,2.0,3.0,4.0,5.0,6.0])
df_w.set_index("a", "b")

# loc(-1): wildcard on last dim, all rows kept
s1 = df_w.loc(-1)
assert s1.num_rows() == 6
assert s1.num_indices() == 2
assert len(s1.column_names()) == 3  # a, b, v

# loc(0, -1): fix a=1, wildcard on b
s2 = df_w.loc(0, -1)
assert s2.num_rows() == 3
assert s2.num_indices() == 1
assert s2["v"] == [1.0, 2.0, 3.0]
assert s2["b"] == [10, 20, 30]

# loc(-1, 0): wildcard a, fix b=10
s3 = df_w.loc(-1, 0)
assert s3.num_rows() == 2
assert s3.num_indices() == 1
assert s3["v"] == [1.0, 4.0]
assert s3["a"] == [1, 2]
assert len(s3.column_names()) == 2  # a, v (no b)
print("PASSED")

# === Python Test 52: loc wildcard 3 dims ===
print("\n=== Python Test 52: loc wildcard 3 dims ===")
df_w3 = pdf.DataFrame()
df_w3.add_column("x", ["A","A","A","A","B","B","B","B"])
df_w3.add_column("y", [1,1,2,2,1,1,2,2])
df_w3.add_column("z", [10,20,10,20,10,20,10,20])
df_w3.add_column("val", [1,2,3,4,5,6,7,8])
df_w3.set_index("x", "y", "z")

# loc(0, -1, 1): x=A, wildcard y, z=20
s4 = df_w3.loc(0, -1, 1)
assert s4.num_rows() == 2
assert s4.num_indices() == 1
assert s4["val"] == [2, 4]
assert len(s4.column_names()) == 2  # y, val

# loc(-1, 0, -1): wildcard x, y=1, wildcard z
s5 = df_w3.loc(-1, 0, -1)
assert s5.num_rows() == 4
assert s5.num_indices() == 2  # x and z remain
assert s5["val"] == [1, 2, 5, 6]
print("PASSED")

# === Test 53: add_grouped_index_groups basic ===
print("\n=== Test 53: add_grouped_index_groups basic ===")
df_r = pdf.DataFrame()
df_r.add_uniform_index("level", [0, 1])
df_r.add_grouped_index_groups("number", [[0, 1, 2], [0, 1]])
assert df_r.num_rows() == 5
assert df_r.num_indices() == 2
dim_n = df_r.get_index_dim("number")
assert dim_n.kind == "grouped"
assert list(dim_n.group_lengths) == [3, 2]
assert df_r["level"] == [0, 0, 0, 1, 1]
assert df_r["number"] == [0, 1, 2, 0, 1]
print("PASSED")

# === Test 54: loc on ragged dimension ===
print("\n=== Test 54: loc on ragged dimension ===")
df_r2 = pdf.DataFrame()
df_r2.add_uniform_index("level", [0, 1])
df_r2.add_grouped_index_groups("number", [[0, 1, 2], [0, 1]])
# Points: (L0,N0)=2pts, (L0,N1)=3pts, (L0,N2)=1pt, (L1,N0)=4pts, (L1,N1)=2pts
df_r2.add_grouped_index_groups("point", [[0, 1], [0, 1, 2], [0], [0, 1, 2, 3], [0, 1]])
assert df_r2.num_rows() == 12

# loc(0): fix point=0 → every contour has pt0: 5 rows
s_r0 = df_r2.loc(0)
assert s_r0.num_rows() == 5
assert s_r0.num_indices() == 2  # level + number remain

# loc(1): fix point=1 → 4 contours have ≥2 pts (not L0N2): 4 rows
s_r1 = df_r2.loc(1)
assert s_r1.num_rows() == 4

# loc(2): fix point=2 → only (L0,N1) and (L1,N0) have ≥3 pts: 2 rows
s_r2 = df_r2.loc(2)
assert s_r2.num_rows() == 2

# loc(0, 0): fix number=0, point=0 → (L0,N0,pt0) + (L1,N0,pt0): 2 rows
s_r00 = df_r2.loc(0, 0)
assert s_r00.num_rows() == 2
assert s_r00.num_indices() == 1  # only level remains

# loc(2, 0): number=2, point=0 → only (L0,N2,pt0): 1 row
s_r20 = df_r2.loc(2, 0)
assert s_r20.num_rows() == 1
print("PASSED")

# === Test 55: set_index rejects ragged; manual construction works ===
print("\n=== Test 55: set_index detects ragged ===")
df_si = pdf.DataFrame()
df_si.add_column("level",  [0, 0, 0, 1, 1])
df_si.add_column("number", [0, 1, 2, 0, 1])
df_si.add_column("val",    [10.0, 20.0, 30.0, 40.0, 50.0])
try:
    df_si.set_index(["level", "number"])
    assert False, "Expected exception for ragged set_index"
except Exception:
    pass  # expected: ragged structure should throw

# Manual construction of ragged index
df_si2 = pdf.DataFrame()
df_si2.add_uniform_index("level", [0, 1])
df_si2.add_grouped_index_groups("number", [[0, 1, 2], [0, 1]])
df_si2.add_column("val", [10.0, 20.0, 30.0, 40.0, 50.0])
assert df_si2.num_indices() == 2
assert df_si2.get_index_dim("level").kind == "uniform"
assert df_si2.get_index_dim("number").kind == "grouped"
s_si = df_si2.loc(1)
assert s_si.num_rows() == 2
assert s_si["val"] == [20.0, 50.0]
print("PASSED")

# === Test 56: flat_index / multi_index Uniform ===
print("\n=== Test 56: flat_index/multi_index Uniform ===")
df56 = pdf.DataFrame()
df56.add_uniform_index("bias", [1, 2])
df56.add_uniform_index("freq", [10, 20, 30])
df56.add_column("v", list(range(6)))
assert df56.flat_index([0, 0]) == 0
assert df56.flat_index([0, 2]) == 2
assert df56.flat_index([1, 0]) == 3
assert df56.flat_index([1, 2]) == 5
assert list(df56.multi_index(0)) == [0, 0]
assert list(df56.multi_index(3)) == [1, 0]
assert list(df56.multi_index(5)) == [1, 2]
for r in range(6):
    assert df56.flat_index(df56.multi_index(r)) == r
print("PASSED")

# === Test 57: flat_index / multi_index Grouped ===
print("\n=== Test 57: flat_index/multi_index Grouped ===")
df57 = pdf.DataFrame()
df57.add_uniform_index("level", [0, 1])
df57.add_grouped_index_groups("number", [[0, 1, 2], [0, 1]])
assert df57.flat_index([0, 0]) == 0
assert df57.flat_index([0, 2]) == 2
assert df57.flat_index([1, 0]) == 3
assert df57.flat_index([1, 1]) == 4
try:
    df57.flat_index([1, 2])
    assert False, "should raise"
except Exception:
    pass
assert list(df57.multi_index(0)) == [0, 0]
assert list(df57.multi_index(2)) == [0, 2]
assert list(df57.multi_index(3)) == [1, 0]
assert list(df57.multi_index(4)) == [1, 1]
for r in range(5):
    assert df57.flat_index(df57.multi_index(r)) == r
print("PASSED")

# === Test 58: flat_index / multi_index mixed Uniform+Grouped ===
print("\n=== Test 58: flat_index/multi_index mixed ===")
df58 = pdf.DataFrame()
df58.add_uniform_index("bias", [1, 2])
df58.add_grouped_index("freq", [1.0, 2.0, 3.0, 1.5, 2.5, 3.5], 3)
df58.add_uniform_index("port", ["S11", "S21"])
df58.add_column("v", list(range(12)))
assert df58.flat_index([0, 0, 0]) == 0
assert df58.flat_index([0, 0, 1]) == 1
assert df58.flat_index([0, 2, 1]) == 5
assert df58.flat_index([1, 0, 0]) == 6
assert df58.flat_index([1, 2, 1]) == 11
assert list(df58.multi_index(0))  == [0, 0, 0]
assert list(df58.multi_index(5))  == [0, 2, 1]
assert list(df58.multi_index(6))  == [1, 0, 0]
assert list(df58.multi_index(11)) == [1, 2, 1]
for r in range(12):
    assert df58.flat_index(df58.multi_index(r)) == r
print("PASSED")

print("\n=== ALL PYTHON TESTS PASSED ===")
