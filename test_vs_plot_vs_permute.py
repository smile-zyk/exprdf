"""Focused tests for vs / plot_vs / permute (ADS-style interface)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "build" / "Debug"))
sys.path.insert(0, str(Path(__file__).resolve().parent / "build"))

import exprdf as pdf


def expect_error(fn):
    threw = False
    try:
        fn()
    except Exception:
        threw = True
    assert threw, "expected an exception"


print("=== vs() ===")
# ADS Example 1: a=[1,2,3]; b=[4,5,6]; c = vs(a, b)
dep_a = pdf.DataFrame()
dep_a.add_column("a", [1.0, 2.0, 3.0])
ind_b = pdf.DataFrame()
ind_b.add_column("b", [4, 5, 6])

c = pdf.vs(dep_a, ind_b)
assert c.num_rows() == 3
assert c.index_names() == ["b"]
assert c["b"] == [4, 5, 6]
assert c["a"] == [1.0, 2.0, 3.0]
assert c.is_index("b")
assert not c.is_index("a")

# vs() with explicit indepName override
c_named = pdf.vs(dep_a, ind_b, "MyCustomX")
assert c_named.index_names() == ["MyCustomX"]
assert c_named["MyCustomX"] == [4, 5, 6]

# vs() uses the LAST column of the independent DataFrame as the X axis,
# even when that DataFrame has its own multi-index columns.
dep2 = pdf.DataFrame()
dep2.add_column("y", [10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
idx2 = pdf.DataFrame()
idx2.add_uniform_index("k", [1, 2])
idx2.add_uniform_index("t", [0.1, 0.2, 0.3])
idx2.add_column("dummy", [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

c2 = pdf.vs(dep2, idx2)              # last col of idx2 is "dummy"
assert c2.index_names() == ["dummy"]
assert c2["dummy"] == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
assert c2["y"] == [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]

c2b = pdf.vs(dep2, idx2.sub("t"))    # sub("t") -> columns [k, t], last col "t"
assert c2b.index_names() == ["t"]
assert c2b["t"] == [0.1, 0.2, 0.3, 0.1, 0.2, 0.3]
assert c2b["y"] == [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]

# vs() requires matching row counts
dep3 = pdf.DataFrame()
dep3.add_column("z", [1.0, 2.0])
expect_error(lambda: pdf.vs(dep3, ind_b))
print("PASSED")

print("=== plot_vs() ===")
# ADS Example 2: dbS11 depends on [Cval(10), Freq(20)]
# flat index = cval*20 + (freq-1)
pf = pdf.DataFrame()
pf.add_uniform_index("Cval", [float(c) for c in range(10)])
pf.add_uniform_index("Freq", [float(f) for f in range(1, 21)])
pf.add_column("dbS11", [float(i) for i in range(200)])

# Case A: independent IS an existing index -> permute it innermost
pv = pdf.plot_vs(pf, pf.sub("Cval"))
assert pv.index_names() == ["Freq", "Cval"]
assert pv.num_rows() == 200
assert pv["Freq"] == [float(f) for f in range(1, 21) for _ in range(10)]
assert pv["Cval"] == [float(c) for _ in range(20) for c in range(10)]
assert pv["dbS11"] == [float(c * 20 + f) for f in range(20) for c in range(10)]

# Case B: dissimilar vector matching an axis size -> splice axis values
cval_h = pdf.DataFrame()
cval_h.add_column("CvalH", [float(c) / 2 for c in range(10)])
pv2 = pdf.plot_vs(pf, cval_h)
assert pv2.index_names() == ["Freq", "CvalH"]
assert pv2.num_rows() == 200
assert pv2["CvalH"] == [float(c) / 2 for _ in range(20) for c in range(10)]
assert pv2["dbS11"] == [float(c * 20 + f) for f in range(20) for c in range(10)]

# Case C: full-size vector -> fresh attach (like vs())
full_vec = pdf.DataFrame()
full_vec.add_column("xfull", [float(i) for i in range(200)])
pv3 = pdf.plot_vs(pf, full_vec)
assert pv3.index_names() == ["xfull"]
assert pv3["xfull"] == [float(i) for i in range(200)]
assert pv3["dbS11"] == [float(i) for i in range(200)]

# Case D: plot_vs(Ids.i, Vgs) style with 2 axes -> swap Vds/Vgs
dc = pdf.DataFrame()
dc.add_uniform_index("Vds", [0.0, 0.5, 1.0])
dc.add_uniform_index("Vgs", [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
dc.add_column("Ids", [float(i) for i in range(18)])
p_dc = pdf.plot_vs(dc, dc.sub("Vds"))
assert p_dc.index_names() == ["Vgs", "Vds"]
assert p_dc["Ids"] == [float(v * 6 + g) for g in range(6) for v in range(3)]

# mismatched size -> error
bad = pdf.DataFrame()
bad.add_column("bad", [1.0, 2.0])
expect_error(lambda: pdf.plot_vs(pf, bad))
print("PASSED")

print("\n=== ALL vs / plot_vs TESTS PASSED ===")