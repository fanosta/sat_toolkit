import collections
import numpy as np
import pytest

from sat_toolkit.formula import CNF, Clause, XorClause, XorClauseList, Truthtable


def test_basic():
    cnf = CNF()
    assert len(cnf) == 0

    cnf.add_clause([1, 2, 3])
    assert len(cnf) == 1
    assert cnf[0] == Clause([1, 2, 3])
    assert cnf.nvars == 3
    assert cnf[0].maxvar == 3

    cnf.add_clause([4, 5, 6])
    assert len(cnf) == 2
    assert cnf[1] == Clause([4, 5, 6])
    assert cnf[1].maxvar == 6
    assert cnf.nvars == 6

    assert np.all(np.array(cnf) == np.array([1, 2, 3, 0, 4, 5, 6, 0]))

    cnf.add_clause([-7, 8, -9])
    assert len(cnf) == 3
    assert cnf[2].maxvar == 9

    assert cnf[-1] == cnf[2]
    assert cnf[-2] == cnf[1]
    assert cnf[-3] == cnf[0]

    with pytest.raises(IndexError):
        cnf[3]
    with pytest.raises(IndexError):
        cnf[-4]

def test_nvars():
    cnf = CNF()
    assert cnf.nvars == 0

    with pytest.raises(ValueError):
        cnf.nvars = -1

    cnf.nvars = 0
    cnf.nvars = 1
    assert cnf.nvars == 1

    cnf.add_clause([1, 2, -3])
    assert cnf.nvars == 3

    with pytest.raises(ValueError):
        cnf.nvars = 2

    cnf.nvars = 3
    assert cnf.nvars == 3
    cnf.nvars = 4
    assert cnf.nvars == 4

def test_get_vars():
    cnf1 = CNF([1, 2, 3, 0, -4, 5, 6, 0])
    assert cnf1.get_vars() == {1, 2, 3, 4, 5, 6}

    cnf2 = CNF.create_xor([1, 2, 3], [4, 5, 6])
    cnf2 += CNF.create_xor([5, 6], [10, 11])
    assert cnf2.get_vars() == {1, 2, 3, 4, 5, 6, 10, 11}

    cnf3 = CNF([-1, -2, 0, 2, 4, 0])
    assert cnf3.get_vars() == {1, 2, 4}

    cnf4 = CNF([-(2**31 - 1), 0])
    assert cnf4.get_vars() == {2**31 - 1}


def test_incompatible_type():
    cnf = CNF([1, 2, 3, 0, -4, 5, 6, 0])
    cnf.add_clause([7, 8, -9])

    cnf.add_clause(Clause([1, -3, 5]))

    with pytest.raises(TypeError):
        cnf.add_clause(XorClause([2, -4, 6]))

    xor_clause_list = XorClauseList([1, 2, 0, -4, 5, 6, 8, 0])
    with pytest.raises(TypeError):
        cnf += xor_clause_list



def test_operators():
    cnf = CNF()
    cnf += CNF([-1, 2, -3, 0, 4, -5, 6, 0])
    cnf += CNF([1, 2, 3, 0])

    assert len(cnf) == 3
    assert cnf[0] == Clause([-1, 2, -3])
    assert cnf[1] == Clause([4, -5, 6])
    assert cnf[2] == Clause([1, 2, 3])

    assert cnf[1] in cnf
    assert XorClause(cnf[1]) not in cnf
    assert cnf.count(XorClause(cnf[1])) == 0
    assert cnf.count(Clause(cnf[1])) == 1

    assert cnf.index(Clause(cnf[1])) == 1
    with pytest.raises(ValueError):
        cnf.index(XorClause(cnf[1]))



def test_from_dimacs():
    dimacs = (
        "p cnf 8 2\n"
        "1 2 3 0\n"
        "4 5 6 0\n"
    )

    cnf = CNF.from_dimacs(dimacs)
    assert len(cnf) == 2
    assert cnf.nvars == 8
    assert cnf[0] == Clause([1, 2, 3])
    assert cnf[1] == Clause([4, 5, 6])

    # wrong number of clauses
    dimacs = (
        "p cnf 8 0\n"
        "1 2 3 0\n"
        "4 5 6 0\n"
    )

    cnf = CNF.from_dimacs(dimacs)
    assert len(cnf) == 2
    assert cnf.nvars == 8
    assert cnf[0] == Clause([1, 2, 3])
    assert cnf[1] == Clause([4, 5, 6])

def test_from_dimacs_format_errors():
    with pytest.raises(ValueError):
        CNF.from_dimacs("")
    with pytest.raises(ValueError):
        CNF.from_dimacs("p cnf 6 x\n")
    with pytest.raises(ValueError):
        CNF.from_dimacs("p cnf x 0\n")
    with pytest.raises(ValueError):
        CNF.from_dimacs("p cnf 6 0\np cnf 6 0\n")
    with pytest.raises(ValueError):
        CNF.from_dimacs("p cnf 6 1\n1 x 0\n")
    with pytest.raises(ValueError):
        CNF.from_dimacs("p cnf 6 1\nx 0\n")
    with pytest.raises(ValueError):
        CNF.from_dimacs("p cnf 6 1\nx1 0\n")

def test_get_units():
    cnf = CNF()
    cnf.add_clause([1, 2, 3])
    cnf.add_clause([4, 5, 6])
    cnf.add_clause([5])
    cnf.add_clause([-2, -3, -4])
    cnf.add_clause([-3])
    cnf.add_clause([-3, 5])

    assert np.all(np.sort(cnf.get_units()) == np.array([-3, 5], dtype=np.int32))


def test_iter():
    cnf = CNF([1, 2, 3, 0, 4, 5, 6, 0])
    cnf2 = CNF()

    for clause in cnf:
        cnf2.add_clause(clause)

    assert cnf2 == cnf


def test_translate():
    cnf = CNF([-1, 2, 3, 0, -4, -5, 6, 0])

    mapping = np.zeros(7, dtype=np.int32)
    mapping[1:4] = [4, -5, -6]
    mapping[4:7] = [1, -2, 3]

    cnf2 = cnf.translate(mapping)

    assert cnf2.nvars == 6
    assert len(cnf2) == 2
    assert cnf2[0] == Clause([-4, -5, -6])
    assert cnf2[1] == Clause([-1, 2, 3])
    assert cnf2 == CNF([-4, -5, -6, 0, -1, 2, 3, 0])


def test_pickle():
    import pickle

    cnf1 = CNF([1, 2, 3, 0, 4, 5, 6, 0])
    cnf2 = CNF([1, 2, 3, 0, 4, 5, 6, 0], nvars=10)

    cnf1_pickled = pickle.dumps(cnf1)
    cnf2_pickled = pickle.dumps(cnf2)

    cnf1_loaded = pickle.loads(cnf1_pickled)
    cnf2_loaded = pickle.loads(cnf2_pickled)

    assert cnf1 == cnf1_loaded
    assert cnf2 == cnf2_loaded
    assert cnf2.nvars == cnf2_loaded.nvars == 10


def test_contains():
    cnf = CNF([1, 2, 3, 0, 4, 5, 6, 0])
    assert Clause([1, 2, 3]) in cnf
    assert Clause([1, -2, 3]) not in cnf
    assert Clause([1, 3, 2]) not in cnf
    assert Clause([1, 2]) not in cnf
    assert Clause([2, 3]) not in cnf
    assert Clause([4, 5]) not in cnf
    assert Clause([5, 6]) not in cnf

    assert cnf.count(Clause([1, 2, 3])) == 1
    assert cnf.count(Clause([4, 5, 6])) == 1
    assert cnf.count(Clause([1, 2])) == 0
    assert cnf.count(Clause([4, 5])) == 0
    cnf.add_clause([1, 2, 3])
    assert cnf.count(Clause([1, 2, 3])) == 2
    assert cnf.count(Clause([1, 2])) == 0


def test_logical_or():
    cnf = CNF([1, 2, 3, 0, 4, 5, 6, 0])
    cnf2 = cnf.logical_or(7)
    assert cnf2 is not cnf

    assert len(cnf2) == 2
    assert cnf2[0] == Clause([1, 2, 3, 7])
    assert cnf2[1] == Clause([4, 5, 6, 7])

    cnf3 = cnf.implied_by(10)
    assert len(cnf3) == 2
    assert cnf3[0] == Clause([1, 2, 3, -10])
    assert cnf3[1] == Clause([4, 5, 6, -10])

def test_clause():
    clause = Clause([1, 2, 3])
    assert len(clause) == 3
    assert clause[0] == 1
    assert clause[1] == 2
    assert clause[2] == 3

    assert isinstance(clause, collections.abc.Sequence)

    assert Clause([1, 2, 3]) == Clause([1, 2, 3])
    assert Clause([1, 2, 3]) != XorClause([1, 2, 3])
    assert XorClause([1, 2, 3]) != Clause([1, 2, 3])

    with pytest.raises(IndexError):
        clause[3]

    with pytest.raises(ValueError):
        Clause([1, 2, 3, 0])

def test_to_cnf():
    lut = np.zeros(16, dtype=np.int8)
    lut[[0, 6, 9, 15]] = 1

    tt = Truthtable.from_lut(lut)
    cnf = tt.to_cnf()
    cnf_ref = CNF.create_xor([1, 2], [4, 3])

    assert cnf.equiv(cnf_ref)


def test_create_xor():
    cnf_equal = CNF.create_xor([1], [2])
    assert len(cnf_equal) == 2
    assert Clause([1, -2]) in cnf_equal
    assert Clause([-1, 2]) in cnf_equal

    cnf_not_equal = CNF.create_xor([1], [2], rhs=np.array([1]))
    assert len(cnf_not_equal) == 2
    assert Clause([1, 2]) in cnf_not_equal
    assert Clause([-1, -2]) in cnf_not_equal

    assert cnf_not_equal.equiv(CNF.create_xor([-1], [2]))
    assert cnf_not_equal.equiv(CNF.create_xor([1], [-2]))

    with pytest.raises(ValueError):
        CNF.create_xor(rhs=[1])
    with pytest.raises(ValueError):
        CNF.create_xor([0], [1])
    with pytest.raises(ValueError):
        CNF.create_xor([1, 2], [3, 0])
    with pytest.raises(ValueError):
        CNF.create_xor([0])

    cnf_xor3 = CNF.create_xor([1], [2], [3])
    assert len(cnf_xor3) == 4
    assert Clause([1, 2, -3]) in cnf_xor3
    assert Clause([1, -2, 3]) in cnf_xor3
    assert Clause([-1, 2, 3]) in cnf_xor3
    assert Clause([-1, -2, -3]) in cnf_xor3

    cnf_xor3_multi = CNF.create_xor([1, 4], [2, 5], [3, 6])
    assert len(cnf_xor3_multi) == 8
    assert Clause([1, 2, -3]) in cnf_xor3_multi
    assert Clause([1, -2, 3]) in cnf_xor3_multi
    assert Clause([-1, 2, 3]) in cnf_xor3_multi
    assert Clause([-1, -2, -3]) in cnf_xor3_multi

    assert Clause([4, 5, -6]) in cnf_xor3_multi
    assert Clause([4, -5, 6]) in cnf_xor3_multi
    assert Clause([-4, 5, 6]) in cnf_xor3_multi
    assert Clause([-4, -5, -6]) in cnf_xor3_multi
