import collections
import numpy as np
import pytest

from sat_toolkit.formula import CNF, Clause, XorClause


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

    xor_clause = XorClause([1, -2, 3])
    assert len(xor_clause) == 3
    assert xor_clause[0] == 1
    assert xor_clause[1] == -2
    assert xor_clause[2] == 3

    with pytest.raises(IndexError):
        xor_clause[3]

    with pytest.raises(IndexError):
        clause[3]

    with pytest.raises(ValueError):
        Clause([1, 2, 3, 0])

    with pytest.raises(ValueError):
        XorClause([1, 2, 3, 0])
