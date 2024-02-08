import collections
import numpy as np
import pytest

from sat_toolkit.formula import Clause, XorClause, XorClauseList

def test_xor_clause_list():
    xor_clauses = XorClauseList()
    assert len(xor_clauses) == 0

    xor_clauses.add_clause([1, 2, 3])
    assert len(xor_clauses) == 1
    assert xor_clauses[0] == XorClause([1, 2, 3])
    assert xor_clauses.nvars == 3
    assert xor_clauses[0].maxvar == 3

    xor_clauses.add_clause([4, 5, 6])
    assert len(xor_clauses) == 2
    assert xor_clauses[1] == XorClause([4, 5, 6])
    assert xor_clauses[1].maxvar == 6
    assert xor_clauses.nvars == 6

    assert np.all(np.array(xor_clauses) == np.array([1, 2, 3, 0, 4, 5, 6, 0]))

    xor_clauses.add_clause([-7, 8, -9])
    assert len(xor_clauses) == 3
    assert xor_clauses[2].maxvar == 9

    assert xor_clauses[-1] == xor_clauses[2]
    assert xor_clauses[-2] == xor_clauses[1]
    assert xor_clauses[-3] == xor_clauses[0]

    with pytest.raises(IndexError):
        xor_clauses[3]
    with pytest.raises(IndexError):
        xor_clauses[-4]


def test_xor_clause_list_operators():
    xor_clauses = XorClauseList()
    xor_clauses += XorClauseList([-1, 2, -3, 0, 4, -5, 6, 0])
    xor_clauses += XorClauseList([1, 2, 3, 0])

    assert len(xor_clauses) == 3
    assert xor_clauses[0] == XorClause([-1, 2, -3])
    assert xor_clauses[1] == XorClause([4, -5, 6])
    assert xor_clauses[2] == XorClause([1, 2, 3])

    assert xor_clauses[1] in xor_clauses
    assert Clause(xor_clauses[1]) not in xor_clauses
    assert xor_clauses.count(Clause(xor_clauses[1])) == 0
    assert xor_clauses.count(XorClause(xor_clauses[1])) == 1

    assert xor_clauses.index(XorClause(xor_clauses[1])) == 1
    assert xor_clauses.index(XorClause(xor_clauses[2])) == 2
    with pytest.raises(ValueError):
        xor_clauses.index(Clause(xor_clauses[1]))


def test_xor_clause_list_iter():
    cnf = XorClauseList([1, 2, 3, 0, 4, 5, 6, 0])
    cnf2 = XorClauseList()

    for clause in cnf:
        cnf2.add_clause(clause)

    assert cnf2 == cnf

def test_xor_clause():
    xor_clause = XorClause([1, 2, 3])
    assert len(xor_clause) == 3
    assert xor_clause[0] == 1
    assert xor_clause[1] == 2
    assert xor_clause[2] == 3

    assert isinstance(xor_clause, collections.abc.Sequence)

    assert XorClause([1, 2, 3]) == XorClause([1, 2, 3])
    assert XorClause([1, 2, 3]) != Clause([1, 2, 3])
    assert Clause([1, 2, 3]) != XorClause([1, 2, 3])

    with pytest.raises(IndexError):
        xor_clause[3]

    with pytest.raises(IndexError):
        xor_clause[-4]

    with pytest.raises(ValueError):
        XorClause([1, 2, 3, 0])
