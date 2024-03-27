import collections
import numpy as np
import pytest
import copy

from sat_toolkit.formula import CNF, Clause, XorClause, XorClauseList, XorCNF

from icecream import ic

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

def test_nvars():
    cnf = XorCNF()
    assert cnf.nvars == 0

    with pytest.raises(ValueError):
        cnf.nvars = -1

    cnf.nvars = 0
    cnf.nvars = 1
    assert cnf.nvars == 1

    cnf.add_clauses([1, 2, -3, 0])
    assert cnf.nvars == 3

    with pytest.raises(ValueError):
        cnf.nvars = 2

    cnf.add_xor_clauses([10, 20, -30, 0])

    with pytest.raises(ValueError):
        cnf.nvars = 10
    with pytest.raises(ValueError):
        cnf.nvars = 25

    cnf.nvars = 30
    assert cnf.nvars == 30
    cnf.nvars = 40
    assert cnf.nvars == 40



def test_xor_cnf():
    xor_cnf = XorCNF()
    xor_cnf += CNF([1, -2, 3, 0, -4, 5, -6, 0])

    xor_cnf += XorClauseList([1, 3, 6, 0, -2, 4, 5, 0])

    dimacs = xor_cnf.to_dimacs()

    with pytest.raises(ValueError):
        # cannot alter while refeerenced by temoprary buffer
        xor_cnf += xor_cnf

    xor_cnf += copy.copy(xor_cnf)

    dimacs = xor_cnf.to_dimacs()

    assert dimacs == ("p cnf 6 8\n"
                      "1 -2 3 0\n"
                      "-4 5 -6 0\n"
                      "1 -2 3 0\n"
                      "-4 5 -6 0\n"
                      "x1 3 6 0\n"
                      "x-2 4 5 0\n"
                      "x1 3 6 0\n"
                      "x-2 4 5 0\n")

def test_equal():
    a = XorCNF()

    a += CNF([1, 2, 3, 0], nvars=6)

    a += XorClauseList([1, 3, 6, 0])

    ic(a._clauses.nvars)
    ic(a._xor_clauses.nvars)


    b = XorCNF()
    b += CNF([1, 2, 3, 0], nvars=3)
    b += XorClauseList([1, 3, 6, 0])

    assert a.nvars == 6
    assert b.nvars == 6
    assert a == b

def test_equal_2():
    a = XorCNF()
    a._clauses.add_clauses(CNF([1, 2, 3, 0], nvars=6))
    a._xor_clauses.add_clauses(XorClauseList([1, 3, 6, 0]))

    b = XorCNF()
    b._clauses.add_clauses(CNF([1, 2, 3, 0], nvars=3))
    b._xor_clauses.add_clauses(XorClauseList([1, 3, 6, 0]))

    assert a._clauses.nvars == 6
    assert a._xor_clauses.nvars == 6

    assert b._clauses.nvars == 3
    assert b._xor_clauses.nvars == 6

    assert a == b


def test_dimacs():
    xor_cnf = XorCNF()
    xor_cnf += CNF([1, -2, 3, 0, -4, 5, -6, 0])

    xor_cnf += XorClauseList([1, 3, 6, 0, -2, 4, 5, 0])

    dimacs = xor_cnf.to_dimacs()

    assert xor_cnf.nvars == 6
    assert dimacs == ("p cnf 6 4\n"
                      "1 -2 3 0\n"
                      "-4 5 -6 0\n"
                      "x1 3 6 0\n"
                      "x-2 4 5 0\n")

    recovered_cnf = XorCNF.from_dimacs(dimacs)
    assert recovered_cnf == xor_cnf

    with pytest.raises(ValueError):
        XorCNF.from_dimacs("")
    with pytest.raises(ValueError):
        XorCNF.from_dimacs("p cnf 6 x\n")
    with pytest.raises(ValueError):
        XorCNF.from_dimacs("p cnf x 0\n")
    with pytest.raises(ValueError):
        XorCNF.from_dimacs("p cnf 6 0\np cnf 6 0\n")



def test_incompatible_type():
    xor_list = XorClauseList([1, 2, 3, 0, -4, 5, 6, 0])
    xor_list.add_clause([7, 8, -9])

    xor_list.add_clause(XorClause([1, -3, 5]))

    with pytest.raises(TypeError):
        xor_list.add_clause(Clause([2, -4, 6]))

    cnf = CNF([1, 2, 0, -4, 5, 6, 8, 0])
    with pytest.raises(TypeError):
        xor_list += cnf


def test_pickle():
    import pickle

    xors1 = XorClauseList([1, 2, 3, 0, 4, 5, 6, 0])
    xors2 = XorClauseList([1, 2, 3, 0, 4, 5, 6, 0], nvars=10)

    xors1_pickled = pickle.dumps(xors1)
    xors2_pickled = pickle.dumps(xors2)

    xors1_loaded = pickle.loads(xors1_pickled)
    xors2_loaded = pickle.loads(xors2_pickled)

    assert xors1 == xors1_loaded
    assert xors2 == xors2_loaded
    assert xors2.nvars == xors2_loaded.nvars == 10

    xor_cnf = XorCNF()
    assert xor_cnf.nvars == 0


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

    tmp_clauses =  XorClauseList([3, -7, -12, 0], 14)
    assert tmp_clauses.nvars == 14
    xor_clauses += tmp_clauses
    assert xor_clauses.nvars == 14

    xor_clauses += XorClauseList([3, -7, -16, 0])
    assert xor_clauses.nvars == 16


def test_xor_cnf_operators():
    cnf1 = XorCNF([1, 2, 3, 0], [])
    cnf2 = XorCNF([], [1, 3, 0])

    assert cnf1 + cnf2 == XorCNF([1, 2, 3, 0], [1, 3, 0])
    cnf1 += cnf2
    assert cnf1 == XorCNF([1, 2, 3, 0], [1, 3, 0])

    assert cnf1 + CNF([5,6,7,0]) == XorCNF([1, 2, 3, 0, 5, 6, 7, 0], [1, 3, 0])
    assert cnf1 + XorClauseList([5,6,7,0]) == XorCNF([1, 2, 3, 0], [1, 3, 0, 5, 6, 7, 0])


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

def test_to_cnf():
    cnf_equal = XorCNF([], [-1, 2, 0]).to_cnf()
    assert len(cnf_equal) == 2
    assert Clause([1, -2]) in cnf_equal
    assert Clause([-1, 2]) in cnf_equal

    cnf_not_equal = XorCNF([], [1, 2, 0]).to_cnf()
    assert len(cnf_not_equal) == 2
    assert Clause([1, 2]) in cnf_not_equal
    assert Clause([-1, -2]) in cnf_not_equal

    assert cnf_not_equal.equiv(XorCNF([], [1, 2, 0]).to_cnf())

    # with pytest.raises(ValueError):
    #     XorCNF([], [0]).to_cnf()

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

def test_create_xor():
    cnf_equal = XorCNF.create_xor(np.array([1]), np.array([2], dtype=np.int32)).to_cnf()
    assert len(cnf_equal) == 2
    assert Clause([1, -2]) in cnf_equal
    assert Clause([-1, 2]) in cnf_equal

    cnf_not_equal = XorCNF.create_xor([1], [2], rhs=np.array([1])).to_cnf()
    assert len(cnf_not_equal) == 2
    assert Clause([1, 2]) in cnf_not_equal
    assert Clause([-1, -2]) in cnf_not_equal

    assert cnf_not_equal.equiv(XorCNF.create_xor([-1], [2]).to_cnf())
    assert cnf_not_equal.equiv(XorCNF.create_xor([1], [-2]).to_cnf())

    with pytest.raises(ValueError):
        XorCNF.create_xor(rhs=[1])

    cnf_xor3 = XorCNF.create_xor([1], [2], [3]).to_cnf()
    assert len(cnf_xor3) == 4
    assert Clause([1, 2, -3]) in cnf_xor3
    assert Clause([1, -2, 3]) in cnf_xor3
    assert Clause([-1, 2, 3]) in cnf_xor3
    assert Clause([-1, -2, -3]) in cnf_xor3

    with pytest.raises(ValueError):
        XorCNF.create_xor(rhs=[1])
    with pytest.raises(ValueError):
        XorCNF.create_xor([0], [1])
    with pytest.raises(ValueError):
        XorCNF.create_xor([1, 2], [3, 0])
    with pytest.raises(ValueError):
        XorCNF.create_xor([0])

    cnf_xor3_multi = XorCNF.create_xor([1, 4], [2, 5], [3, 6]).to_cnf()
    assert len(cnf_xor3_multi) == 8
    assert Clause([1, 2, -3]) in cnf_xor3_multi
    assert Clause([1, -2, 3]) in cnf_xor3_multi
    assert Clause([-1, 2, 3]) in cnf_xor3_multi
    assert Clause([-1, -2, -3]) in cnf_xor3_multi

    assert Clause([4, 5, -6]) in cnf_xor3_multi
    assert Clause([4, -5, 6]) in cnf_xor3_multi
    assert Clause([-4, 5, 6]) in cnf_xor3_multi
    assert Clause([-4, -5, -6]) in cnf_xor3_multi

def test_solve_dimacs():
    cnf = XorCNF.create_xor([1], [2], [3])
    cnf += XorCNF([], [1, 3, 0])
    cnf += CNF([1, 2, 3, 0])
    cnf += CNF([1, -2, 0])

    for seed in range(10):
        args = ['cryptominisat5', f'--random={seed}', '--polar=rnd']
        is_sat, result = cnf.solve_dimacs(args)
        assert is_sat

        assert result[1] ^ result[2] ^ result[3] == 0
        assert result[1] ^ result[3] == 1

        assert result[1] or result[2] or result[3] == 1
        assert result[1] or not result[2] == 1
