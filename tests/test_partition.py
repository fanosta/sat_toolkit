import random

from sat_toolkit.formula import CNF, Clause, XorClause, XorClauseList, XorCNF

from icecream import ic


def test_partition():
    cnf = CNF([1, 2, 3, 0, -4, 5, 6, -7, 0, 8, 9, 0, -9, -10, 0])
    partitions = cnf.partition()

    for partition in partitions:
        ic(partition)
        assert isinstance(partition, CNF)

    assert len(partitions) == 3
    assert partitions[0] == CNF([1, 2, 3, 0], cnf.nvars)
    assert partitions[1] == CNF([-4, 5, 6, -7, 0], cnf.nvars)
    assert partitions[2] == CNF([8, 9, 0, -9, -10, 0], cnf.nvars)

def test_shuffled_xors():
    random.seed("test_shuffled_xors")

    xor1 = CNF.create_xor([1], [2], [3], [4])
    xor1.nvars = 8
    xor2 = CNF.create_xor([5], [6], [7], [8])

    cnf = CNF()
    cnf += xor1
    cnf += xor2

    cnf_list = list(cnf)
    assert len(cnf_list) == 16
    random.shuffle(cnf_list)
    cnf = CNF()
    for clause in cnf_list:
        clause_list = list(clause)
        # random.shuffle(clause_list)
        cnf.append(clause_list)

    partitions = cnf.partition()
    assert len(partitions) == 2

    ic(partitions[0])
    ic(partitions[1])

    assert len(partitions[0]) == len(partitions[1]) == len(xor1) == len(xor2)

    assert (
        partitions[0].equiv(xor1)
        or partitions[0].equiv(xor2)
    )
    assert (
        partitions[1].equiv(xor1)
        or partitions[1].equiv(xor2)
    )
    assert not partitions[0].equiv(partitions[1])

def test_late_merge():
    cnf = CNF()

    cnf += CNF.create_xor([1], [2], [3], [4])
    cnf += CNF.create_xor([5], [6], [7], [8])
    cnf += CNF.create_xor([9], [10], [11], [12])

    cnf += CNF([1, 5, 9, 0])

    partitions = cnf.partition()
    for partition in partitions:
        ic(partition)

    assert len(partitions) == 1
    assert partitions[0] == cnf

def test_add_after_merge():
    cnf = CNF()

    cnf += CNF([1, 2, 0])
    cnf += CNF([3, 4, 0])
    cnf += CNF([2, 3, 0])
    cnf += CNF([1, 0])
    cnf += CNF([4, 0])

    parts = cnf.partition()
    for part in parts:
        print(part, end="\n\n")

    assert len(parts) == 1

def test_multiple_merges():
    cnf = CNF()

    cnf += CNF.create_xor([1], [2], [3])
    cnf += CNF.create_xor([4], [5], [6])
    cnf += CNF.create_xor([7], [8], [9])
    cnf += CNF.create_xor([10], [11], [12])

    cnf += CNF([1, 4, 0])
    cnf += CNF([-1, 7, 0])
    cnf += CNF([-7, -11, 0])

    partitions = cnf.partition()
    for partition in partitions:
        ic(partition)

    assert len(partitions) == 1
    assert partitions[0].equiv(cnf)


def test_empy_clauses():
    xor1 = CNF.create_xor([1], [2], [3], [4])
    xor2 = CNF.create_xor([5], [6], [7], [8])

    empty_clauses = CNF([0, 0])

    xor1.nvars = 8
    xor2.nvars = 8
    empty_clauses.nvars = 8

    cnf = xor1 + xor2 + empty_clauses

    partitions = cnf.partition()
    assert len(partitions) == 3
    assert sum(p == xor1 for p in partitions) == 1
    assert sum(p == xor2 for p in partitions) == 1
    assert sum(p == empty_clauses for p in partitions) == 1
