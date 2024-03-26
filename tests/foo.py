#!/usr/bin/env python3

from sat_toolkit.formula import CNF
import random

from icecream import ic

if __name__ == "__main__":
    cnf = CNF()
    cnf.add_clause([1])

    cnf.add_clause([2, 3])
    cnf.add_clause([4, 6, 5])
    cnf.add_clause([-2, -3])

    cnf.add_clause([4, -5])
    cnf.add_clause([5, -4])
    cnf.add_clause([7, 1, 6, 5])

    cnf.add_clause([6, 7])
    cnf.add_clause([6, -7])

    cnf += CNF.create_xor([10], [20], [30])
    cnf += CNF.create_xor([10], [20], [25])
    cnf += CNF.create_xor([40], [20], [30])
    cnf += CNF.create_xor([10], [20], [35], rhs=[1])

    cnf_list = list(cnf)

    random.seed(1337)
    # random.shuffle(cnf_list)
    cnf = CNF()
    for clause in cnf_list:
        clause_list = list(clause)
        # random.shuffle(clause_list)
        cnf.add_clause(clause_list)
    print(cnf)

    cnf_copy = CNF(cnf)
    ic(cnf_copy == cnf)

    xor_cnf = cnf._to_xor_cnf()

    ic(cnf_copy == cnf)
    ic(cnf.equiv(cnf_copy))

    print(xor_cnf)
    assert xor_cnf.to_cnf().equiv(cnf)
