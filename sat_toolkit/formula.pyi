from __future__ import annotations
from typing import Literal, Iterable, Iterator, overload, Self
import io
from collections import abc

import numpy as np
import numpy.typing as npt

class _BaseClause(abc.Sequence):
    _clause: list[int]

    def __init__(self, clause: Iterable[int]):
        ...

    def to_linear_constraint(self, variables: list):
        """
        return the current clause as a linear constraint suitable for MILP modeling.
        the resulting constraint is suitable if all variables are constrained to [0, 1].

        :param variables:   List of MILP (or similar) variables with overloaded
                            __add__, __neg__, and __geq__. The list is shifted compared to the
                            indexing, i.e. variables[0] correspsonds to index 1 in the Clause.
        :return:    a linear constraint over variables
        """
        ...

    @property
    def maxvar(self) -> int:
        ...

    def __getitem__(self, idx: int) -> int:
        ...

    def __len__(self) -> int:
        ...

    def __contains__(self, needle: int) -> bool:
        ...

    def __iter__(self) -> Iterator[int]:
        ...

    def __reversed__(self) -> Iterator[int]:
        ...

    def count(self, needle) -> int:
        "Return the number of times needle appears in the list."
        ...

    def index(self, needle: int, start=None, end=None) -> int:
        """
        Return zero-based index in the list of the first item whose value is
        equal to needle. Raises a ValueError if there is no such item.

        The optional arguments start and end are interpreted as in the slice
        notation and are used to limit the search to a particular subsequence
        of the list. The returned index is computed relative to the beginning
        of the full sequence rather than the start argument.
        """
        ...

    @overload
    def __eq__ (self, other: Self) -> bool:
        ...
    @overload
    def __eq__ (self, other) -> Literal[False]:
        ...

class Clause(_BaseClause):
    ...


class XorClause(_BaseClause):
    ...


class _ClauseList[T: _BaseClause](abc.Sequence):

    def __init__(self, clauses: Self|npt.ArrayLike|None=None, nvars: int = -1):
        ...

    @property
    def _clauses(self) -> list[int]:
        ...

    @property
    def _start_indices(self) -> list[int]:
        ...


    @property
    def nvars(self) -> int:
        ...

    @nvars.setter
    def nvars(self, val: int):
        ...

    def get_vars(self) -> set[int]:
        """
        returns the set of all variables used in the clauses of the CNF.
        """
        ...

    def clear(self) -> None:
        """
        Remove all clauses from the CNF and set the number of variables to 0.
        """
        self._clauses.clear()
        self._start_indices.clear()
        self.nvars = 0

    def partition(self) -> list[Self]:
        """
        partition the CNF/XorClauseList into a list of CNF/XorClauseLists with disjoints variables.
        """
        ...

    def translate(self, mapping: npt.ArrayLike) -> Self:
        """
        Translate all variables in the CNF/XorClauseList to a new index.
        The translation mapping is given by the mapping paramter.

        Index 0 must always map to index 0 again.

        :return: a new CNF with variables changed according to mapping parameter
        """
        ...

    def add_clauses(self, clauses: Self|npt.ArrayLike) -> None:
        ...

    def add_clause(self, clause: T|npt.ArrayLike) -> None:
        """Add a single clause to CNF formula. Specify the clause without trailing 0."""
        ...

    append = add_clause

    def __iadd__(self, clauses: Self|npt.ArrayLike) -> Self:
        ...

    def __add__(self, other: Self) -> Self:
        ...

    def __getitem__(self, idx: int) -> T:
        ...

    def __len__(self) -> int:
        ...

    def __iter__(self) -> Iterator[T]:
        ...


    def __reversed__(self) -> Iterator[T]:
        ...


    @overload
    def __eq__ (self, other: Self) -> bool:
        ...

    @overload
    def __eq__ (self, other) -> Literal[False]:
        ...

    @overload
    def __contains__(self, needle: T) -> bool:
        ...

    @overload
    def __contains__(self, needle) -> Literal[False]:
        ...

    @overload
    def count(self, needle: T) -> int:
        ...

    @overload
    def count(self, needle) -> Literal[0]:
        ...


    @overload
    def index(self, needle: T, start: int|None=None, end: int|None=None) -> int:
        ...

    @overload
    def index(self, needle, start: int|None=None, end: int|None=None) -> Literal[0]:
        ...

    def index(self, needle, start=None, end=None):
        """
        Return zero-based index in the list of the first item whose value is
        equal to needle. Raises a ValueError if there is no such item.

        The optional arguments start and end are interpreted as in the slice
        notation and are used to limit the search to a particular subsequence
        of the list. The returned index is computed relative to the beginning
        of the full sequence rather than the start argument.
        """
        ...

    def __reduce__(self):
        ...

    # copy support
    def __copy__(self) -> Self:
        ...

    def copy(self) -> Self:
        ...


class CNF(_ClauseList[Clause]):
    """
    Class for storing and manipulating CNF formulas.

    The CNF is represented in a format closely related to the DIMACS format.
    A CNF is conjuction (logical AND) of clauses.
    Each clause (logical AND) is a list of variables that ends with a 0.
    A positive number indicates the presence of a variable in the clause while
    a negative number indicates the presence of the negated varaible in the
    clause.

    For example, the CNF (x1 or not x2) and (x2 or x3) can be represented as
    CNF([1, -2, 0, 2, 3, 0]).
    """

    @staticmethod
    def from_dimacs(dimacs: str) -> CNF:
        ...

    @staticmethod
    def from_espresso(espresso: str) -> CNF:
        ...

    @staticmethod
    def create_all_zero(indices: npt.ArrayLike) -> CNF:
        """
        creates a CNF that asserts that all variables for the provided indices
        are zero.
        """
        ...

    @staticmethod
    def _create_all_zero(indices: memoryview) -> CNF:
        ...


    @staticmethod
    def create_all_equal(lhs: npt.ArrayLike, rhs: npt.ArrayLike) -> CNF:
        "creates a CNF that asserts lhs[i] == rhs[i] for all i."
        ...


    @staticmethod
    def _create_all_equal(lhs: memoryview, rhs: memoryview) -> CNF:
        ...

    @staticmethod
    def create_xor(*args, rhs=None) -> CNF:
        """
        creates a CNF specifying the xor of the arguments is equal to the right
        hand side (rhs).

        If rhs is None, the default value of 0 is used.

        Each argument is a 1-D array. The xors are computed elementwise.
        """
        ...


    def logical_or(self, var: int) -> CNF:
        """
        return a new CNF corresponding to (var or self). I.e., the new CNF is
        build by appending var to each clause

        >>> cnf = CNF([1,2,0, -1,-2,0])
        >>> print(cnf.logical_or(5))
        p cnf 5 2
        1 2 5 0
        -1 -2 5 0


        >>> cnf = CNF([1,2,0, -1,-2,0])
        >>> print(cnf.logical_or(-5))
        p cnf 5 2
        1 2 -5 0
        -1 -2 -5 0
        """
        ...

    def implied_by(self, var: int) -> CNF:
        """
        return a new CNF corresponding to (var -> self). I.e., the new CNF is
        build by appending ~var to each clause
        """
        ...

    def get_units(self) -> np.ndarray:
        """
        Returns all unit clauses in the CNF. The unit clauses are returned as
        a numpy array without the separating zeros.

        >>> cnf = CNF([1,0, 1,2,3,0, -3,0, -1,-2,0, -2,0])
        >>> cnf.get_units()
        array([ 1, -3, -2], dtype=int32)
        """
        ...


    def to_dimacs(self) -> str:
        '''
        return the CNF formatted in the DIMACS file format
        '''
        ...

    def to_espresso(self, print_numvars: bool = True) -> str:
        '''
        return the CNF formatted in the espresso file format
        '''
        ...

    def minimize_espresso(self, espresso_args: list[str] = []) -> CNF:
        """
        Uses espresso to minimize the given CNF.

        :param espresso_args: extra parameters given when calling espresso, defaults to []

        :return: a new CNF object as minimized by espresso
        :rtype: CNF
        """
        ...

    def _minimize_dimacs(self, args: list[str], outfile: str) -> CNF:
        """
        calls args with self.to_dimacs() as stdin, waits for the command to
        finish with exit code 10 or 20 and parses `outfile` as DIAMCS`.

        :return: a new CNF object read from `outfile`
        :rtype: CNF
        """
        ...

    def minimize_lingeling(self, optlevel=None, timeout=0, extra_args: list[str] = []) -> CNF:
        """
        Uses Lingeling to minimize the given CNF.

        :param optlevel: optimization level given to Lingeling via -O<optlevel>
        :param timeout: timeout given to Lingeling via -T <timeout>
        :param extra_args: extra parameters given when calling Lingeling, defaults to []

        :return: a new CNF object minimized by Lingeling
        :rtype: CNF
        """
        ...

    def solve_dimacs(self, command: list[str], verbose=False) -> tuple[Literal[True], np.ndarray] | tuple[Literal[False], None]:
        """
        solves the SAT by calling a DIMACS the compliant sat solver given by command.

        Returns (True, np.array(model, dtype=np.uint8)) for SAT instances.
        Returns (False, None) for UNSAT instances.
        """
        ...

    def check_solution(self, solution: memoryview) -> bool:
        ...

    def to_truthtable(self) -> Truthtable:
        """
        Return the current CNF as a Truthtable by iterating over all possible
        variable assignments.
        """
        ...

    def __repr__(self) -> str:
        ...

    def __str__(self) -> str:
        ...

    def equiv(self, other: CNF) -> bool:
        """
        Check for logical eqivalence between self and other.
        Calls `espresso -Dverify` to perform the comparison.
        """
        ...


class XorClauseList(_ClauseList[XorClause]):
    """
    Class for storing and manipulating a list of xor clauses.
    """

    def foo(self):
        reveal_type(self[0])
        reveal_type(Clause([1, 2, 3, 0]) in self)
        reveal_type(XorClause([1, 2, 3, 0]) in self)
        reveal_type([1, 2, 3, 0] in self)


class XorCNF:
    def __init__(self, clauses: CNF|npt.ArrayLike|None=None, xor_clauses: XorClauseList|npt.ArrayLike|None=None, nvars: int|None=None):
        ...

    @property
    def _clauses(self) -> CNF:
        ...

    @property
    def _xor_clauses(self) -> XorClauseList:
        ...

    @staticmethod
    def from_dimacs(dimacs: str) -> XorCNF:
        ...

    @staticmethod
    def create_xor(*args, rhs=None) -> XorCNF:
        """
        creates an XorCNF specifying the xor of the arguments is equal to the right
        hand side (rhs).

        If rhs is None, the default value of 0 is used. This is in contrast to
        the internal representation, where the default value is 1.

        Each argument is a 1-D array. The xors are computed elementwise.
        """
        ...

    def partition(self) -> tuple[list[CNF], list[XorClauseList]]:
        """
        Separately partition the CNF and XorClauseList into a lists of CNFs and
        XorClauseLists with disjoint variables.

        returns a tuple of two lists, the first containing CNFs and the second
        containing XorClauseLists.
        """
        ...

    def translate(self, mapping) -> XorCNF:
        """
        Translate all variables in the XorCNF to a new index.
        The translation mapping is given by the mapping paramter.

        Index 0 must always map to index 0 again.

        :return: a new XorCNF with variables changed according to mapping parameter
        """
        ...

    def to_cnf(self) -> CNF:
        ...

    def add_clauses(self, clauses: CNF|npt.ArrayLike) -> None:
        ...

    def add_xor_clauses(self, xor_clauses: XorClauseList|npt.ArrayLike) -> None:
        ...

    @property
    def nvars(self) -> int:
        ...

    @nvars.setter
    def nvars(self, val: int):
        ...

    @property
    def nclauses(self) -> int:
        """returns the number of CNF clauses (excluding xor clauses)"""
        ...

    @property
    def nxor_clauses(self) -> int:
        """returns the number of xor clauses (excluding CNF clauses)"""
        ...

    def __repr__(self) -> str:
        ...

    def __str__(self) -> str:
        ...

    def __iadd__(self, other: CNF|XorClauseList|XorCNF) -> Self:
        ...

    def __add__(self, other: CNF|XorClauseList|XorCNF) -> Self:
        ...

    @overload
    def __contains__(self, needle: Clause|XorClause) -> bool:
        ...

    @overload
    def __contains__(self, needle) -> Literal[False]:
        ...

    def to_dimacs(self) -> str:
        '''
        return the XorCNF formatted in the extended DIMACS file format
        '''
        ...

    # pickle support
    def __reduce__(self):
        ...

    # copy support
    def __copy__(self) -> Self:
        ...

        reveal_type(Clause([1, 23, 0]) in self)
        reveal_type(XorClause([1, 23, 0]) in self)
        reveal_type([1, 23, 0] in self)

    def copy(self) -> Self:
        ...

    def __eq__ (self, other) -> bool:
        ...


    def solve_dimacs(self, command: list[str]=['cryptominisat5'], verbose=False) -> tuple[Literal[True], np.ndarray] | tuple[Literal[False], None]:
        """
        solves the SAT by calling a DIMACS compliant sat solver that also
        supports XORs given by command. The solver defaults to cryptominisat5.
        For other solvers that do not support XORs,
        self.to_cnf().solve_dimacs() can be used.

        Returns (True, np.array(model, dtype=np.uint8)) for SAT instances.
        Returns (False, None) for UNSAT instances.
        """
        ...


class Truthtable:
    """
    A boolean function represented as a truth table.
    The truth table is stored using a table for the ON set and the DC set of the function.
    The ON set denotes where the value of the function is 1/true, while the DC set
    denotes where the value of the function is left unspecified.
    The DC set allows more efficient CNF representations when optimizing with espresso.
    The OFF set is the complement of the two sets.
    """
    on_set: np.ndarray
    dc_set: np.ndarray
    numbits: int

    @classmethod
    def from_lut(cls, on_lut: np.ndarray, dc_lut: np.ndarray|None = None) -> Self:
        """
        Creates a Truthtable from a lookup table (LUT).

        :param on_lut: specifies where the Truthtable should be on, i.e., 1
        :param dc_lut: specifies where the value of the Truthtable can be ignored, this allows for better optimization, defaults to None

        :return: Truthtable with on_set and dc_set initialized according to parameters
        :rtype: Truthtable
        """
        ...

    @classmethod
    def from_indices(cls, numbits: int, on_indices: np.ndarray, dc_indices = np.array([], dtype=int)) -> Truthtable:
        """
        Creates a Truthtable from the set of indices with value 1 and optionally indices where the value can be ignored.

        :param numbits: number of input bits for the Truthtable
        :param on_indices: set of indices where the Truthtable should be on, i.e., 1
        :param dc_indices: set of indices where the Truthtable value can be ignored, this allows for better optimization, defaults to []

        :return: Truthtable with on_set and dc_set initialized according to parameters
        :rtype: Truthtable
        """
        ...


    def _write(self, io: io.TextIOBase, espresso: bool = False, invert: bool = False) -> None:
        ...

    def to_espresso(self, phase='cnf') -> str:
        """
        return the truthtable in espresso format

        :param phase: wether to minimize the off_set ('cnf'/0) or on_set ('dnf'/1)

        :return: Truthtable with on_set and dc_set initialized according to parameters
        :rtype: Truthtable
        """
        ...

    def __repr__(self) -> str:
        ...

    def __eq__(self, other) -> bool:
        ...

    def to_cnf(self, espresso_args: list[str] = []) -> CNF:
        """
        Uses espresso to convert the Truthtable to a minimized CNF.

        The resulting CNF will be indexed by [1, self.numbits], where index 1
        corresponds to the least significant bit in the truthtable index.

        :param espresso_args: extra parameters given when calling espresso, defaults to []

        :return: CNF minimized by espresso
        :rtype: CNF
        """
        ...

abc.Sequence.register(_BaseClause)
abc.Sequence.register(_ClauseList)
