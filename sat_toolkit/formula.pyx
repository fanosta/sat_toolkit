#cython: language_level=3, annotation_typing=True, embedsignature=True, boundscheck=False, wraparound=False, cdivision=True
#distutils: language = c++
cimport cython
from cython.view cimport array as cvarray
from cpython cimport buffer

from cpython.buffer cimport PyBUF_FORMAT, PyBUF_ND, PyBUF_STRIDES, PyBUF_WRITABLE
from cpython.exc cimport PyErr_CheckSignals
from libcpp.vector cimport vector
from libc.stdio cimport printf, snprintf, sscanf
from libc.stdlib cimport malloc, free
from libc.string cimport strcpy, memset, strncmp, strlen, memcmp
from sat_toolkit.types cimport *

cdef extern from "fmt.h":
    ctypedef struct fmt_buf:
        size_t len, capacity
        char *buf

    fmt_buf make_buf()
    void free_buf(fmt_buf *buf)
    int buf_printf(fmt_buf *buf, const char *fmt, ...)


cdef extern from *:
    """
    #define ffs(x) __builtin_ffsll(x)
    #define popcount(x) __builtin_popcountll(x)
    """
    int ffs(int x) nogil
    int popcount(unsigned int x) nogil
    int likely(int) nogil
    int unlikely(int) nogil


import io, sys
import numpy as np
import subprocess as sp
from tempfile import NamedTemporaryFile
import collections

from typing import TYPE_CHECKING
# if TYPE_CHECKING:
#     from typing import *

from dataclasses import dataclass

__all__ = ['Clause', 'CNF', 'Truthtable']

cdef int startswith(const unsigned char[:] buf, const char *start) noexcept nogil:
    cdef ssize_t l = strlen(start)
    if buf.shape[0] < l:
        return 0
    return strncmp(<const char *> &buf[0], <const char *> &start[0], l) == 0

cdef int abs(int val) noexcept nogil:
    if val > 0:
        return val
    return -val


cdef class Clause:
    cdef vector[int] clause
    def __init__(self, clause):
        cdef int c
        for c in clause:
            if c == 0:
                raise ValueError('cannot use 0 as part of clause')
            self.clause.push_back(c)
        self.clause.shrink_to_fit()

    @staticmethod
    cdef Clause from_memview(const int[:] clause):
        cdef size_t i
        cdef Clause res = Clause.__new__(Clause)

        res.clause.resize(clause.shape[0])
        for i in range(res.clause.size()):
            if clause[i] == 0:
                raise ValueError('cannot use 0 as part of clause')
            res.clause[i] = clause[i]

        return res

    def to_linear_constraint(self, variables: list):
        """
        return the current clause as a linear constraint suitable for MILP modeling.
        the resulting constraint is suitable if all variables are constrained to [0, 1].

        :param variables:   List of MILP (or similar) variables with overloaded
                            __add__, __neg__, and __geq__. The list is shifted compared to the
                            indexing, i.e. variables[0] correspsonds to index 1 in the Clause.
        :return:    a linear constraint over variables
        """
        cdef size_t i
        cdef int c

        cdef object lhs = 0
        cdef int rhs = 1

        if <size_t> len(variables) < self._maxvar():
            raise IndexError(f'too few variables supplied')

        for i in range(self.clause.size()):
            c = self.clause[i]
            var = variables[abs(c) - 1]
            if c > 0:
                lhs += var
            if c < 0:
                lhs += -var
                rhs -= 1

        return lhs >= rhs

    cdef size_t _maxvar(self):
        cdef size_t i
        cdef int c, res = 0
        for i in range(self.clause.size()):
            c = abs(self.clause[i])
            if c > res:
                res = c
        return res

    @property
    def maxvar(self) -> int:
        return self._maxvar()

    def __repr__(self) -> str:
        return f'Clause({self.clause})'

    cdef size_t _get_absolute_index(self, ssize_t idx) except -1:
        if idx < 0:
            idx += self.clause.size()
        if idx < 0 or <size_t> idx >= self.clause.size():
            raise IndexError('index out of range')
        return <size_t> idx

    cdef size_t _get_slice_index(self, ssize_t idx):
        if idx < 0:
            idx += self.clause.size()
        if idx < 0:
            return 0
        if <size_t> idx >= self.clause.size():
            return self.clause.size()
        return <size_t> idx


    def __getitem__(self, ssize_t idx) -> int:
        return self.clause[self._get_absolute_index(idx)]

    def __len__(self) -> int:
        return self.clause.size()

    def __contains__(self, int needle) -> bint:
        cdef size_t i
        cdef int c

        for i in range(self.clause.size()):
            if self.clause[i] == needle:
                return True

        return False

    def __iter__(self) -> Iterable[int]:
        cdef size_t i = 0
        for i in range(self.clause.size()):
            yield self.clause[i]

    def __reversed__(self) -> Iterable[int]:
        cdef size_t i = 0
        for i in reversed(range(self.clause.size())):
            yield self.clause[i]

    def count(self, int needle) -> int:
        "Return the number of times needle appears in the list."
        cdef size_t i
        cdef int c
        cdef size_t count = 0

        for i in range(self.clause.size()):
            if self.clause[i] == needle:
                count += 1

        return count

    def index(self, int needle, start=None, end=None) -> int:
        """
        Return zero-based index in the list of the first item whose value is
        equal to needle. Raises a ValueError if there is no such item.

        The optional arguments start and end are interpreted as in the slice
        notation and are used to limit the search to a particular subsequence
        of the list. The returned index is computed relative to the beginning
        of the full sequence rather than the start argument.
        """
        cdef size_t i, start_idx = 0, end_idx = self.clause.size()

        if start is not None:
            start_idx = self._get_slice_index(start)
        if end is not None:
            end_idx = self._get_slice_index(end)

        for i in range(start_idx, end_idx):
            if self.clause[i] == needle:
                return i

        raise ValueError(f'{needle} is not in Clause')

    def __eq__(self, other) -> bool:
        cdef Clause other_clause
        try:
            other_clause = other
        except TypeError:
            return False
        return self.clause == other_clause.clause

collections.abc.Sequence.register(Clause)


cdef class CNF:
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
    cdef readonly vector[int] clauses
    cdef readonly vector[size_t] start_indices

    cdef readonly int nvars

    #used for the buffer support
    cdef ssize_t shape[1]
    cdef ssize_t view_count

    def __init__(self, clauses = None, nvars = -1):
        if clauses is not None:
            self.add_clauses(clauses)
        if nvars != -1:
            if nvars < self.nvars:
                raise ValueError(f"explicitly given nvars too small ({nvars} < {self.nvars})")
            self.nvars = nvars

    def __cinit__(self):
        self.shape[0] = 0
        self.view_count = 0
        self.nvars = 0


    @staticmethod
    cdef CNF _from_dimacs(const uint8_t[::1] dimacs):
        cdef CNF res = CNF.__new__(CNF)
        cdef const char *buf = <const char *> &dimacs[0]

        cdef size_t off = -1, nclauses
        cdef int nvars, tmp

        while (sscanf(buf, "c %*[^\n]\n%zn", &off) or 1) and off != -1ULL:
            buf += off
            off = -1


        off = -1
        if (sscanf(buf, "p cnf %d %zd\n%zn", &nvars, &nclauses, &off) or 1) and off == -1ULL:
            raise ValueError('file format error')
        buf += off

        cdef size_t remaining_len = dimacs.shape[0] - (buf - <const char *> &dimacs[0])
        values = np.fromstring(buf[:remaining_len], dtype=np.int32, sep=' ')

        res._add_clauses(values)
        assert nvars >= res.nvars
        res.nvars = nvars
        return res

    @staticmethod
    def from_dimacs(dimacs: str) -> CNF:
        return CNF._from_dimacs(dimacs.encode())

    @staticmethod
    def from_espresso(espresso: str) -> CNF:
        cdef vector[int] clauses
        cdef vector[size_t] start_indices
        cdef const unsigned char[:] line
        cdef char val
        cdef uint64_t mask, sign
        cdef unsigned int numbits = 0
        cdef size_t i

        f = io.BytesIO(espresso.encode())

        for line_py in f:
            line = line_py.rstrip()
            if startswith(line, b'.i '):
                _, bits = line_py.split()
                numbits = int(bits)
            if startswith(line, b'.p '):
                ...
            if startswith(line, b'#') or startswith(line, b'.'):
                continue

            val = line[line.shape[0] - 1]
            if line.shape[0] != numbits + 2 or line[line.shape[0] - 2] != b' ':
                raise ValueError('file format error')

            #espresso_mask, val = line.split()

            if val != b'1':
                raise ValueError(f'got unexpected value for product term: {val!r} expected \'1\'.')

            for i in range(numbits):
                val = line[numbits - 1 - i]
                if val == b'-':
                    continue
                # 0/1 swapped according to De Morgan's law
                if val == b'0':
                    clauses.push_back((i + 1))
                    continue
                if val == b'1':
                    clauses.push_back(-(i + 1))
                    continue
                raise ValueError(f'file format error in line {bytes(line)}')
            clauses.push_back(0)

        cdef CNF res = CNF.__new__(CNF)
        if clauses.size() > 0:
            res._add_clauses(<int[:clauses.size()]> clauses.data())
        assert numbits >= <unsigned int> res.nvars
        res.nvars = numbits
        return res

    @staticmethod
    def create_all_zero(indices) -> CNF:
        """
        creates a CNF that asserts that all variables for the provided indices
        are zero.
        """
        indices = np.array(indices, dtype=np.int32, copy=False)
        return CNF._create_all_zero(indices)

    @staticmethod
    def _create_all_zero(const int[:] indices not None) -> CNF:
        cdef vector[int] clauses
        cdef size_t i, l
        cdef int val

        l = indices.shape[0]

        clauses.resize(l * 2)
        for i in range(l):
            val = indices[i]
            if val == 0:
                raise ValueError('all indices must be nonzero')
            clauses[2*i] = -val


        cdef CNF res = CNF.__new__(CNF)
        if clauses.size() > 0:
            res._add_clauses(<int[:clauses.size()]> clauses.data())
        return res

    @staticmethod
    def create_all_equal(lhs, rhs) -> CNF:
        "creates a CNF that asserts lhs[i] == rhs[i] for all i."
        lhs = np.array(lhs, dtype=np.int32, copy=False)
        rhs = np.array(rhs, dtype=np.int32, copy=False)
        return CNF._create_all_equal(lhs, rhs)

    @staticmethod
    def _create_all_equal(const int[:] lhs not None, const int[:] rhs not None) -> CNF:
        cdef vector[int] clauses
        cdef size_t i, length
        cdef int l, r

        length = lhs.shape[0]
        if rhs.shape[0] != length:
            raise ValueError('lhs and rhs must have the same length')

        clauses.resize(length * 6)
        for i in range(length):
            l = lhs[i]
            r = rhs[i]
            if l == 0 or r == 0:
                raise ValueError('all indices must be nonzero')

            clauses[6 * i + 0] = -l
            clauses[6 * i + 1] = r
            clauses[6 * i + 2] = 0

            clauses[6 * i + 3] = l
            clauses[6 * i + 4] = -r
            clauses[6 * i + 5] = 0

        cdef CNF res = CNF.__new__(CNF)
        if clauses.size() > 0:
            res._add_clauses(<int[:clauses.size()]> clauses.data())
        return res

    @staticmethod
    cdef CNF _create_xor(const int[:, ::1] packed_args):
        cdef CNF res = CNF.__new__(CNF)

        cdef size_t num_inputs, num_xors
        cdef size_t row, col, j
        cdef size_t mask
        cdef int rhs, var_idx

        num_xors = packed_args.shape[0]
        num_inputs = packed_args.shape[1] - 1

        cdef size_t needed_space = num_inputs * ((1 << (num_inputs - 1)) + 1)
        cdef int[::1] tmp_clause = np.empty(num_inputs + 1, np.int32)
        tmp_clause[num_inputs] = 0


        for row in range(num_xors):

            for col in range(num_inputs):
                if packed_args[row, col] <= 0:
                    raise ValueError("create_xor only supports positive indexes")

            rhs = packed_args[row, num_inputs]
            if rhs not in [0, 1]:
                raise ValueError(f"right hand side must be 0 or 1, not {rhs}")


            for mask in range(1 << num_inputs):
                if popcount(mask) % 2 == rhs:
                    continue

                for col in range(num_inputs):
                    var_idx = packed_args[row, col]
                    tmp_clause[col] = -var_idx if (mask >> col) & 1 else var_idx

                res._add_clauses(tmp_clause)

        return res

    @staticmethod
    def create_xor(*args, rhs=None) -> CNF:
        """
        creates a CNF specifying the xor of the arguments is equal to the right
        hand side (rhs).

        Each argument is a 1-D array. The xors are computed elementwise.
        """
        cdef size_t num_args = len(args)
        cdef size_t num_xors = -1
        cdef size_t i


        if num_args < 1:
            raise ValueError('must provide at least one argument')

        num_xors = len(args[0])

        packed_args = np.zeros((num_xors, num_args + 1), np.int32)

        for i, arg in enumerate(args):
            packed_args[:, i] = arg

        if rhs is not None:
            packed_args[:, num_args] = rhs

        return CNF._create_xor(packed_args)




    cdef int _add_clauses(self, const int[:] clauses) except -1 nogil:
        cdef size_t l, old_len, i

        if self.view_count > 0:
            raise ValueError('cannot alter while referenced by buffer')

        l = clauses.shape[0]
        if l > 0 and clauses[l - 1] != 0:
            raise ValueError('last clause not terminated with 0')

        old_len = self.clauses.size()
        self.clauses.resize(self.clauses.size() + l)

        if l > 0:
            self.start_indices.push_back(old_len)

        for i in range(l):
            self.clauses[old_len + i] = clauses[i]
            if abs(clauses[i]) > self.nvars:
                self.nvars = abs(clauses[i])

            if clauses[i] == 0 and i + 1 < l:
                self.start_indices.push_back(old_len + i + 1)

        return 0

    def add_clauses(self, clauses) -> None:
        cdef const int[:] clauses_view = None

        try:
            clauses_view = clauses
        except (TypeError, ValueError):
            np_clauses = np.array(clauses, copy=False, dtype=np.int32)
            clauses_view = np_clauses

        self._add_clauses(clauses_view)

    def add_clause(self, clause) -> None:
        """Add a single clause to CNF formula. Specify the clause without trailing 0."""
        cdef ssize_t clause_len = len(clause)
        np_clause = np.zeros(clause_len + 1, dtype=np.int32)
        np_clause[:clause_len] = clause
        if np.count_nonzero(np_clause) != clause_len:
            raise ValueError('cannot use 0 in clause')
        self._add_clauses(np_clause)

    append = add_clause

    def __iadd__(self, clauses) -> CNF:
        self.add_clauses(clauses)
        return self


    def logical_or(self, int var) -> CNF:
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
        if var == 0:
            raise ValueError("var must not be zero")

        cdef vector[int] new_clauses
        cdef ssize_t src = 0, dst = 0
        new_clauses.resize(self.clauses.size() + self.start_indices.size())

        for src in range(<ssize_t> self.clauses.size()):
            if self.clauses[src] == 0:
                new_clauses[dst] = var
                dst += 1
            new_clauses[dst] = self.clauses[src]
            dst += 1

        cdef CNF res = CNF.__new__(CNF)
        res._add_clauses(<int[:new_clauses.size()]> new_clauses.data())
        return res

    def implied_by(self, int var) -> CNF:
        """
        return a new CNF corresponding to (var -> self). I.e., the new CNF is
        build by appending ~var to each clause
        """
        return self.logical_or(-var)

    cdef vector[int] _get_units(CNF self):
        """
        Returns all unit clauses in the CNF. The unit clauses are returned as
        a numpy array without the separating zeros.
        """
        cdef vector[int] units
        cdef ssize_t idx, begin, end
        cdef size_t numclauses = self.start_indices.size()

        for idx in range(numclauses):
            begin = self.start_indices[idx]
            end = self.start_indices[idx + 1] if <size_t> idx + 1 < numclauses else self.clauses.size()

            if end == begin + 2:
                units.push_back(self.clauses[begin])

        units.shrink_to_fit()

        return units

    def get_units(self) -> np.ndarray:
        """
        Returns all unit clauses in the CNF. The unit clauses are returned as
        a numpy array without the separating zeros.

        >>> cnf = CNF([1,0, 1,2,3,0, -3,0, -1,-2,0, -2,0])
        >>> cnf.get_units()
        array([ 1, -3, -2], dtype=int32)
        """
        cdef vector[int] vec_result
        cdef int[::1] result_view
        cdef ssize_t idx

        vec_result = self._get_units()
        result = np.empty(vec_result.size(), dtype=np.int32)
        result_view = result

        for idx in range(vec_result.size()):
            result_view[idx] = vec_result[idx]

        return result


    def to_dimacs(self) -> str:
        '''
        return the CNF formatted in the DIMACS file format
        '''
        cdef size_t max_len = snprintf(NULL, 0, "%d", -self.nvars)
        cdef size_t nclauses = self.start_indices.size()
        cdef ssize_t buf_size = (self.clauses.size() - nclauses) * (max_len + 1) + nclauses * (2)
        cdef char *buf = <char *> malloc(buf_size + 1)
        cdef ssize_t buf_idx = 0, written
        cdef size_t i

        for i in range(self.clauses.size()):
            if self.clauses[i] == 0:
                written = snprintf(&buf[buf_idx], buf_size, "0\n")
                if written < 0:
                    assert 0
                buf_idx += written
                buf_size -= written
            else:
                written = snprintf(&buf[buf_idx], buf_size, "%d ", self.clauses[i])
                if written < 0:
                    assert 0
                buf_idx += written
                buf_size -= written

        assert buf_size >= 0
        py_str = buf[:buf_idx]
        free(buf)

        return 'p cnf ' + str(self.nvars) + ' ' + str(nclauses) + '\n' + py_str.decode()

    def to_espresso(self, print_numvars: bool = True) -> str:
        '''
        return the CNF formatted in the espresso file format
        '''
        cdef size_t i, j
        cdef int lit
        cdef char target_char
        cdef char *line_buf
        cdef fmt_buf buf = make_buf()

        if print_numvars:
            buf_printf(&buf, ".i %d\n", self.nvars)
            buf_printf(&buf, ".o 1\n", self.nvars)
        buf_printf(&buf, ".p %zd\n", self.start_indices.size())

        line_buf = <char *> malloc(self.nvars + 4)
        strcpy(line_buf + self.nvars, " 1\n");

        for i in range(self.start_indices.size()):
            memset(line_buf, b'-', self.nvars);

            j = 0
            while self.clauses[self.start_indices[i] + j] != 0:
                lit = self.clauses[self.start_indices[i] + j]

                target_char = b'0'
                if lit < 0:
                    target_char = b'1'
                    lit = -lit

                line_buf[self.nvars - lit] = target_char

                j += 1

            buf_printf(&buf, "%s", line_buf)

        # handle the case of a CNF with 0 clauses, a.k.a. a tautology
        if self.start_indices.size() == 0:
            memset(line_buf, b'-', self.nvars);
            line_buf[self.nvars + 1] = b'0'
            buf_printf(&buf, "%s", line_buf)

        buf_printf(&buf, ".e\n")
        py_bytes = buf.buf[:buf.len]

        free_buf(&buf)
        return py_bytes.decode()

    def minimize_espresso(self, espresso_args: List[str] = []) -> CNF:
        """
        Uses espresso to minimize the given CNF.

        :param espresso_args: extra parameters given when calling espresso, defaults to []

        :return: a new CNF object as minimized by espresso
        :rtype: CNF
        """
        cdef int ret_code

        with sp.Popen(['espresso'] + espresso_args, stdin=sp.PIPE, stdout=sp.PIPE, text=True) as espresso:
            espresso.stdin.write(self.to_espresso())
            espresso.stdin.close()

            cnf = CNF.from_espresso(espresso.stdout.read())

            ret_code = espresso.wait()
            if ret_code != 0:
                raise sp.CalledProcessError(ret_code, ' '.join(espresso.args))

            return cnf

    def _minimize_dimacs(self, args: List[str], outfile: str) -> CNF:
        """
        calls args with self.to_dimacs() as stdin, waits for the command to
        finish with exit code 10 or 20 and parses `outfile` as DIAMCS`.

        :return: a new CNF object read from `outfile`
        :rtype: CNF
        """
        cdef int ret_code = 0
        with sp.Popen(args, stdin=sp.PIPE, text=True) as minimizer:
            minimizer.stdin.write(self.to_dimacs())
            minimizer.stdin.close()

            ret_code = minimizer.wait()
            if ret_code not in [10, 20]:
                # for SAT solvers ret_code == 10 corresponds to SATISFIABLE and
                # ret == 20 to UNSATISFIABLE (see for example
                # http://www.satcompetition.org/2004/format-solvers2004.html)
                raise sp.CalledProcessError(ret_code, ' '.join(minimizer.args))

        with open(outfile, 'r') as f:
            return CNF.from_dimacs(f.read())

    def minimize_lingeling(self, optlevel=None, timeout=0, extra_args: List[str] = []) -> CNF:
        """
        Uses Lingeling to minimize the given CNF.

        :param optlevel: optimization level given to Lingeling via -O<optlevel>
        :param timeout: timeout given to Lingeling via -T <timeout>
        :param extra_args: extra parameters given when calling Lingeling, defaults to []

        :return: a new CNF object minimized by Lingeling
        :rtype: CNF
        """
        cdef int ret_code = 0

        args = ['lingeling', '-s']
        if optlevel is not None:
            args += [f'-O{optlevel}']
        if timeout is not None:
            args += ['-T', f'{timeout}']

        with NamedTemporaryFile(prefix='lingeling_', suffix='.cnf') as f:
            args += ['-o', f.name]
            args += extra_args
            return self._minimize_dimacs(args, f.name)



    def solve_dimacs(self, command: List[str], verbose=False) -> np.array:
        """
        solves the SAT by calling a DIMACS the compliant sat solver given by command.

        Returns (True, np.array(model, dtype=np.uint8)) for SAT instances.
        Returns (False, None) for UNSAT instances.
        """
        cdef int echo_comments, ret_code = 0
        cdef uint8_t[:] result_view

        result = np.full(self.nvars + 1, 255, dtype=np.uint8)
        result_view = result
        echo_comments = verbose

        cdef int32_t[::1] buf = None;
        cdef int32_t lit
        cdef ssize_t i
        cdef bint is_sat = False


        with NamedTemporaryFile('w', prefix='cnf_toolkit_', suffix='.cnf') as f:
            f.write(self.to_dimacs())
            f.flush()

            args = command + [f.name]
            with sp.Popen(args, stdin=sp.DEVNULL, stdout=sp.PIPE, text=True) as solver:

                for line in solver.stdout:
                    if echo_comments:
                        print(line, end='')

                    if line.startswith('c '):
                        continue

                    if line.startswith('s '):
                        line = line[2:].strip()
                        if line == 'SATISFIABLE':
                            is_sat = True
                        elif line == 'UNSATISFIABLE':
                            is_sat = False
                        else:
                            raise ValueError(f'unknown status: {line}')

                    if line.startswith('v '):
                        buf = np.fromstring(line[2:], dtype=np.int32, sep=' ')
                        for i in range(buf.shape[0]):
                            lit = buf[i]
                            if abs(lit) >= result_view.shape[0]:
                                raise IndexError('solver returned model index that is out of bounds')
                            result_view[abs(lit)] = lit > 0


                ret_code = solver.wait()
                if ret_code not in [10, 20]:
                    # for SAT solvers ret_code == 10 corresponds to SATISFIABLE and
                    # ret == 20 to UNSATISFIABLE (see for example
                    # http://www.satcompetition.org/2004/format-solvers2004.html)
                    raise sp.CalledProcessError(ret_code, ' '.join(solver.args))

                is_sat = ret_code == 10

            if is_sat:
                return True, result
            else:
                return False, None



    def check_solution(self, uint8_t[:] solution) -> bool:
        cdef size_t i
        for i in range(solution.shape[0]):
           if solution[i] not in [0, 1]:
               raise ValueError(f'all elements in solution must be in [0, 1]')

        return self._check_solution(solution)

    cdef int _check_solution(self, uint8_t[:] solution) except -1 nogil:
        cdef size_t i, clause_pos, clause_elem
        cdef uint8_t expected

        if self.nvars >= solution.shape[0]:
            raise IndexError(f'solution of length {solution.shape[0]} too short for CNF with {self.nvars} variables')

        for i in range(self.start_indices.size()):
            if not self._check_solution_for_single_clause(i, solution):
                return 0

        return 1

    cdef int _check_solution_for_single_clause(self, size_t clause_idx, uint8_t[:] solution) nogil:
        cdef size_t clause_pos
        cdef uint8_t expected
        cdef int clause_elem

        clause_pos = self.start_indices[clause_idx]

        while self.clauses[clause_pos] != 0:
            clause_elem = self.clauses[clause_pos]
            expected = clause_elem > 0
            if clause_elem < 0:
                clause_elem = -clause_elem

            if expected == solution[clause_elem]:
                return True

            clause_pos += 1

        return False

    def to_truthtable(self) -> Truthtable:
        """
        Return the current CNF as a Truthtable by iterating over all possible
        variable assignments.
        """
        cdef size_t i, j
        cdef uint8_t[:] truthtable_view
        cdef uint8_t[:] current_sol_view

        if self.nvars > 32:
            raise ValueError('cannot find all solutions for CNF with more than 32 variables')

        truthtable = np.empty((<uint64_t> 1) << self.nvars, np.uint8)
        current_sol = np.empty(1 + self.nvars, np.uint8)

        truthtable_view = truthtable
        current_sol_view = current_sol
        current_sol_view[0] = 0

        for i in range((<uint64_t> 1) << self.nvars):
            for j in range(1, self.nvars + 1):
                current_sol_view[j] = (i >> (j - 1)) & 1
            truthtable_view[i] = self._check_solution(current_sol_view)
            PyErr_CheckSignals()

        return Truthtable.from_lut(truthtable)


    cdef Clause get_clause(self, ssize_t idx):
        cdef size_t begin, end, i
        cdef size_t numclauses = self.start_indices.size()
        cdef Clause result

        if idx < 0:
            idx += numclauses
        if idx < 0 or <size_t> idx >= numclauses:
            raise IndexError('index out of range')

        begin = self.start_indices[idx]
        end = self.start_indices[idx + 1] if <size_t> idx + 1 < numclauses else self.clauses.size()
        result = Clause.from_memview((<int[:self.clauses.size()]> self.clauses.data())[begin:end - 1])

        return result

    def __getitem__(self, ssize_t idx) -> Clause:
        return self.get_clause(idx)

    def __len__(self) -> int:
        return self.start_indices.size()

    def __iter__(self) -> Iterable[Clause]:
        cdef size_t i = 0
        while i < self.start_indices.size():
            yield self.get_clause(i)
            i += 1

    def __reversed__(self) -> Iterable[Clause]:
        cdef size_t i
        for i in reversed(range(self.start_indices.size())):
            yield self.get_clause(i)

    cdef int _compare_clause(self, size_t idx, Clause other) except -1 nogil:
        cdef size_t begin, end, i
        cdef size_t numclauses = self.start_indices.size()

        if idx >= numclauses:
            with gil:
                raise IndexError('index out of range')

        begin = self.start_indices[idx]
        end = self.start_indices[idx + 1] if <size_t> idx + 1 < numclauses else self.clauses.size()

        if end - 1 - begin != other.clause.size():
            return 0

        for i in range(end - 1 - begin):
            if self.clauses[begin + i] != other.clause[i]:
                return 0

        return 1

    def __contains__(self, Clause needle) -> bool:
        cdef size_t i

        for i in range(self.start_indices.size()):
            if self._compare_clause(i, needle):
                return True

    def count(self, Clause needle) -> bool:
        cdef size_t i, count = 0


        for i in range(self.start_indices.size()):
            if self._compare_clause(i, needle):
                count += 1
        return count

    cdef size_t _get_slice_index(self, ssize_t idx):
        if idx < 0:
            idx += self.start_indices.size()
        if idx < 0:
            return 0
        if <size_t> idx >= self.start_indices.size():
            return self.start_indices.size()
        return <size_t> idx

    def index(self, Clause needle, start=None, end=None) -> int:
        """
        Return zero-based index in the list of the first item whose value is
        equal to needle. Raises a ValueError if there is no such item.

        The optional arguments start and end are interpreted as in the slice
        notation and are used to limit the search to a particular subsequence
        of the list. The returned index is computed relative to the beginning
        of the full sequence rather than the start argument.
        """

        cdef size_t i, start_idx = 0, end_idx = self.start_indices.size()
        if start is not None:
            start_idx = self._get_slice_index(start)
        if end is not None:
            end_idx = self._get_slice_index(end)

        for i in range(start_idx, end_idx):
            if self._compare_clause(i, needle):
                return i

        raise ValueError(f'{needle} is not in CNF')


    def __eq__(self, other) -> bool:
        if not isinstance(other, CNF):
            return False

        cdef CNF c_other = <CNF> other

        if self.nvars != c_other.nvars:
            return False
        if self.start_indices != c_other.start_indices:
            return False
        if self.clauses != c_other.clauses:
            return False
        return True

    def __repr__(self) -> str:
        return f'CNF over {self.nvars} variables with {self.start_indices.size()} clauses'

    def __str__(self) -> str:
        return self.to_dimacs().rstrip()

    # pickle support
    def __reduce__(self):
        return CNF, (np.array(self), self.nvars)

    def equiv(self, CNF other) -> bool:
        """
        Check for logical eqivalence between self and other.
        Calls `espresso -Dverify` to perform the comparison.
        """
        cdef int ret_code = 0

        if self.nvars != other.nvars:
            return False

        with sp.Popen(['espresso', '-Dverify'], stdin=sp.PIPE, stdout=sp.PIPE, text=True) as espresso:
            espresso.stdin.write(self.to_espresso())
            espresso.stdin.write(other.to_espresso(print_numvars = False))
            espresso.stdin.close()

            ret_code = espresso.wait()
            if ret_code not in [0, 1]:
                raise sp.CalledProcessError(ret_code, ' '.join(espresso.args))

            return ret_code == 0

    def translate(self, mapping) -> CNF:
        """
        Translate all variables in the CNF to a new index.
        The translation mapping is given by the mapping paramter.

        Index 0 must always map to index 0 again.

        :return: a new CNF with variables changed according to mapping parameter
        """
        cdef np_vars = np.array(mapping, copy=False, dtype=np.int32)
        cdef int [::1] var_view = np_vars
        cdef size_t i

        if var_view.shape[0] != self.nvars + 1:
            raise ValueError('need to provide translation for all 1+{self.nvars} variables')
        if var_view[0] != 0:
            raise ValueError('variable 0 must map to 0 again')

        for i in range(1, <size_t> var_view.shape[0]):
            if var_view[i] == 0:
                raise ValueError('variable must not be mapped to 0')

        cdef vector[int] new_clauses
        new_clauses.resize(self.clauses.size())

        cdef int var
        for i in range(self.clauses.size()):
            var = self.clauses[i]
            new_clauses[i] = var_view[abs(var)] * (1 if var > 0 else -1)

        cdef CNF res = CNF.__new__(CNF)
        res._add_clauses(<int[:new_clauses.size()]> new_clauses.data())
        return res

    # buffer support
    def __getbuffer__(self, cython.Py_buffer *buffer, int flags):
        if flags & PyBUF_WRITABLE:
            raise ValueError('cannot provide a writable buffer for CNF')

        self.shape[0] = self.clauses.size()
        self.view_count += 1

        buffer.buf = <char *>&(self.clauses[0])

        if (flags & PyBUF_FORMAT) == PyBUF_FORMAT:
            buffer.format = 'i'                     # int
        else:
            buffer.format = NULL
        buffer.internal = NULL
        buffer.itemsize = sizeof(int)
        buffer.len = self.shape[0] * sizeof(int)
        buffer.obj = self
        buffer.readonly = 1
        if (flags & PyBUF_ND) == PyBUF_ND:
            buffer.ndim = 1
            buffer.shape = &self.shape[0]
        else:
            buffer.ndim = 0
            buffer.shape = NULL

        if (flags & PyBUF_STRIDES) == PyBUF_STRIDES:
            buffer.strides = &buffer.itemsize
        else:
            buffer.strides = NULL

        buffer.suboffsets = NULL                # for pointer arrays only

    def __releasebuffer__(self, Py_buffer *buffer):
        buffer.buf = NULL;
        self.view_count -= 1

collections.abc.Sequence.register(CNF)


cdef class Truthtable:
    """
    A boolean function represented as a truth table.
    The truth table is stored using a table for the ON set and the DC set of the function.
    The ON set denotes where the value of the function is 1/true, while the DC set
    denotes where the value of the function is left unspecified.
    The DC set allows more efficient CNF representations when optimizing with espresso.
    The OFF set is the complement of the two sets.
    """
    cdef public on_set
    cdef public dc_set
    cdef public uint64_t numbits

    INIT_INDEX_SET = object()

    def __init__(self, init_key, numbits, on_set, dc_set):
        if init_key is not self.INIT_INDEX_SET:
            raise ValueError('use .from_lut or .from_indices')

        if len(on_set.shape) != 1 or (dc_set is not None and len(dc_set.shape) != 1):
            raise ValueError('parameters must be one dimensional')

        self.numbits = numbits
        self.on_set = np.sort(on_set)
        self.dc_set = np.sort(dc_set)


    @classmethod
    def from_lut(cls, on_lut: np.ndarray, dc_lut: Optional[np.ndarray] = None) -> Truthtable:
        """
        Creates a Truthtable from a lookup table (LUT).

        :param on_lut: specifies where the Truthtable should be on, i.e., 1
        :param dc_lut: specifies where the value of the Truthtable can be ignored, this allows for better optimization, defaults to None

        :return: Truthtable with on_set and dc_set initialized according to parameters
        :rtype: Truthtable
        """
        if len(on_lut.shape) != 1 or (dc_lut is not None and len(dc_lut.shape) != 1):
            raise ValueError('parameters must be one dimensional')

        if dc_lut is not None and on_lut.shape != dc_lut.shape:
            raise ValueError('parameters must have the same shape')

        numvals = on_lut.shape[0]
        if (numvals & (numvals - 1)) != 0 or numvals == 0:
            raise ValueError('length parameter must be a power of 2')

        numbits = numvals.bit_length() - 1

        if dc_lut is not None and np.any(on_lut & dc_lut):
            raise ValueError('on_lut and dc_lut must be distinct')

        on_set, = np.where(on_lut)
        dc_set, = np.where(dc_lut) if dc_lut is not None else (np.array([], int),)
        return cls(cls.INIT_INDEX_SET, numbits, on_set, dc_set)

    @classmethod
    def from_indices(cls, numbits: int, on_indices: np.array, dc_indices = np.array([], dtype=int)) -> Truthtable:
        """
        Creates a Truthtable from the set of indices with value 1 and optionally indices where the value can be ignored.

        :param numbits: number of input bits for the Truthtable
        :param on_indices: set of indices where the Truthtable should be on, i.e., 1
        :param dc_indices: set of indices where the Truthtable value can be ignored, this allows for better optimization, defaults to []

        :return: Truthtable with on_set and dc_set initialized according to parameters
        :rtype: Truthtable
        """
        return cls(cls.INIT_INDEX_SET, numbits, on_indices, dc_indices)


    def _write(self, io: io.TextIOBase, espresso: bool = False, invert: bool = False) -> None:
        i_list = range(1 << self.numbits)

        if espresso:
            i_list = np.concatenate((self.on_set, self.dc_set), axis=0)
            i_list.sort()
            io.write(f'.i {self.numbits}\n')
            io.write(f'.o 1\n')
            io.write(f'.p {len(i_list)}\n')

            # this parameter is important so espresso minimizes the inverted function
            io.write(f'.phase {not invert:b}\n')

        def sorted_contains(a, v):
            idx = np.searchsorted(a, v)
            return idx < len(a) and a[idx] == v

        for i in i_list:
            on, dc = sorted_contains(self.on_set, i), sorted_contains(self.dc_set, i)
            desc = '-' if dc else f'{on:b}'
            io.write(f'{i:0{self.numbits}b} {desc}\n')

        # ensure that we write some PLA description
        if len(i_list) == 0:
            io.write('-' * self.numbits + ' 0\n')

        if espresso:
            io.write(f'.end\n')

    def to_espresso(self, phase='cnf') -> str:
        """
        return the truthtable in espresso format

        :param phase: wether to minimize the off_set ('cnf'/0) or on_set ('dnf'/1)

        :return: Truthtable with on_set and dc_set initialized according to parameters
        :rtype: Truthtable
        """
        if phase == 'cnf' or phase == '0' or phase == 0:
            invert = True
        elif phase == 'dnf' or phase == '1' or phase == 1:
            invert = False
        else:
            raise ValueError('invalid value for `phase`')

        res = io.StringIO()
        self._write(res, True, invert)
        return res.getvalue()

    def __repr__(self) -> str:
        result = io.StringIO()
        result.write("Truthtable\n")
        self._write(result)
        return result.getvalue()

    def __eq__(self, other) -> bool:
        if self.numbits != other.numbits:
            return False
        if not np.all(self.on_set == other.on_set):
            return False
        if self.dc_set is None and other.dc_set is not None:
            return False
        if other.dc_set is None and self.dc_set is not None:
            return False
        if not np.all(self.dc_set == other.dc_set):
            return False
        return True

    def to_cnf(self, espresso_args: List[str] = []) -> CNF:
        """
        Uses espresso to convert the Truthtable to a minimized CNF.

        The resulting CNF will be indexed by [1, self.numbits], where index 1
        corresponds to the least significant bit in the truthtable index.

        :param espresso_args: extra parameters given when calling espresso, defaults to []

        :return: CNF minimized by espresso
        :rtype: CNF
        """
        with sp.Popen(['espresso'] + espresso_args, stdin=sp.PIPE, stdout=sp.PIPE, text=True) as espresso:
            self._write(espresso.stdin, True, True)
            espresso.stdin.close()

            cnf = CNF.from_espresso(espresso.stdout.read())

            ret_code = espresso.wait()
            if ret_code != 0:
                raise sp.CalledProcessError(ret_code, ' '.join(espresso.args))

            return cnf
