from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import *

class IndexSet:
    numvars: int

    def __init__(self):
        self.numvars = 0
        self._fieldnames = set()

    def add_index_array(self, name: str, shape: Tuple[int, ...]):
        length = np.prod(shape, dtype=np.int32)
        res = np.arange(self.numvars + 1, self.numvars + 1 + length, dtype=np.int32)
        res = res.reshape(shape)
        res.flags.writeable = 0
        self.numvars += length

        self._fieldnames.add(name)

        setattr(self, name, res)

    def __repr__(self):
        res = f'{self.__class__.__name__}(\n'
        for fieldname in self._fieldnames:
            field = getattr(self, fieldname)
            min_val = field.ravel()[0]
            max_val = field.ravel()[-1]
            res += f'  {fieldname} = np.arange({min_val}, {max_val + 1}).reshape({field.shape!r})\n'
        res += ')'

        return res
