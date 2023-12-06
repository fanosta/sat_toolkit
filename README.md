# SAT-toolkit
A tool for efficiently handling CNF and DNF formulas in Python

## Installation

To install from pypi use:
```bash
pip install sat-toolkit
```

To simplify CNFs and convert DNFs to CNFs, you need to compile
[espresso](https://github.com/classabbyamp/espresso-logic) and make it
available on inside your $PATH.


## Usage Example

The following example shows how to create a minified CNF for a boolean function that is true at values 0, 6, 9, and 15.

```python
from sat_toolkit.formula import CNF, Truthtable
import numpy as np

table = np.zeros(16, int)
table[[0, 6, 9, 15]] = 1

tt = Truthtable.from_lut(table)
cnf = tt.to_cnf()
print(cnf)
```
