#!/usr/bin/env python3
from setuptools import Extension, setup
from Cython.Build import cythonize


setup(
    ext_modules=cythonize([
        Extension("sat_toolkit/*", ["sat_toolkit/*.pyx"]),
    ]),
)
