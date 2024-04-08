#!/usr/bin/env python3
from setuptools import Extension, setup, find_packages

try:
    from Cython.Build import cythonize

    USE_CYTHON = True
except ImportError:
    USE_CYTHON = False

with open("README.md", "r") as fh:
    long_description = fh.read()

extension = ".pyx" if USE_CYTHON else ".cpp"
extensions = [
    Extension("sat_toolkit/*", ["sat_toolkit/*" + extension]),
]

if USE_CYTHON:
    extensions = cythonize(extensions)

setup(
    name="sat-toolkit",
    version="0.4.2",
    author="Marcel Nageler",
    author_email="marcel.nageler@iaik.tugraz.at",
    description="Tool for manipulating CNF formulas",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/fanosta/sat_toolkit",
    packages=find_packages(),
    install_requires=["numpy>=1.21"],
    ext_modules=extensions,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
    ],
)
