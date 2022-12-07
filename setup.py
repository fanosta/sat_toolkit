#!/usr/bin/env python3
from setuptools import Extension, setup, find_packages
from Cython.Build import cythonize
from Cython.Build.Dependencies import default_create_extension


def my_create_extension(template, kwds):
    libs = kwds.get('libraries', []) + ["mylib"]
    kwds['libraries'] = libs
    kwds['include_dirs'] = ['sat_toolkit'] + kwds.get("include_dirs", [])
    kwds['library_dirs'] = ['sat_toolkit'] + kwds.get("library_dirs", [])
    print('#'*80)
    print(kwds)
    print('#'*80)
    return default_create_extension(template, kwds)


with open("README.md", "r") as fh:
    long_description = fh.read()

extensions = [
    Extension("sat_toolkit/*", ["sat_toolkit/*.pyx"]),
]

setup(
    name="sat-toolkit",
    version="0.0.1",
    author="fanosta",
    author_email="fanosta@users.noreply.github.com",
    description="Tool for manipulating CNF/DNF formulas",
    long_description=long_description,
    long_description_content_type="text/markdown",
    # url="https://github.com/fanosta/sat-toolkit",
    packages=find_packages(),
    install_requires=['numpy>=1.21'],
    ext_modules=cythonize(extensions),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
