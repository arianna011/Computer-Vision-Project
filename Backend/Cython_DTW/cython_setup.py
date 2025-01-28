from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=cythonize(
        "optimized_dtw.pyx",
        compiler_directives={"language_level": "3"}  # Ensure Python 3 syntax
    ),
    include_dirs=[numpy.get_include()],  # Include NumPy headers
)
