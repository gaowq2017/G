from setuptools import setup
from Cython.Build import cythonize

setup(name='my_module', ext_modules=cythonize(r'D:\iPhone_Plug\colorCompare.py'))
