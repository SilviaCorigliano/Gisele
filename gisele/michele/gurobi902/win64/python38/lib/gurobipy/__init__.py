import os
if hasattr(os, 'add_dll_directory'):
  os.add_dll_directory(os.path.join(os.getenv('GUROBI_HOME'),'bin'))
from .gurobipy import *
