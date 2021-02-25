from pyomo.environ import Objective, minimize, Constraint
from pyomo.opt import SolverFactory
from pyomo.dataportal.DataPortal import DataPortal




def Model_Resolution(model):
    '''
    This function creates the model and calls Pyomo to solve the instance of the project

    :param model: Pyomo model as defined in the components_creation library

    :return: The solution inside an object call instance.
    '''

    instance = model.create_instance(r'gisele\michele\Inputs\data.dat')  # load parameters
    # opt = SolverFactory('cplex',executable=r'C:\Program Files\IBM\ILOG\CPLEX_Studio1210\cplex\bin\x64_win64\cplex')
    opt = SolverFactory('gurobi')  # Solver use during the optimization
    opt.options['mipgap'] = 0.05
    print('Begin Optimization')
    results = opt.solve(instance, tee=True)  # Solving a model instance
    instance.solutions.load_from(results)  # Loading solution into instance
    return instance

