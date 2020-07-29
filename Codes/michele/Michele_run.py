from pyomo.environ import AbstractModel, Var, value

from Codes.michele.Michele_model_creation import Model_Creation
from Codes.michele.Michele_model_resolution import Model_Resolution
from Codes.michele.Michele_results import Load_results
from Codes.michele.Michele_initialize import importing


def start(load_profile, pv_avg, wt_avg):
    input_load, wt_prod, pv_prod = importing(load_profile, pv_avg, wt_avg)

    model = AbstractModel()  # define type of optimization problem

    # Optimization model
    print('Starting model creation')
    Model_Creation(model, input_load, wt_prod, pv_prod)  # Creation of the Sets, parameters and variables.
    print('Starting model resolution')
    instance = Model_Resolution(model)  # Resolution of the instance

    print('Show results')

    inst_pv, inst_wind, inst_dg, inst_bess, inst_inv, init_cost, rep_cost, \
        om_cost, salvage_value, gen_energy, load_energy \
        = Load_results(instance)

    return inst_pv, inst_wind, inst_dg, inst_bess, inst_inv, init_cost, \
        rep_cost, om_cost, salvage_value, gen_energy, load_energy



