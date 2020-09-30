from pyomo.environ import Objective, minimize, Constraint
from pyomo.opt import SolverFactory
from pyomo.dataportal.DataPortal import DataPortal


from Codes.michele.Michele_constraints import *
from Codes.michele.Michele_initialize import *

def Model_Resolution(model):
    '''
    This function creates the model and call Pyomo to solve the instance of the proyect 

    :param model: Pyomo model as defined in the Model_creation library
    :param datapath: path to the input data file

    :return: The solution inside an object call instance.
    '''


    # OBJETIVE FUNTION:
    model.ObjectiveFuntion = Objective(rule=total_net_present_cost, sense=minimize)

    # CONSTRAINTS
    # to compute OF
    model.TotalInitialInvestment = Constraint(rule=total_initial_investment)
    model.TotalReplacementCost = Constraint(rule=total_replacement_cost)
    model.TotalOMCost = Constraint(rule=total_OM_cost)
    model.TotalSalvageValue = Constraint(rule=total_salvage_value)
    # to design the system
    model.TotalLoad = Constraint(model.hours, rule=total_load)
    model.PvInstalled = Constraint(model.pv, rule=pv_installed)
    model.WtInstalled = Constraint(model.wt, rule=wt_installed)
    model.BessInstalled = Constraint(model.bess, rule=bess_installed)
    model.DgInstalled = Constraint(model.dg, rule=dg_installed)
    model.ResEnergy = Constraint(model.hours, rule=res_energy)
    model.SystemBalance = Constraint(model.hours, rule=system_balance)
    model.TotalEnergyReq = Constraint(model.hours, rule=total_energy_req)
    model.TotalLostLoad = Constraint(model.hours, rule=total_lost_load)
    model.LimitLostLoad = Constraint(model.hours, rule=limit_lost_load)
    model.TotalReserveReq = Constraint(model.hours, rule=total_reserve_req)
    model.ReserveAllocation = Constraint(model.hours, rule=reserve_allocation)
    # constraints related to diesel generators
    model.FuelConsumptionCurve = Constraint(model.hours,model.dg, rule=fuel_consumption_curve)
    model.DgPowerMax = Constraint(model.hours, model.dg, rule=dg_power_max)
    model.DgPowerMin = Constraint(model.hours, model.dg, rule=dg_power_min)
    model.DgOnline = Constraint(model.hours, model.dg, rule=dg_online)
    # constraints related to batteries
    model.BatteryPowerMax = Constraint(model.hours,model.bess, rule=battery_power_max)
    model.BessCondition1 = Constraint(model.hours, model.bess, rule=bess_condition1)
    model.BessCondition2 = Constraint(model.hours, model.bess, rule=bess_condition2)
    model.BessChargingLevel = Constraint(model.hours,model.bess, rule=bess_charging_level)
    model.BessChargingLevelMin = Constraint(model.hours,model.bess, rule=bess_charging_level_min)
    model.BessChargingLevelMax = Constraint(model.hours,model.bess, rule=bess_charging_level_max)
    model.BessChPowerMax = Constraint(model.hours,model.bess, rule=bess_ch_power_max)
    model.BessDisPowerMax = Constraint(model.hours, model.bess, rule=bess_dis_power_max)

    instance = model.create_instance(r'Codes\michele\Inputs\data.dat')  # load parameters
    #opt = SolverFactory('cplex',executable=r'C:\Program Files\IBM\ILOG\CPLEX_Studio1210\cplex\bin\x64_win64\cplex')
    opt = SolverFactory('gurobi')  # Solver use during the optimization
    opt.options['mipgap'] = 0.05
    print('Begin Optimization')
    results = opt.solve(instance, tee=True)  # Solving a model instance
    instance.solutions.load_from(results)  # Loading solution into instance
    return instance

