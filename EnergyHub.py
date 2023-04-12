import numpy as np
import pandas as pd
from cvxpy import *
import matplotlib.pyplot as plt
import openpyxl
import os
import sys



def main():
    # Set the path to the Gurobi library
    GRB_HOME = 'C:\gurobi1001\win64\python311'
    os.environ['GUROBI_HOME'] = GRB_HOME
    sys.path.append(os.path.join(GRB_HOME, 'lib'))  # Append the path to the Gurobi library to PYTHONPATH

    hospital = EnergyHub()
    hospital.run_optimisation('co2', print_results=True)
    #hospital.plot_data(xlim=240, xmin=216)
    hospital.pareto_front(granularity=5)




class EnergyHub:
    def __init__(self):
        # Energy demands
        # Data for each type of energy demand should be in the form of a 1x8760 vector
        demand_data = pd.read_excel('electricity_heat_cooling_demand.xlsx', header=None)
        elec_demand_standard = demand_data.iloc[:, 0].to_numpy().flatten() #/ 5.5  # Electricity demand [kWh]
        self.heat_demand = demand_data.iloc[:, 1].to_numpy().flatten() #/ 5.5  # Heating demand [kWh]
        cool_demand = demand_data.iloc[:, 2].to_numpy().flatten() #/ 5.5  # Cooling demand [kWh]
    
    
        # Energy efficient consumer appliances
        self.energy_efficiency_adoption = Variable(1)  # Adaption to energy efficient consumer appliances (in % (0 = no adaption, 1 = full adaption))
        max_reduction = 0.3 # maximum reduction possible (in %)
        Cap_energy_efficiency = 2500000  # self.cost for increasing energy efficiency (in USD for full adaption)
        self.elec_demand = elec_demand_standard * (1 - self.energy_efficiency_adoption * max_reduction)
        efficiency_con = [self.energy_efficiency_adoption >= 0, self.energy_efficiency_adoption <= 1]
    
        # Renewable energy potentials
        # Similarly to energy demands, data should be in the form of a 1x8760 vector
        solar = pd.read_excel('temperature_irraidance.xlsx')
        solar = [0] + solar.iloc[:, 1].to_numpy().flatten() / 1000  # Incoming solar radiation patterns [kWh/m2]
        solar = np.insert(solar, 0, 0)

        # Optimization horizon
        horizon = 8760  # hours in a calendar year
    
        # Discounted cash flow calculations
        d = 0.1  # Interest rate used to discount future operational cashflows

    
        ## Connections with energy grids at the input of the energy system
        # ========================
        # Variable definitions
        self.Imp_elec = Variable(horizon)  # Electricity import from the grid for every time step [kWh]
        self.Imp_oil = Variable(horizon)  # Diesel consumption for every time step [kWh]
        Exp_elec = Variable(horizon)  # Electricity export from the grid for every time step [kWh]

    
        # Parameter definitions
        price_oil = 1.078  # Diesel price [USD/kWh]
        esc_oil = 0.02  # Escalation rate per year for diesel price
        price_elec = 0.27  # Grid electricity price [USD/kWh]
        max_grid_import = 700  # Maximum electricity import from grid (in kW)
        esc_elec = 0.02  # Escalation rate per year for electricity price
        exp_price_elec = 0.27  # Feed-in tariff for exported electricity [USD/kWh]
        esc_elec_exp = 0.02  # Escalation rate per year for feed-in tariff for exported electricity [%]
        co2_oil = 0.84  # Diesel emission factor [kgCO2/kWh]
        co2_elec = 0.84  # Electricity emission factor [kgCO2/kWh]
    
        # Define transition probabilities of a blackout =>
        # 90% probability that after a blackout hour the next hour stays without current
        # 80% probability that after an hour with current the next hour stays with current
        transition_probability = np.array([[0.9, 0.1], [0.2, 0.8]])
    
        # Generate Markov chain
        grid_access = np.zeros(horizon)
        grid_access[0] = 1  # Initial state is assumed to be no blackout
        for n in range(1, horizon):
            grid_access[n] = np.random.rand() < transition_probability[int(grid_access[n - 1]), 1]

        # Constraint definitions
        # ----------------------
        grid_con = [max_grid_import >= self.Imp_elec, self.Imp_elec >= 0, self.Imp_elec == multiply(self.Imp_elec, grid_access), self.Imp_oil >= 0, Exp_elec >= 0]


        ## Oil boiler (ob)
        # ========================
        # Parameter definitions
        # ---------------------
        eff_ob = 0.9  # Conversion efficiency of oil boiler
        cost_ob = 99  # Investment self.cost for oil boiler [CHF, EUR, USD/kW]
    
        # Capacity variable
        # ------------------
        self.Cap_ob = Variable(1)  # Capacity of oil boiler [kW]
    
        # Input and output variables
        # --------------------------
        self.P_in_ob = Variable(horizon)  # Input energy to oil boiler [kWh]
        self.P_out_ob = Variable(horizon)  # Heat generation by oil boiler [kWh]
    
        # oil boiler self.constraints
        # ----------------------
        ob_con = [self.Cap_ob >= 0, self.P_out_ob == self.P_in_ob * eff_ob, self.P_in_ob >= 0, self.P_out_ob >= 0, self.P_out_ob <= self.Cap_ob]
    
    
        ## Ground-source heat pump (gshp)
        # ===============================
        # Parameter definitions
        # ---------------------
        eff_gshp = 4  # Conversion efficiency (Coefficient of Performance) of ground-source heat pump
        cost_gshp = 938.64  # Investment self.cost for ground-source heat pump [USD/kW]
    
        # Capacity variables
        # ------------------
        self.Cap_gshp = Variable(1)  # Capacity of ground-source heat pump [kW]
    
        # Input and output variables
        # --------------------------
        self.P_in_gshp = Variable(horizon)  # Input energy to ground-source heat pump [kWh]
        self.P_out_gshp = Variable(horizon)  # Heat generation by ground-source heat pump [kWh]
    
        # GSHP self.constraints
        # ----------------
        gshp_con = [self.Cap_gshp >= 0, self.P_out_gshp == self.P_in_gshp * eff_gshp, self.P_in_gshp >= 0, self.P_out_gshp >= 0, self.P_out_gshp <= self.Cap_gshp]
    
    
        # Ground-source heat pump for cooling (gshp_cool)
        # ==============================================
        # Parameter definitions
        # ---------------------
        eff_gshp_cool = 4  # Conversion efficiency (Coefficient of Performance) of ground-source heat pump
        cost_gshp_cool = 938.64  # Investment self.cost for ground-source heat pump [USD/kW]
    
        # Capacity variables
        # ------------------
        self.Cap_gshp_cool = Variable(1)  # Capacity of ground-source heat pump [kW]
    
        # Input and output variables
        # --------------------------
        self.P_in_gshp_cool = Variable(horizon)  # Input energy to ground-source heat pump [kWh]
        P_out_gshp_cool = Variable(horizon)  # Heat generation by ground-source heat pump [kWh]
    
        # GSHP_cool self.constraints
        # ----------------
        gshp_cool_con = [self.Cap_gshp_cool >= 0,  P_out_gshp_cool == self.P_in_gshp_cool * eff_gshp_cool,  self.P_in_gshp_cool >= 0,  P_out_gshp_cool >= 0,  P_out_gshp_cool <= self.Cap_gshp_cool]
    
    
        # Combined heat and power engine (chp)
        # =====================================
        # Parameter definitions
        # ---------------------
        eff_elec_chp = 0.3  # Electrical efficiency of combined heat and power engine
        eff_heat_chp = 0.6  # Thermal efficiency of combined heat and power engine
        cost_chp = 773  # Investment self.cost for combined heat and power engine [USD/kWe]
    
        # Capacity variable
        # -----------------
        self.Cap_chp = Variable(1)  # Electrical capacity of combined heat and power engine [kWe]
    
        # Input and output variables
        # --------------------------
        self.P_in_chp = Variable(horizon)  # Input energy to combined heat and power engine (diesel) [kWh]
        self.P_out_heat_chp = Variable(horizon)  # Heat generation by combined heat and power engine [kWh]
        self.P_out_elec_chp = Variable(horizon)  # Electricity generation by combined heat and power engine [kWh]
    
        # CHP self.constraints
        # ---------------
        chp_con = [self.Cap_chp >= 0, self.P_out_heat_chp == self.P_in_chp * eff_heat_chp, self.P_out_elec_chp == self.P_in_chp * eff_elec_chp, self.P_in_chp >= 0, self.P_out_heat_chp >= 0, self.P_out_elec_chp >= 0, self.P_out_elec_chp <= self.Cap_chp]
    
    
        # Photovoltaic panels
        # ====================
        # Parameter definitions
        # ---------------------
        eff_pv = 0.15  # Conversion efficiency (Coefficient of Performance) of photovoltaic panels
        cost_pv = 276  # Investment self.cost for photovoltaic panels [USD/m2]
        max_solar_area = 10000  # Maximum available area to accommodate photovoltaic panels [m2]
    
        # Capacity variable
        # -----------------
        self.Cap_pv = Variable(1)  # Capacity of photovoltaic panels [m2]
    
        # Input and output variables
        # --------------------------
        self.P_out_pv = Variable(horizon)  # Electricity generation by photovoltaic panels [kWh]
    
        # PV self.constraints
        # --------------
        pv_con = [self.Cap_pv >= 0, self.Cap_pv <= max_solar_area, self.P_out_pv >= 0, self.P_out_pv == solar * self.Cap_pv * eff_pv]
    
    
        # Thermal storage tank
        # =====================
        # Parameter definitions
        # ---------------------
        self_dis_ts = 0.01  # Self-discharging losses of thermal storage tank (in % per hour)
        ch_eff_ts = 0.9  # Charging efficiency of thermal storage tank
        dis_eff_ts = 0.9  # Discharging efficiency of thermal storage tank
        max_ch_ts = 0.25  # Maximum charging rate of thermal storage tank (in %)
        max_dis_ts = 0.25  # Maximum discharging rate of thermal storage tank (in %)
        cost_ts = 10  # Investment self.cost for thermal storage tank [USD/kWh]
    
        # Capacity variables
        # ------------------
        self.Cap_ts = Variable(1)  # Capacity of thermal storage tank [kWh]
    
        # Storage variables
        # -----------------
        self.Q_in_ts = Variable(horizon)  # Input energy flow to thermal storage tank [kWh]
        self.Q_out_ts = Variable(horizon)  # Output energy flow from thermal storage tank [kWh]
        E_ts = Variable(horizon + 1)  # Stored energy in thermal storage tank [kWh]
    
        # Storage tank self.constraints
        # ------------------------
        ts_con_1 = [self.Cap_ts >= 0, self.Q_in_ts >= 0, self.Q_out_ts >= 0, E_ts >= 0, E_ts <= self.Cap_ts, self.Q_in_ts <= max_ch_ts * self.Cap_ts, self.Q_out_ts <= max_dis_ts * self.Cap_ts]
    
        # Storage self.constraints
        # -------------------
        ts_con_2 = [E_ts[1:] == (1 - self_dis_ts) * E_ts[:-1] + ch_eff_ts * self.Q_in_ts - (1 / dis_eff_ts) * self.Q_out_ts, E_ts[0] == 0]
    
        # Combine self.constraints
        # -------------------
        ts_con = ts_con_1 + ts_con_2
    
    
        # Battery
        # ========
        # Parameter definitions
        # ---------------------
        self_dis_bat = 0.001  # Self-discharging losses of battery
        ch_eff_bat = 0.95  # Charging efficiency of battery
        dis_eff_bat = 0.95  # Discharging efficiency of battery
        max_ch_bat = 0.3  # Maximum charging rate of battery
        max_dis_bat = 0.3  # Maximum discharging rate of battery
        cost_bat = 386  # Investment self.cost for battery [USD/kWh]
    
        # Capacity variables
        # ------------------
        self.Cap_bat = Variable(1)  # Capacity of battery [kWh]
        max_cap_bat = 1000  # Maximal battery size [kWh]
    
        # Storage variables
        # -----------------
        self.Q_in_bat = Variable(horizon)  # Input energy flow to battery [kWh]
        self.Q_out_bat = Variable(horizon)  # Output energy flow from battery [kWh]
        E_bat = Variable(horizon + 1)  # Stored energy in battery [kWh]
    
        # Battery self.constraints
        # -------------------
        bat_con_1 = [self.Cap_bat >= 0, self.Cap_bat <= max_cap_bat, self.Q_in_bat >= 0, self.Q_out_bat >= 0, E_bat >= 0, E_bat <= self.Cap_bat,
                     self.Q_in_bat <= max_ch_bat * self.Cap_bat, self.Q_out_bat <= max_dis_bat * self.Cap_bat]
    
        # Battery self.constraints
        # -------------------
        bat_con_2 = [E_bat[1:] == (1 - self_dis_bat) * E_bat[0:-1] + ch_eff_bat * self.Q_in_bat - (1 / dis_eff_bat) * self.Q_out_bat, E_bat[0] == 0]
    
        # Combine self.constraints
        # -------------------
        bat_con = bat_con_1 + bat_con_2


        # How much self reliance
        self_reliance = True
        if self_reliance:
            self_heating = 0.8
            self_cooling = 0.5
            self_elec = 0.9
        else: 
            self_heating = 0
            self_cooling = 0
            self_elec = 0

        cap_con = [self.Cap_pv + self.Cap_chp >= self_heating*self.P_in_gshp + self_cooling*self.P_in_gshp_cool + self_elec*self.elec_demand[0:horizon]]

        ## Balance equations
        # ==================
        heat_con = [self.P_out_heat_chp + self.P_out_gshp + self.P_out_ob + self.Q_out_ts - self.Q_in_ts == self.heat_demand[0:horizon]]  # Heat balance
        cool_con = [P_out_gshp_cool == cool_demand[0:horizon]]  # "Cool" balance
        power_con = [self.Imp_elec + self.P_out_pv + self.P_out_elec_chp - self.P_in_gshp - self.P_in_gshp_cool + self.Q_out_bat - self.Q_in_bat == self.elec_demand[0:horizon] + Exp_elec]  # Electricity balance
        oil_con = [self.Imp_oil - self.P_in_ob - self.P_in_chp == 0]  # diesel balance
    
        ## Objective function
        # ===================
        # Total costs: Investment costs + 25 years of energy costs
        # --------------------------------------------------------
        Inv = Cap_energy_efficiency * self.energy_efficiency_adoption + self.Cap_ob * cost_ob + self.Cap_gshp * cost_gshp + self.Cap_gshp_cool * cost_gshp_cool + self.Cap_chp * cost_chp + self.Cap_pv * cost_pv + self.Cap_ts * cost_ts + self.Cap_bat * cost_bat
    
        Op = Variable(25)
        op_con = []
        for y in range(1, 26):
            op_con += [Op[y - 1] == sum(self.Imp_oil * price_oil * ((1 + esc_oil) ** (y - 1))) + sum(self.Imp_elec * price_elec * ((1 + esc_elec) ** (y - 1))) - sum(Exp_elec * exp_price_elec * ((1 + esc_elec_exp) ** (y - 1)))]
    
        self.cost = Inv + sum(Op / (1 + d) ** np.arange(1, 26))
        self.co2 = 25 * sum(self.Imp_oil * co2_oil + self.Imp_elec * co2_elec)  # self.co2 emissions from solar as well as batteries, thermal storage and heat pumps are neglected ...


        ## Collect all self.constraints
        # ========================
        self.constraints = cap_con + efficiency_con + grid_con + ob_con + gshp_con + chp_con + pv_con + ts_con + bat_con + heat_con + gshp_cool_con + cool_con + power_con + oil_con + op_con
    
        return

    def run_optimisation(self, objective: str = 'cost', opt_constraints: list = [], print_results=False):
        """
        Runs the oprimisation of the EnergHub

        :param objective: objective can only be "cost" or "co2"
        :param opt_constraints: constraints in a list if, optional, if left empty standard constraints from the EnergyHub is appleid
        :param print_results: prints the values in the console
        :return: tuple (cost, co2)
        """
        if not opt_constraints:
            opt_constraints = self.constraints

        if objective == 'cost':
            prob = Problem(Minimize(self.cost), opt_constraints)
        elif objective == 'co2':
            prob = Problem(Minimize(self.co2), opt_constraints)
        else:
            print('Objective can only be "cost" or "co2"')
            return

        prob.solve(solver=GUROBI, verbose=True)
    
        if print_results:
            # Output optimal energy system design
            # -----------------------------------
            print(f"The value of the total system self.cost is equal to: {self.cost.value} USD")
            print(f"The value of the total system emissions is equal to: {self.co2.value} kg CO_2")
            print(f"The capacity of the oil boiler is: {int(self.Cap_ob.value)} kW")
            print(f"The adoption of energy efficient consumer appliances is: {self.energy_efficiency_adoption.value * 100} %")
            print(f"The capacity of the combined heat and power engine is: {int(self.Cap_chp.value)} kW")
            print(f"The capacity of the ground-source heat pump is: {int(self.Cap_gshp.value)} kW")
            print(f"The capacity of the ground-source heat pump for cooling is: {int(self.Cap_gshp_cool.value)} kW")
            print(f"The capacity of the photovoltaic panels is: {int(self.Cap_pv.value)} m2")
            print(f"The capacity of the thermal storage is: {int(self.Cap_ts.value)} kWh")
            print(f"The capacity of the battery is: {int(self.Cap_bat.value)} kWh")

        return float(self.cost.value), float(self.co2.value)
    
    def pie_chart_electricity_gen(self, pie_ax, pv, chp, bat, grid, pietitle = 'Electricity generation'):
        p_out = [pv, chp, bat, grid]
        labels = ["PV", "CHP", "Battery out", "Electricity grid"]
        pie_ax.pie(p_out, labels=labels, autopct='%1.1f%%', startangle=90)
        pie_ax.set_title(pietitle)
        return
    
        
    def plot_data(self, xlim = 8760, xmin=0):
        """
        Plot the optimal energy system operation results
        """
        t = np.arange(len(self.heat_demand))

        """
        # Plot electricity by component
        colorList = ['#FFA500', '#FF8C00', '#FF5733', '#8B4513', '#A0522D', '#D2691E']
        # colorList = ['#ffdb8d', '#f4bcd5', '#d1afd2', '#51b7d6', '#7accc8', '#0c8e85', '#b5ce78', '#7ebf00', '#fac4a8', '#b7b7b7', '#577fbd']

        fig = plt.figure(figsize=(16, 10))
        ax = fig.add_subplot()

        ax.fill_between(t, self.P_out_pv.value,
                        0,
                        color=colorList[0], linewidth=0, label='PV')
        ax.fill_between(t, self.Q_out_bat.value + self.P_out_pv.value,
                        self.P_out_pv.value,
                        color=colorList[1], linewidth=0, label='Battery out')
        ax.fill_between(t, self.P_out_elec_chp.value + self.P_out_pv.value + self.Q_out_bat.value,
                        self.P_out_pv.value + self.Q_out_bat.value,
                        color=colorList[2], linewidth=0, label='CHP')


        ax.fill_between(t, -self.P_in_gshp.value,
                        0,
                        color=colorList[3], linewidth=0, label='Heat pump')
        ax.fill_between(t, -self.P_in_gshp_cool.value - self.P_in_gshp.value,
                        -self.P_in_gshp.value,
                        color=colorList[4], linewidth=0, label='Heat pump for cooling')
        ax.fill_between(t, -self.Q_in_bat.value - self.P_in_gshp_cool.value - self.P_in_gshp.value,
                        -self.P_in_gshp_cool.value - self.P_in_gshp.value,
                        color=colorList[5], linewidth=0, label='Battery in')
        ax.axhline(y=0., color='k', linewidth=0.3)
        ax.axhline(y=1.5, color='k', linewidth=0.3)

        ax.plot(t, self.elec_demand.value, 'k', linewidth=0.8, label='Demand')
        ax.plot(t, self.Imp_elec.value, 'b', linewidth=0.8, label='Grid import')

        ax.set_title('Electricity', fontsize=14)
        ax.set_xlabel('Time [h]', fontsize=12)
        ax.set_ylabel('Output [kW]', fontsize=12)
        ax.legend(loc="lower right")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, linestyle='--', linewidth=0.5)
        ax.set_xlim(0, xlim)
        """

        # Figure with subplots
        fig, axs = plt.subplots(2, 2, figsize=(20, 12))
        #colors = ['#ff0000', '#ff7f00', '#ffff00', '#7fff00', '#00ff00', '#00ffff', '#0000ff', '#8b00ff']
        colors = ['#FF0000', '#ff7f00', '#ffff00', '#006400', '#0000FF', '#2E2B5F', '#8B00FF', '#FF00FF']
        linewidth = 0.8
        alpha = 0.3

        # Electricity demand
        axs[0, 0].plot(t, self.elec_demand.value, color=colors[0], linewidth=linewidth, label='Electricity demand')
        axs[0, 0].fill_between(t, self.elec_demand.value, color=colors[0], alpha=alpha)
        axs[0, 0].plot(t, self.P_in_gshp.value, color=colors[2], linewidth=linewidth, label='GSHP')
        axs[0, 0].fill_between(t, self.P_in_gshp.value, color=colors[2], alpha=alpha)
        axs[0, 0].plot(t, self.P_in_gshp_cool.value, color=colors[1], linewidth=linewidth, label='GSHP cool')
        axs[0, 0].fill_between(t, self.P_in_gshp_cool.value, color=colors[1], alpha=alpha)
        axs[0, 0].plot(t, self.Q_in_bat.value, color=colors[3], linewidth=linewidth, label='Battery in')
        axs[0, 0].fill_between(t, self.Q_in_bat.value, color=colors[3], alpha=alpha)
        axs[0, 0].plot(t, self.elec_demand.value + self.P_in_gshp.value + self.P_in_gshp_cool.value + self.Q_in_bat.value,
                       color='k', linewidth=linewidth, label='Total demand (sum)')
        axs[0, 0].fill_between(t, self.elec_demand.value + self.P_in_gshp.value + self.P_in_gshp_cool.value + self.Q_in_bat.value,
                               color='k', alpha=0.1)

        axs[0, 0].set_xlabel('Time [h]', fontsize=12)
        axs[0, 0].set_ylabel('Output [kW]', fontsize=12)
        axs[0, 0].set_title('Electricity demand', fontsize=14)
        axs[0, 0].legend(fontsize=10, loc='upper right')
        axs[0, 0].grid(True, linestyle='--', linewidth=0.5)

        # Electricity import / production
        axs[1, 0].plot(t, self.P_out_pv.value, color=colors[1], linewidth=linewidth, label='PV')
        axs[1, 0].fill_between(t, self.P_out_pv.value, color=colors[1], alpha=0.5)
        axs[1, 0].plot(t, self.Q_out_bat.value, color=colors[3], linewidth=linewidth, label='Battery out')
        axs[1, 0].fill_between(t, self.Q_out_bat.value, color=colors[3], alpha=alpha)
        axs[1, 0].plot(t, self.P_out_elec_chp.value, color=colors[4], linewidth=linewidth, label='CHP')
        axs[1, 0].fill_between(t, self.P_out_elec_chp.value, color=colors[4], alpha=alpha)
        axs[1, 0].plot(t, self.Imp_elec.value, color=colors[6], linewidth=linewidth, label='Electricity grid')
        axs[1, 0].fill_between(t, self.Imp_elec.value, color=colors[6], alpha=alpha)

        axs[1, 0].set_xlabel('Time [h]', fontsize=12)
        axs[1, 0].set_ylabel('Output [kW]', fontsize=12)
        axs[1, 0].set_title('Electricity Import / Production', fontsize=14)
        axs[1, 0].legend(fontsize=10, loc='upper right')
        axs[1, 0].grid(True, linestyle='--', linewidth=0.5)


        # Heat node subplot
        axs[0, 1].plot(t, self.P_out_heat_chp.value, color=colors[4], linewidth=linewidth, label='CHP')
        axs[0, 1].fill_between(t, self.P_out_heat_chp.value, color=colors[4], alpha=alpha)
        axs[0, 1].plot(t, self.P_out_gshp.value, color=colors[2], linewidth=linewidth, label='GSHP')
        axs[0, 1].fill_between(t, self.P_out_gshp.value, color=colors[2], alpha=alpha)
        axs[0, 1].plot(t, self.P_out_ob.value, color=colors[7], linewidth=linewidth, label='Oil boiler')
        axs[0, 1].fill_between(t, self.P_out_ob.value, color=colors[7], alpha=alpha)
        axs[0, 1].plot(t, self.Q_out_ts.value-self.Q_in_ts.value, color=colors[0], linewidth=linewidth, label='Storage tank out')
        axs[0, 1].fill_between(t, self.Q_out_ts.value-self.Q_in_ts.value, color=colors[0], alpha=alpha)
        # axs[0, 1].plot(t, -self.Q_in_ts.value, color=colors[0], linewidth=linewidth, label='Storage tank in')
        # axs[0, 1].fill_between(t, -self.Q_in_ts.value, color=colors[0], alpha=alpha)
        axs[0, 1].plot(t, self.heat_demand, color='k', linewidth=linewidth, label='Heat demand')

        axs[0, 1].set_xlabel('Time [h]', fontsize=12)
        axs[0, 1].set_ylabel('Output [kW]', fontsize=12)
        axs[0, 1].set_title('Heat node', fontsize=14)
        axs[0, 1].legend(fontsize=10, loc='upper right')
        axs[0, 1].grid(True, linestyle='--', linewidth=0.5)


        # Oil subplot
        axs[1, 1].plot(t, self.P_in_chp.value, color=colors[4], linewidth=linewidth, label='CHP')
        axs[1, 1].fill_between(t, self.P_in_chp.value, color=colors[4], alpha=alpha)
        axs[1, 1].plot(t, self.P_in_ob.value, color=colors[7], linewidth=linewidth, label='Oil boiler')
        axs[1, 1].fill_between(t, self.P_in_ob.value, color=colors[7], alpha=alpha)
        axs[1, 1].plot(t, self.Imp_oil.value, color='k', linewidth=linewidth, label='Oil import')
        axs[1, 1].set_xlabel('Time [h]', fontsize=12)
        axs[1, 1].set_ylabel('Output [kW]', fontsize=12)
        axs[1, 1].set_title('Diesel Consumption', fontsize=14)
        axs[1, 1].legend(fontsize=10, loc='upper right')
        axs[1, 1].grid(True, linestyle='--', linewidth=0.5)

        axs[0, 0].set_xlim(xmin, xlim)
        axs[0, 1].set_xlim(xmin, xlim)
        axs[1, 0].set_xlim(xmin, xlim)
        axs[1, 1].set_xlim(xmin, xlim)
        plt.show()
        return
    

    def pareto_front(self, granularity=6):
        # run optimisation for cost and for co2 to have the beginning and the end point of the pareto curve
        fig1, (ax1,ax2) = plt.subplots(1,2)
        cost_min, co2_max = self.run_optimisation(objective='cost')
        self.pie_chart_electricity_gen(ax1, self.P_out_pv.value.sum(), self.P_out_elec_chp.value.sum(),self.Q_out_bat.value.sum(), self.Imp_elec.value.sum(), 'Electricity generation: optimized cost')
        
        cost_max, co2_min = self.run_optimisation(objective='co2')
        self.pie_chart_electricity_gen(ax2, self.P_out_pv.value.sum(), self.P_out_elec_chp.value.sum(),self.Q_out_bat.value.sum(), self.Imp_elec.value.sum(), 'Electricity generation: optimized CO2')
        
        plt.show()

        cost_array = [cost_max]
        co2_array = [co2_min]

        # Multi objective opt & pareto front (E-constraint method)
        # =========================================================

        #for i in np.linspace((1/(granularity + 1)), (granularity/(granularity + 1)), granularity):
            # e_co2 = co2_max - (co2_max - co2_min) * i
        set = self.non_lin_space(co2_min, co2_max, granularity)
        for co2_value in set:
            # Set the additional constraints
            e_con = [self.co2 >= 0, self.co2 <= co2_value]

            # Optimize the design of the energy system with the cost objective but with the constraint for the co2 emissions
            cost_i, co2_i = self.run_optimisation(objective='cost', opt_constraints=self.constraints+e_con)
            cost_array.append(cost_i)
            co2_array.append(co2_i)



        cost_array.append(cost_min)
        co2_array.append(co2_max)


        print(cost_array)
        print(co2_array)

        # Plot the data
        plt.figure(figsize=(8, 6))
        plt.plot(cost_array, co2_array, color='b', linewidth=2)
        plt.fill_between(cost_array, co2_array, y2=(np.min(co2_array)-(np.max(co2_array) - np.min(co2_array))/100), color='#ADD8E6')
        plt.xlabel('System lifetime cost (USD)', fontsize=12)
        plt.ylabel('System lifetime emissions (CO2eq)', fontsize=12)
        plt.title('Pareto front', fontsize=14)
        plt.grid(True, linestyle='--', linewidth=0.5)
        plt.legend(['Pareto front'], loc='upper right', fontsize=10)
        plt.tick_params(axis='both', which='major', labelsize=10)
        plt.show()

    def non_lin_space(self, start, end, n):
        step = (end - start) / ((n + 2) * (n + 1) / 2)
        positions = []
        for i in range(1, n + 1):
            positions.append(start + step * i * (i + 1) / 2)
        return positions


if __name__ == '__main__':
    main()
