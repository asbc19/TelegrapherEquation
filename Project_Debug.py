#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 19:33:48 2021

@author: asbc
"""

# Libraries

import numpy as np

import matplotlib.pyplot as plt


# Methods

"""
Determine the Total Number of Points in Time grid.

If we select the condition of equality (magic time step), we obtain the exact solution without considerable high frequency ringing.

v --> phase velocity
N --> total number of points in space grid
T_f --> simulation time
L --> Transmission Line length

Return M
"""
def Time_Mpoint(v, N, T_f, L, vrb = False):
    
    # Determine the number of points in the time grid (magical time step)
    M = N*int(v*T_f/L)
    
    if vrb:
        print(f'Number of Total Points in Time Grid: {M}\n')
    
    return M


"""
Determine the temporal and spatial time steps

M --> total number of points in time grid
N --> total number of points in space grid
T_f --> simulation time
L --> Transmission Line length

return dz, dt
"""
def g_step(M, N, T_f, L, vrb = False):
    
    # Spacing
    dz = L/N
    
    # Time step
    dt = T_f/M
    
    if vrb:
        print(f'Spatial Grid Spacing: {dz}\n')
        print(f'Time step: {dt}\n')
    
    return dz, dt

###############
# Spatial Grid
###############

# In Python, indexes start with 0. Hence, we need to change the grid limits to match with the indexes.

"""
Create spatial grid for Voltage and Current

For accuracy and stability E and H field solutions are not at the same point but staggered one half-cell apart

N --> total number of points in space grid
L --> Transmission Line length

Return --> v_gz and i_gz
"""
def space_grid(N, L, dz, vrb = False):
    
    # Voltage Spatial Grid has N+1 points
    v_gz = np.linspace(0, L, N+1)
    
    # Current Spatial Grid starts at dz/2 and has N points
    i_gz = np.linspace(0.5*dz, L-(0.5*dz), N)
    
    if vrb:
        print(f'Voltage Spatial Grid: \n {v_gz}')
        print(f'and shape {v_gz.shape} \n')
        print(f'Current Spatial Grid: \n {i_gz}')
        print(f'and shape {i_gz.shape} \n')
        
    return v_gz, i_gz

################
# Temporal Grid
################

"""
Create temporal grid for Voltage and Current

For accuracy and stability E and H field solutions are not at the same point but staggered one half-cell apart

M --> total number of points in time
T_f --> Simulation time

Return --> v_gt and i_gt
"""
def time_grid(M, T_f, dt, vrb = False):
    
    # Voltage Temporal Grid has M+1 points
    v_gt = np.linspace(0, T_f, M+1)
    
    # Current Temporal Grid starts at dt/2 and has M points
    i_gt = np.linspace(0.5*dt, T_f-(0.5*dt), M)
    
    if vrb:
        print(f'Voltage Temporal Grid: \n {v_gt}')
        print(f'and shape {v_gt.shape} \n')
        print(f'Current Temporal Grid: \n {i_gt}')
        print(f'and shape {i_gt.shape} \n')
        
    return v_gt, i_gt


# Source Voltage

"""
Generate the lumped source voltage considering the rise time (rt) of the signal

V_s --> Source Voltage
rt --> rise time (10-90 criterion)
dt --> time step
T_f --> Simulation time
M --> total number of points in time grid
Ideal --> No rise time signal

Return --> V_st
"""
def source_V(V_s, rt, dt, T_f, M, Ideal = True, vrb = False):
    
    # Initilize the signal
    V_st = V_s*np.ones(M+1)
    
    # Create an non-ideal signal --> rise time
    if not Ideal:
        # Define the time when the ramp ends
        t_3 = rt/0.8
        
        # Define the number of discrete points to replace with the ramp
        M_ramp = int(t_3/dt) + 1
        
        # Generate and replace the ramp values
        for i in range(M_ramp):
            V_st[i] = 0.8*(V_s/rt)*(i*dt)
            
    # Print the results for visual verification
    if vrb:
        print(f'Source Voltage: \n {V_st}')
        print(f'and shape: {V_st.shape}\n')
        
        # Simple plot
        
        # Get time grid
        v_gt, _ = time_grid(M, T_f, dt, vrb = False)
        
        # Set the plot --> size(width, height)
        fig1 = plt.figure(figsize=(10,8))

        plt.plot(v_gt, V_st, 'b', linewidth = 1.5)

        plt. title('Voltage Source Waveform', fontsize = 20)
        plt.xlabel('Time [s]', fontsize = 15)
        plt.ylabel('Voltage [V]', fontsize = 15)
        plt.grid()
        #plt.legend(['Trapezoidal', 'Midpoint'])

        plt.show()
    
    return V_st

##################################
# Voltage and Current Simulation
##################################

"""
Simulate a two-conductor transmission line (lossless)

v --> phase velocity
Z_0 --> characteristic impedance
R_s --> Source Impedance
R_L --> Load Impedance
V_s --> Source Voltage
rt --> rise time of source voltage
V_L --> Load Voltage

M --> total number of points in time grid
N --> total number of points in space grid
dz --> spacing
dt --> time step
T_f --> Simulation time

Idel --> not considering rt in the source voltage

Return --> V_TL and I_TL
"""
def TL_ll_simulation(v, Z_0, R_s, R_L, V_s, rt, V_L, M, N, dz, dt, T_f, Ideal = True, vrb = False):
    
    # Calculte inductance (l_i) and capacitance (c_c) from the formulas
    # Phase velocity --> v = 1/sqrt(lc)
    # Characteristic impedance --> Z_0 = sqrt(l/c)
    c_c = 1/(v*Z_0)          # Capacitance
    l_i = 1/(c_c*(v**2))     # Inductance
        
    # TL initial conditions (t=0) --> Zero current and voltage
    V_0 = np.zeros(N+1)
    I_0 = np.zeros(N)
    
    # Initialize Voltage and Current results
    # Shallow copy to create an independent object
    V_TL = np.copy(V_0.reshape((-1,1)))
    I_TL = np.copy(I_0.reshape((-1,1)))
    
    ##########################################################################
    # After store the initial values in the final matrix, 
    # V_0 and I_0 will be used to hold the on-going results of the calculation
    ##########################################################################
    
    # Initialize Source and Load Voltages (only a function of time)
    V_st = source_V(V_s, rt, dt, T_f, M, Ideal = Ideal, vrb = vrb)
    
    V_Lt = V_L*np.ones(M+1)
    
    V_TL[0][0] = V_st[0] 
    
    # Common factor used in the following calculations
    c_aux = (c_c*dz)/(2*dt)
    
    # Iterate for all time points
    for n in range(M):
        
        # Iterate for all space points
        # The 1st and last point have different formulas
        for k in range(N+1):
            
            #################
            # Solving Voltage
            #################
            if k == 0:
                V_0[k] = (1/((R_s*c_aux) + 0.5))*((((R_s*c_aux) - 0.5)*V_TL[k][n]) 
                                                  - (R_s*I_TL[k][n]) + (0.5*(V_st[n+1] + V_st[n])))
                
            elif k == N:
                V_0[k] = (1/((R_L*c_aux) + 0.5))*((((R_L*c_aux) - 0.5)*V_TL[k][n]) 
                                                  + (R_L*I_TL[k-1][n]) + (0.5*(V_Lt[n+1] + V_Lt[n])))
            
            else:
                V_0[k] = V_TL[k][n] - ((dt/(dz*c_c))*(I_TL[k][n] - I_TL[k-1][n]))
                
            
            #################
            # Solving Current
            #################
            # Current vector only has N points
            # and it is calculated based on the two
            # previous voltages
            if k >= 1:
                
                # Calculate current element with modified index
                I_0[k-1] = I_TL[k-1][n] - ((dt/(dz*l_i))*(V_0[k] - V_0[k-1]))
        
        # Add the solution to the final matrix
        V_TL = np.concatenate((V_TL, V_0.reshape(-1,1)), axis = 1)
        
        # Current Matrix only has M columns
        if n < M-1:
            I_TL = np.concatenate((I_TL, I_0.reshape(-1,1)), axis = 1)
            
    if vrb:
        print(f'TL Capacitance: {c_c} [F]\n')
        print(f'TL Inductance: {l_i} [H]\n')
        
        print(f'Final Simulated Voltage: \n {V_TL}')
        print(f'and shape: {V_TL.shape}\n')
        
        print(f'Final Simulated Current: \n {I_TL}')
        print(f'and shape: {I_TL.shape}\n')

    return V_TL, I_TL


"""
Main Method
"""
def main():
    # Constants

    L = 400 # length 400 [m]
    v = 2e8 # phase velocity [m/s]
    V_s = 30 # Source Voltage [V]
    rt = 0.1e-6 # Rise time [s]
    V_L = 0 # Load Voltage [V] --> Reference plane
    Z_0 = 50 # Characteristic impedance [ohm]
    R_s = 0 # Source Impedance [ohm]
    R_L = 100 # Load Impedance [ohm]
    
    # Grid Points
    N = 200 # Space grid
    T_f = 20e-6 # Simulation time [s]
    
    M = Time_Mpoint(v, N, T_f, L, vrb = True)
    dz, dt = g_step(M, N, T_f, L, vrb = True)
    
    v_gz, i_gz = space_grid(N, L, dz, vrb = True)
    v_gt, i_gt = time_grid(M, T_f, dt, vrb = True)
    
    #V_st = source_V(V_s, rt, dt, T_f, M, Ideal = False, vrb = True)
    
    # Electric Constants
    electr_c = (v, Z_0, R_s, R_L, V_s, rt, V_L)
    
    # Grid Constants
    grid_c = (M, N, dz, dt, T_f)
    
    # Simulate the lossless Line
    V_TL, I_TL = TL_ll_simulation(*electr_c, *grid_c, Ideal = True, vrb = True)
    
    return V_TL, I_TL, v_gt
    

# Main Function to run the application
if __name__ == "__main__":
    V_TL, I_TL, v_gt = main()
    
    # Simple plot
        
    # Set the plot --> size(width, height)
    fig1 = plt.figure(figsize=(10,8))

    plt.plot(v_gt, V_TL[0][:], '--b', v_gt, V_TL[-1][:], 'r', linewidth = 1.5)

    plt. title('Voltages', fontsize = 20)
    plt.xlabel('Time [s]', fontsize = 15)
    plt.ylabel('Voltage [V]', fontsize = 15)
    plt.grid()
    plt.legend(['Source', 'Load'])

    plt.show()