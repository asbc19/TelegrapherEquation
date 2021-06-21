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
    # Modify the length units to reduce the time points
    
    M = N*int(v*T_f/(L))
    
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
# for multiple lines
##################################

"""
Simulate a multiple conductor transmission line (lossless)
We only consider two conductors with the same return path

line 0: return path
line 1: victim
line 2: aggressor


R_sm --> Source Impedance Matrix
R_Lm --> Load Impedance Matrix
C_m --> Capcitance Matrix
L_m --> Inductance Matrix
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
def multiTL_simulation(R_sm, R_Lm, C_m, L_m, V_s, rt, V_L, M, N, dz, dt, T_f, Ideal = True, vrb = False):
    
    # Increase one more step of time simulation to compensate the repetition
    # of the 1st state
    #M += 1
    #T_f += dt
    
    # Number of conductors
    n_c = 2
        
    # TL initial conditions (t=0) --> Zero current and voltage
    
    V_0 = np.zeros((n_c, N+1))
    I_0 = np.zeros((n_c, N))
    
    
    # We need 3 dimensions to store the results for each conductor
    # We use concatenate, with this function the values of the new
    # array are independent from the original (no shallow copy needed)
   
    V_TL = np.concatenate((V_0[0,:].reshape((1, -1, 1)), 
                           V_0[0,:].reshape((1, -1, 1))), axis = 0)
    
    I_TL = np.concatenate((I_0[0,:].reshape((1, -1, 1)), 
                           I_0[0,:].reshape((1, -1, 1))), axis = 0)
    
    ##########################################################################
    # After store the initial values in the final matrix, 
    # V_0 and I_0 will be used to hold the on-going results of the calculation
    ##########################################################################
    
    # Initialize Source and Load Voltages (only a function of time)
    V_st = source_V(V_s, rt, dt, T_f, M, Ideal = Ideal, vrb = vrb)
    
    V_Lt = V_L*np.ones(M+1)
    
    # Only the line 2 (rightmost line) has a source connected to it (aggressor)
    #V_TL[1,0,0] = 0.5*V_st[0] 
    
    # The Sorce and Load Voltages are represented as matrices
    V_sm = np.concatenate((np.zeros_like(V_st.reshape((1,-1))), 
                           V_st.reshape((1,-1))), axis = 0)     # Line 2: Voltage Source
    V_Lm = np.concatenate((V_Lt.reshape((1,-1)), 
                           V_Lt.reshape((1,-1))), axis = 0)
    
    # Common factor used in the following calculations
    r_aux = dz/dt
    
    # Iterate for all time points
    for n in range(M):
        
        # Iterate for all space points
        # The 1st and last point have different formulas
        for k in range(N+1):
            
            #################
            # Solving Voltage
            #################
            if k == 0:
                # Recover voltage and current from the 3D matrix solution
                V_n_1 = np.array([[V_TL[0,k,n]], [V_TL[1,k,n]]])
                I_n5_1 = np.array([[I_TL[0,k,n]], [I_TL[1,k,n]]])
                
                V_0[:,k] = (np.linalg.inv((r_aux*(R_sm@C_m)) + 1)@((((r_aux*(R_sm@C_m)) - 1)@V_n_1) 
                                    - (2*(R_sm@I_n5_1)) 
                                    + (V_sm[:,n+1].reshape((-1,1)) 
                                       + V_sm[:,n].reshape((-1,1))))).flatten()
                
            elif k == N:
                # Recover voltage and current from the 3D matrix solution
                V_n1_1 = np.array([[V_TL[0,k,n]], [V_TL[1,k,n]]])
                I_n5_2 = np.array([[I_TL[0,k-1,n]], [I_TL[1,k-1,n]]])
                
                V_0[:,k] = (np.linalg.inv((r_aux*(R_Lm@C_m)) + 1)@((((r_aux*(R_Lm@C_m)) - 1)@V_n1_1) 
                                    + (2*(R_Lm@I_n5_2)) 
                                    + (V_Lm[:,n+1].reshape((-1,1)) 
                                       + V_Lm[:,n].reshape((-1,1))))).flatten()
            
            else:
                # Recover voltage and currents from the 3D matrix solution
                V_n_k = np.array([[V_TL[0,k,n]], [V_TL[1,k,n]]])
                I_n5_k = np.array([[I_TL[0,k,n]], [I_TL[1,k,n]]])
                I_n5_k1 = np.array([[I_TL[0,k-1,n]], [I_TL[1,k-1,n]]])
                
                V_0[:,k] = (V_n_k - (r_aux*np.linalg.inv(C_m)@(I_n5_k + I_n5_k1))).flatten()
                
            
            #################
            # Solving Current
            #################
            # Current vector only has N points
            # and it is calculated based on the two previous voltages
            
            if k >= 1:
                
                # Recover the current from the 3D matrix solution
                I_n5_k = np.array([[I_TL[0,k-1,n]], [I_TL[1,k-1,n]]])
                
                # Calculate current element with modified index
                I_0[:,k-1] = (I_n5_k - (r_aux*np.linalg.inv(L_m)@(V_0[:,k].reshape((-1,1)) 
                                        + V_0[:,k-1].reshape((-1,1))))).flatten()
        
        
        # Reshape the results to concatenate in the last matrix
        V_aux = np.concatenate((V_0[0,:].reshape((1,-1,1)), V_0[1,:].reshape((1,-1,1))), axis = 0)
        I_aux = np.concatenate((I_0[0,:].reshape((1,-1,1)), I_0[1,:].reshape((1,-1,1))), axis = 0)
        
        # Add the solution to the final 3D matrix
        V_TL = np.concatenate((V_TL, V_aux), axis = 2)
        
        # Current Matrix only has M columns
        if n < M-1:
            I_TL = np.concatenate((I_TL, I_aux), axis = 2)
            
    if vrb:
        
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

    L = 0.254 # length [m]
    
    # Two Mode Velocities
    # we use the smaller, v1
    v1 = 1.80065e8 # phase velocity [m/s]
    v2 = 1.92236e8 # phase velocity [m/s]
    
    # Voltage source is represented as a column vector
    V_s = 1 # Source Voltage [V]
    rt = 6.25e-9 # Rise time [s]
    
    # Load Voltage is also represented as a column vector
    V_L = 0 # Load Voltage [V] --> Reference plane
    
    # Source and Load Impedance are represented as matrixes 
    R_s = 50 # Source Impedance [ohm]
    R_L = 50 # Load Impedance [ohm]
    
    R_sm = R_s*np.identity(2)
    print(f'Source Impedance Matrix: \n {R_sm}')
    print(f'and shape: {R_sm.shape} \n')
    R_Lm = R_L*np.identity(2)
    print(f'Load Impedance Matrix: \n {R_Lm}')
    print(f'and shape: {R_Lm.shape} \n')
    
    # Simulation time [s]
    T_f = 40e-9 
    
    # Per-unit-length Capacitance and Inductance Matrices
    C_m = (1e-12)*np.array([[40.6280, -20.3140], [-20.3140, 29.7632]])
    print(f'Capcitance Matrix: \n {C_m}')
    print(f'and shape: {C_m.shape} \n')
    
    L_m = (1e-6)*np.array([[1.10418, 0.690094], [0.690094, 1.38019]])
    print(f'Inductance Matrix: \n {L_m}')
    print(f'and shape: {L_m.shape} \n')
        
    # Grid Points
    N = 2 # Space grid
    
    M = Time_Mpoint(v2, N, T_f, L, vrb = True)
    dz, dt = g_step(M, N, T_f, L, vrb = True)
    
    v_gz, i_gz = space_grid(N, L, dz, vrb = True)
    v_gt, i_gt = time_grid(M, T_f, dt, vrb = True)
    
    
    # Electric Constants
    electr_c = (R_sm, R_Lm, C_m, L_m, V_s, rt, V_L)
    
    # Grid Constants
    grid_c = (M, N, dz, dt, T_f)
    
    # Simulate the lossless Line
    V_TL, I_TL = multiTL_simulation(*electr_c, *grid_c, Ideal = True, vrb = True)
    
    return V_TL, I_TL, v_gt
    

# Main Function to run the application
if __name__ == "__main__":
    V_TL, I_TL, v_gt = main()
    
    # Simple plot
        
    # Set the plot --> size(width, height)
    fig1 = plt.figure(figsize=(10,8))

    plt.plot(v_gt, V_TL[1,0,:], '--b', v_gt, V_TL[0,0,:], 'r', linewidth = 1.5)

    plt. title('Voltages', fontsize = 20)
    plt.xlabel('Time [s]', fontsize = 15)
    plt.ylabel('Voltage [V]', fontsize = 15)
    plt.grid()
    plt.legend(['Source', 'Near-End Victim'])

    plt.show()