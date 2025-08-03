"""
Implementation of a Model for an MEC
Oliver Roy Mangosing
August 2025

Implementation is based on the MEC model proposed by [1]. For testing, values of parameters were mainly taken from [1],
with missing parameter values taken from [2], [3], and [4].

References:
    [1] R. P. Pinto, B. Srinivasan, and B. Tartakovsky,
    “A UNIFIED MODEL FOR ELECTRICITY AND HYDROGEN PRODUCTION IN MICROBIAL ELECTROCHEMICAL CELLS,” IFAC Proceedings
    Volumes, vol. 44, no. 1, pp. 5046–5051, Jan. 2011, doi: 10.3182/20110828-6-it-1002.01636.
    [2] V. Alcaraz–Gonzalez, G. Rodriguez–Valenzuela, J. J. Gomez–Martinez, G. L. Dotto, and R. A. Flores–Estrella,
    “Hydrogen production automatic control in continuous microbial electrolysis cells reactors used in wastewater
    treatment,” Journal of Environmental Management, vol. 281, p. 111869, Dec. 2020, doi: 10.1016/j.jenvman.2020.111869.
    [3] M. Z. U. Rahman, M. Rizwan, R. Liaquat, V. Leiva, and M. Muddasar, “Model-based optimal and robust control of
    renewable hydrogen gas production in a fed-batch microbial electrolysis cell,” International Journal of Hydrogen
    Energy, vol. 48, no. 79, pp. 30685–30701, May 2023, doi: 10.1016/j.ijhydene.2023.04.184.
    [4] R. A. Flores-Estrella, U. De Jesús Garza-Rubalcava, A. Haarstrick, and V. Alcaraz-González, “A dynamic biofilm
    model for a microbial electrolysis cell,” Processes, vol. 7, no. 4, p. 183, Mar. 2019, doi: 10.3390/pr7040183.

"""

import scipy
import numpy as np
import matplotlib.pyplot as plt


def solve_ke(u_max, K_A, q_max, A, M_ox, K_M):
    # Solves the kinetic equations

    # Unpack required parameters
    u_maxa, u_maxm = u_max
    q_maxa, q_maxm = q_max
    K_Aa, K_Am = K_A

    # Solve equations
    u_a = u_maxa * A * M_ox / ((K_Aa + A) * (K_M + M_ox)) # (15)
    u_m = u_maxm * A / (K_Am + A) # (16)

    q_a = q_maxa * A * M_ox / ((K_Aa + A) * (K_M + M_ox)) # (17)
    q_m = q_maxm * A / (K_Am + A) # (18)

    # Return results
    return [u_a, u_m, q_a, q_m]

def solve_uhc(u_maxh, H_2, K_h):
    # Solves for the growth rate of hydrogenotrophic methanogens

    # Solve equation
    u_h = u_maxh * H_2  / (K_h + H_2) # (19) with Q_H2 > 0 for MECs

    # Return results
    return u_h

def solve_a(u, K, x, X_max):
    # Solves for the biofilm retention constants \alpha

    # Unpack required parameters
    u_a, u_m, u_h = u
    K_da, K_dm, K_dh = K
    x_a, x_m, x_h = x
    X_max1, X_max2 = X_max

    # Solve equation
    if x_a + x_m >= X_max1:
        a_1 = ((u_a - K_da) * x_a + (u_m - K_dm) * x_m) / (x_a + x_m) # (9)
    else:
        a_1 = 0

    if x_h >= X_max2:
        a_2 = u_h - K_dh # (9)
    else:
        a_2 = 0

    # Return results
    return [a_1, a_2]

def solve_M_red(M_Total, M_ox):
    # Solves for reduced mediator fraction

    # Solve equation
    M_red =  M_Total - M_ox # (13)

    # Return results
    return M_red

def solve_elrxn(T, M_red, m, R_MIN, R_MAX, I_0, beta, M_Total, A_surA, E_CEF, K_R, x_a):
    # Solves the electrochemical reactions

    # Define constants in SI units
    R = 8.314 # Ideal gas constant (J/mol K)
    F = 96485 # Faraday constant (s A/mol)
    e = 0 # Set to zero as stated in [1]

    # Solve preliminary equations
    n_conc = R * T * np.log(M_Total / M_red) / (m * F) # (21)
    R_int = R_MIN + (R_MAX - R_MIN) * np.exp(-K_R * x_a) # (25)

    # Define a system of equations composed of (22), Ohm's law, (20), negative voltage, and (24)
    # This system is to be solved by root finding, that is, LHS = 0
    # Some equations were rewritten to avoid division
    def system(x):
        I_MEC, n_act, n_ohm, E, E_applied = x # Unpack values
        eq22 = R * T * np.arcsinh(I_MEC / (A_surA * I_0)) - n_act * beta * m * F # (22)
        eq205 = I_MEC * R_int - n_ohm # Ohm's law as stated in p. 5048 of [1]
        eq20 = E_CEF - n_ohm - n_conc - n_act - E # (20)
        eq225 = -E - E_applied # Negative voltage as stated in p. 5048 of [1]
        eq_24 = (E_CEF + E_applied - R * T / (m * F) * np.log(M_Total / M_red) - n_act * I_MEC) * M_red - I_MEC * (R_int * (e + M_red)) # (24)
        return [eq22, eq205, eq20, eq225, eq_24]
    # Solve the system with initial guess values of 0
    res = scipy.optimize.root(system, [0, 0, 0, 0, 0], tol=1E-9)
    I_MEC, n_act, n_ohm, E, E_applied = res.x # Unpack the solution

    # Return results
    return [n_conc, R_int, I_MEC, n_act, n_ohm, E, E_applied]


def solve_de(t, y, u_maxa, u_maxm, K_Aa, K_Am, q_maxa, q_maxm, K_M, F_in, V, A_0, K_da, K_dm, K_dh, u_maxh, H_2, K_h, X_max1, X_max2, gamma, m, T, M_Total, R_MIN, R_MAX, I_0, beta, A_surA, E_CEF, K_R, Y_M):
    # Solves the main system of differential equations

    # Define constants
    F = 96485 # Faraday constant (s A/mol)

    # Unpack solution
    A, x_a, x_m, x_h, M_ox = y

    # Pack input parameters and solve the kinetic equations (solve_ke) and growth rate (solve_uhc)
    u_max = [u_maxa, u_maxm]
    K_A = [K_Aa, K_Am]
    q_max = [q_maxa, q_maxm]
    u_a, u_m, q_a, q_m = solve_ke(u_max, K_A, q_max, A, M_ox, K_M)
    u_h = solve_uhc(u_maxh, H_2, K_h)

    # Pack input parameters and solve for \alpha (solve_a), M_red (solve_M_red), and electrochemical reactions (solve_elrxn)
    u = [u_a, u_m, u_h]
    K = [K_da, K_dm, K_dh]
    x = [x_a, x_m, x_h]
    X_max = [X_max1, X_max2]
    a_1, a_2 = solve_a(u, K, x, X_max)
    M_red = solve_M_red(M_Total, M_ox)
    n_conc, R_int, I_MEC, n_act, n_ohm, E, E_applied = solve_elrxn(T, M_red, m, R_MIN, R_MAX, I_0, beta, M_Total, A_surA, E_CEF, K_R, x_a)

    # Solve for the dilution rate
    D = F_in / V

    # Return values of the differential equations
    dAdt = -q_a * x_a - q_m * x_m + D * (A_0 - A)  # (5)
    dx_adt = u_a * x_a - K_da * x_a - a_1 * x_a  # (6)
    dx_mdt = u_m * x_m - K_dm * x_m - a_1 * x_m  # (7)
    dx_hdt = u_h * x_h - K_dh * x_h - a_2 * x_h  # (8)
    dM_oxdt = -Y_M * q_a + gamma * I_MEC / (V * x_a * m * F)  # (14)
    return [dAdt, dx_adt, dx_mdt, dx_hdt, dM_oxdt]

def solve_Q(Y, q_m, x, Y_h, u_h, V, I_MEC, T, m, P):
    # Solves for the production rates

    # Define cosntants
    R = 8.314 # Ideal gas constant (J/mol K)
    F = 96485 # Faraday constant (s A/mol)

    # Unpack required parameters
    Y_CH4, Y_H2CH4, Y_H2 = Y
    x_m, x_h = x

    # Solve equations
    Q_CH41 = Y_CH4 * q_m * x_m * V # (10)
    Q_CH42 = Y_H2CH4 * Y_h * u_h * x_h * V # (11)
    Q_H2 = Y_H2 * (I_MEC * R * T / (m * F * P)) - Y_h * u_h * x_h * V # (12)

    # Return values
    return [Q_CH41, Q_CH42, Q_H2]

def main():
    # Main script. Solves the model containing the differential equations and other equations

    # Parameters in SI units
    Y_CH4 = 0.28 # mL/mg -> m3/kg [1]
    q_maxm = 14.12 * (1 / 24) * (1 / 3600) # 1/d -> 1/s [1]
    q_maxa = 13.14 * (1 / 24) * (1 / 3600) # 1/d -> 1/s [1]
    u_maxa = 1.97 * (1 / 24) * (1 / 3600) # 1/d -> 1/s [1]
    u_maxm = 0.3 * (1 / 24) * (1 / 3600) # 1/d -> 1/s [1]
    u_maxh = 0.5 * (1 / 24) * (1 / 3600) # 1/d -> 1/s [1]
    K_Aa = 20 * (1 / 1000) * (1 / 1000) * (1000 / 1) # mg/L -> kg/m3 [1]
    K_Am = 80 * (1 / 1000) * (1 / 1000) * (1000 / 1) # mg/L -> kg/m3 [1]
    m = 2 # mol/mol [1]
    gamma = 663400 * (1 / 1000) * (1 / 1000) # mg/mol -> kg/mol [1]
    M_Total = 1000 # mg/mg [2]
    K_M = 0.01 * (1 / 1000) * (1 / 1000) * (1000 / 1) # mg/L -> kg/m3 [1]
    K_da = 0.04 * (1 / 24) * (1 / 3600) # 1/d -> 1/s [1]
    K_dm = 0.006 * (1 / 24) * (1 / 3600) # 1/d -> 1/s [1]
    K_dh = 0.01 * (1 / 24) * (1 / 3600) # 1/d -> 1/s [1]
    K_h = 0.0001 * (1 / 1000) * (1 / 1000) * (1000 / 1) # mg/L -> kg/m3 [1]
    H_2 = 1 * (1 / 1000) * (1 / 1000) * (1000 / 1) # mg/L -> kg/m3 [1]
    Y_H2CH4 = 0.25 # mL/mL [1]
    Y_H2 = 0.9 # [1]
    X_max1 = 512.5 * (1 / 1000) * (1 / 1000) * (1000 / 1) # mg/L -> kg/m3 [1]
    X_max2 =  1215 * (1 / 1000) * (1 / 1000) * (1000 / 1) # mg/L -> kg/m3 [2]
    Y_h = 0.05 # mL/mg -> m3/kg [2]
    A_surA = 0.01 # m^2 [2]
    R_MIN = 20 # ohm [1]
    R_MAX = 2000 # ohm [1]
    K_R = 0.024 * (1 / 1000) * (1000 / 1) * (1000 / 1) # L/mg -> m3/kg [1]

    Y_M = 34.85 # mg/mg [1]
    E_CEF = -0.35 # V [1]
    beta = 0.5 # [1]
    I_0 = 0.005 # A/m2 [1]
    T = 298.15 # K [3]
    V = 250 * (1 / 1000) * (1 / 1000) # mL -> m3 [3]
    F_in = 400 * (1 / 1000) * (1 / 1000) * (1 / 24) * (1 / 3600) # mL/d -> m3/s [4]
    A_0 = 500 * (1 / 1000) * (1 / 1000) * (1000 / 1) # mg/L -> kg/m3 [1]
    P = 1 * (101325) # atm -> Pa [3]

    # Initial conditions
    A0 = 2000 * (1 / 1000) * (1 / 1000) * (1000 / 1) # mg/L -> kg/m3 [3]
    x_a0 = 1 * (1 / 1000) * (1 / 1000) * (1000 / 1) # mg/L -> kg/m3 [3]
    x_m0 = 10 * (1 / 1000) * (1 / 1000) * (1000 / 1) # mg/L -> kg/m3 [3]
    x_h0 = 50 * (1 / 1000) * (1 / 1000) * (1000 / 1) # mg/L -> kg/m3 [3]
    M_ox0 = 100 # mg/mg [2]

    # Timespan for integration
    start = 0 # s
    end = 3 * 24 * 3600 # day -> s

    # Solve the differential equations
    sol = scipy.integrate.solve_ivp(solve_de, [start, end], [A0, x_a0, x_m0, x_h0, M_ox0], t_eval=np.linspace(start, end, 1000), method = "DOP853", rtol=1E-6, atol=1E-9, args=[u_maxa, u_maxm, K_Aa, K_Am, q_maxa, q_maxm, K_M, F_in, V, A_0, K_da, K_dm, K_dh, u_maxh, H_2, K_h, X_max1, X_max2, gamma, m, T, M_Total, R_MIN, R_MAX, I_0, beta, A_surA, E_CEF, K_R, Y_M])
    t = sol.t # Time points
    A, x_a, x_m, x_h, M_ox = sol.y # Values of the solution at t

    # Pack input parameters and solve kinetic equations (solve_ke), growth rate (solve_uhc), and reduced mediator fraction (solve_M_red) for each time t
    Y = [Y_CH4, Y_H2CH4, Y_H2]
    u_max = [u_maxa, u_maxm]
    q_max = [q_maxa, q_maxm]
    K_A = [K_Aa, K_Am]
    u_a, u_m, q_a, q_m = solve_ke(u_max, K_A, q_max, A, M_ox, K_M)
    x = [x_m, x_h]
    u_h = solve_uhc(u_maxh, H_2, K_h)
    M_red = solve_M_red(M_Total, M_ox)

    # Solve electrochemical reactions for each time t
    results = [] # Blank list
    for M_red_1, x_a_1 in zip(M_red, x_a): # Loop through each time t
        res = solve_elrxn(T, M_red_1, m, R_MIN, R_MAX, I_0, beta, M_Total, A_surA, E_CEF, K_R, x_a_1)
        results.append(res) # Append results to blank list
    results = np.array(results) # Convert list to numpy array
    # Slice from results
    n_conc = results[:,0]
    R_int = results[:,1]
    I_MEC = results[:,2]
    n_act = results[:,3]
    n_ohm = results[:,4]
    E = results[:,5]
    E_applied = results[:,6]

    # Solve for the production rates at each time t
    Q_CH41, Q_CH42, Q_H2 = solve_Q(Y, q_m, x, Y_h, u_h, V, I_MEC, T, m, P)

    # Return variables of interest (change as needed)
    return

print(main())