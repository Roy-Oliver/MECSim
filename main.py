import scipy
import numpy as np
import matplotlib.pyplot as plt


def solve_uq(u_max, K_A, q_max, A, M_ox, K_M):
    u_maxa, u_maxm = u_max
    q_maxa, q_maxm = q_max
    K_Aa, K_Am = K_A

    u_a = u_maxa * A * M_ox / ((K_Aa + A) * (K_M + M_ox))
    u_m = u_maxm * A / (K_Am + A)

    q_a = q_maxa * A * M_ox / ((K_Aa + A) * (K_M + M_ox))
    q_m = q_maxm * A / (K_Am + A)

    return [u_a, u_m, q_a, q_m]

def solve_u(u_maxh, H_2, K_h):
    u_h = u_maxh * H_2  / (K_h + H_2)

    return u_h

def solve_a(u, K, x, X_max):
    u_a, u_m, u_h = u
    K_da, K_dm, K_dh = K
    x_a, x_m, x_h = x
    X_max1, X_max2 = X_max

    if x_a + x_m >= X_max1:
        a_1 = ((u_a - K_da) * x_a + (u_m - K_dm) * x_m) / (x_a + x_m)
    else:
        a_1 = 0

    if x_h >= X_max2:
        a_2 = u_h - K_dh
    else:
        a_2 = 0

    return [a_1, a_2]

def solve_M(M_Total, M_ox):
    M_red =  M_Total - M_ox # (13)

    return M_red

def solve_En(T, M_red, m, R_MIN, R_MAX, I_0, beta, M_Total, A_surA, E_CEF, K_R, x_a):

    R = 8.314#0.0521 # L * atm / mol * K
    F = 96485 #* (1 / 3600) * (1 / 24)  # (A * d / mol)
    e = 0

    n_conc = R * T * np.log(M_Total / M_red) / (m * F)
    R_int = R_MIN + (R_MAX - R_MIN) * np.exp(-K_R * x_a)

    def system(x):
        I_MEC, n_act, n_ohm, E, E_applied = x
        eq22 = R * T * np.arcsinh(I_MEC / (A_surA * I_0)) - n_act * beta * m * F
        eq205 = I_MEC * R_int - n_ohm
        eq20 = E_CEF - n_ohm - n_conc - n_act - E
        eq225 = -E - E_applied
        eq_24 = (E_CEF + E_applied - R * T / (m * F) * np.log(M_Total / M_red) - n_act * I_MEC) * M_red - I_MEC * (R_int * (e + M_red))
        return [eq22, eq205, eq20, eq225, eq_24]

    I_MEC, n_act, n_ohm, E, E_applied = scipy.optimize.fsolve(system, [0, 0, 0, 0, 0], xtol=1E-9)

    return [n_conc, R_int, I_MEC, n_act, n_ohm, E, E_applied]


def solve_de(t, y, u_maxa, u_maxm, K_Aa, K_Am, q_maxa, q_maxm, K_M, F_in, V, A_0, K_da, K_dm, K_dh, u_maxh, H_2, K_h, X_max1, X_max2, gamma, m, T, M_Total, R_MIN, R_MAX, I_0, beta, A_surA, E_CEF, K_R, Y_M):
    F = 96485 #* (1 / 3600) * (1 / 24)  # (A * d / mol)
    A, x_a, x_m, x_h, M_ox = y


    u_max = [u_maxa, u_maxm]
    K_A = [K_Aa, K_Am]
    q_max = [q_maxa, q_maxm]
    u_a, u_m, q_a, q_m = solve_uq(u_max, K_A, q_max, A, M_ox, K_M)

    D = F_in / V

    u_h = solve_u(u_maxh, H_2, K_h)

    u = [u_a, u_m, u_h]
    K = [K_da, K_dm, K_dh]
    x = [x_a, x_m, x_h]
    X_max = [X_max1, X_max2]
    a_1, a_2 = solve_a(u, K, x, X_max)


    M_red = solve_M(M_Total, M_ox)
    n_conc, R_int, I_MEC, n_act, n_ohm, E, E_applied = solve_En(T, M_red, m, R_MIN, R_MAX, I_0, beta, M_Total, A_surA, E_CEF, K_R, x_a)


    dAdt = -q_a * x_a - q_m * x_m + D * (A_0 - A)  # (5)
    dx_adt = u_a * x_a - K_da * x_a - a_1 * x_a  # (6)
    dx_mdt = u_m * x_m - K_dm * x_m - a_1 * x_m  # (7)
    dx_hdt = u_h * x_h - K_dh * x_h - a_2 * x_h  # (8)
    dM_oxdt = -Y_M * q_a + gamma * I_MEC / (V * x_a * m * F)  # (14)

    return [dAdt, dx_adt, dx_mdt, dx_hdt, dM_oxdt]

def solve_Q(Y, q_m, x, Y_h, u_h, V, I_MEC, T, m, P):
    Y_CH4, Y_H2CH4, Y_H2 = Y
    x_m, x_h = x
    R = 8.314 #0.0521 # L * atm / mol * K
    F = 96485 #* (1 / 3600) * (1 / 24) # (A * d / mol)

    Q_CH41 = Y_CH4 * q_m * x_m * V
    Q_CH42 = Y_H2CH4 * Y_h * u_h * x_h * V
    Q_H2 = Y_H2 * (I_MEC * R * T / (m * F * P)) - Y_h * u_h * x_h * V

    return [Q_CH41, Q_CH42, Q_H2]

def main():

    # Parameters
    Y_CH4 = 0.28
    q_maxm = 14.12 * (1 / 24) * (1 / 3600)
    q_maxa = 13.14 * (1 / 24) * (1 / 3600)
    u_maxa = 1.97 * (1 / 24) * (1 / 3600)
    u_maxm = 0.3 * (1 / 24) * (1 / 3600)
    u_maxh = 0.5 * (1 / 24) * (1 / 3600)
    K_Aa = 20 * (1 / 1000) * (1 / 1000) * (1000 / 1)
    K_Am = 80 * (1 / 1000) * (1 / 1000) * (1000 / 1)
    m = 2
    gamma = 663400 * (1 / 1000) * (1 / 1000)
    M_Total = 1000 # mg/mg Victor Alcaraz–Gonzalez
    K_M = 0.01 * (1 / 1000) * (1 / 1000) * (1000 / 1)
    K_da = 0.04 * (1 / 24) * (1 / 3600)
    K_dm = 0.006 * (1 / 24) * (1 / 3600)
    K_dh = 0.01 * (1 / 24) * (1 / 3600)
    K_h = 0.0001 * (1 / 1000) * (1 / 1000) * (1000 / 1)
    H_2 = 1 * (1 / 1000) * (1 / 1000) * (1000 / 1)
    Y_H2CH4 = 0.25
    Y_H2 = 0.9
    X_max1 = 512.5 * (1 / 1000) * (1 / 1000) * (1000 / 1)
    X_max2 =  1215 * (1 / 1000) * (1 / 1000) * (1000 / 1) # mg/L Victor Alcaraz–Gonzalez
    Y_h = 0.05 # mL/mg Victor Alcaraz–Gonzalez
    A_surA = 0.01 # m^2 Victor Alcaraz–Gonzalez
    R_MIN = 20
    R_MAX = 2000
    K_R = 0.024 * (1 / 1000) * (1000 / 1) * (1000 / 1)

    Y_M = 34.85
    E_CEF = -0.35
    beta = 0.5
    I_0 = 0.005
    T = 298.15 # K Muhammad Zia Ur Rahman
    V = 250 * (1 / 1000) * (1 / 1000) # mL Muhammad Zia Ur Rahman
    F_in = 400 * (1 / 1000) * (1 / 1000) * (1 / 24) * (1 / 3600) # mL/d René Alejandro Flores-Estrella
    A_0 = 500 * (1 / 1000) * (1 / 1000) * (1000 / 1) # mg/L
    P = 1 * (101325) # atm Muhammad Zia Ur Rahman

    # Initial conditions
    A0 = 2000 * (1 / 1000) * (1 / 1000) * (1000 / 1) # mg/L Muhammad Zia Ur Rahman
    x_a0 = 1 * (1 / 1000) * (1 / 1000) * (1000 / 1) # mg/L Muhammad Zia Ur Rahman
    x_m0 = 10 * (1 / 1000) * (1 / 1000) * (1000 / 1) # mg/L Muhammad Zia Ur Rahman
    x_h0 = 50 * (1 / 1000) * (1 / 1000) * (1000 / 1) # mg/L Muhammad Zia Ur Rahman
    M_ox0 = 100 # mg/mg Victor Alcaraz–Gonzalez

    start = 0
    end = 3 * 24 * 3600
    sol = scipy.integrate.solve_ivp(solve_de, [start, end], [A0, x_a0, x_m0, x_h0, M_ox0], t_eval=np.linspace(start, end, 1000), method = "DOP853", rtol=1E-6, atol=1E-9, args=[u_maxa, u_maxm, K_Aa, K_Am, q_maxa, q_maxm, K_M, F_in, V, A_0, K_da, K_dm, K_dh, u_maxh, H_2, K_h, X_max1, X_max2, gamma, m, T, M_Total, R_MIN, R_MAX, I_0, beta, A_surA, E_CEF, K_R, Y_M])
    t = sol.t
    A, x_a, x_m, x_h, M_ox = sol.y


    Y = [Y_CH4, Y_H2CH4, Y_H2]
    u_max = [u_maxa, u_maxm]
    q_max = [q_maxa, q_maxm]
    K_A = [K_Aa, K_Am]
    q_m = solve_uq(u_max, K_A, q_max, A, M_ox, K_M)[3]
    x = [x_m, x_h]
    u_h = solve_u(u_maxh, H_2, K_h)
    M_red = solve_M(M_Total, M_ox)

    n_conc = []
    R_int = []
    I_MEC = []
    n_act = []
    n_ohm = []
    E = []
    E_applied = []
    for M_red_1, x_a_1 in zip(M_red, x_a):
        res = solve_En(T, M_red_1, m, R_MIN, R_MAX, I_0, beta, M_Total, A_surA, E_CEF, K_R, x_a_1)
        n_conc.append(res[0])
        R_int.append(res[1])
        I_MEC.append(res[2])
        n_act.append(res[3])
        n_ohm.append(res[4])
        E.append(res[5])
        E_applied.append(res[6])
    n_conc = np.array(n_conc)
    R_int = np.array(R_int)
    I_MEC = np.array(I_MEC)
    n_act = np.array(n_act)
    n_ohm = np.array(n_ohm)
    E = np.array(E)
    E_applied = np.array(E_applied)

    Q_CH41, Q_CH42, Q_H2 = solve_Q(Y, q_m, x, Y_h, u_h, V, I_MEC, T, m, P)

    plt.plot(t, E)
    plt.show()


main()