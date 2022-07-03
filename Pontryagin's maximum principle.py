"""
Task: put the problem of the optimal control for the machine
terial point moving rectilinearly under the action of two forces.
"""
import scipy.optimize as opt
import numpy as np
from scipy.integrate import odeint, simpson
import matplotlib.pyplot as plt


alpha_1 = 0
alpha_2 = 1
t_k0 = 8
a = 1
b = 1
c = 1
# Weight
m = 1
# Start coordinate and speed
x_10 = 1
x_20 = 0
# Start approximations for psi_10 and psi_20
psi_i0 = [-0.013675243344332288, -0.0306218084807793]


def plot_sol(t, x_1, x_2, u):
    plt.figure(figsize=[11, 9])

    # Plot the subplots
    # Plot 1
    plt.subplot(2, 2, 1)
    plt.plot(t, x_1, 'g')
    plt.xlabel('t', fontsize=15)
    plt.ylabel('x, м', fontsize=15)
    plt.grid()
    plt.title('a) coordinate', fontsize=15, y=1.05)

    # Plot 2
    plt.subplot(2, 2, 2)
    plt.plot(t, x_2, 'r')
    plt.xlabel('t', fontsize=15)
    plt.ylabel('v, м/с', fontsize=15)
    plt.grid()
    plt.title('b) speed', fontsize=15, y=1.05)

    # Plot 3
    plt.subplot(2, 2, 3)
    plt.plot(x_1, x_2, 'y')
    plt.xlabel('x', fontsize=15)
    plt.ylabel('v', fontsize=15)
    plt.grid()
    plt.title('c) Phase portrait', fontsize=15, pad=17)

    # Plot 4
    plt.subplot(2, 2, 4)
    plt.plot(t, u, 'b')
    plt.xlabel('t', fontsize=15)
    plt.ylabel('u', fontsize=15)
    plt.grid()
    plt.title('d) Opt. control', fontsize=15, pad=17)
    plt.tight_layout()

    plt.show()


def u_opt(x):
    u = x[3] / (2 * alpha_2)
    if abs(u) > 1:
        return np.sign(u)
    else:
        return u


def f(x):
    x_1, x_2, psi_1, psi_2 = x
    return [x_2, - c/ m * x_1 + u_opt(x)]


def func(x, t):
    x_1, x_2, psi_1, psi_2 = x
    f_1, f_2 = f(x)
    f_3 = c / m * psi_2
    f_4 = - psi_1
    return [f_1, f_2, f_3, f_4]


def hamilton(x):
    x_1, x_2, psi_1, psi_2 = x
    f_1, f_2 = f(x)
    return - alpha_1 - alpha_2 * u_opt(x) ** 2 + psi_1 * f_1 + psi_2 * f_2


def nev(psi_0):
    t_k, psi_10, psi_20 = psi_0
    ans = odeint(func, [x_10, x_20, psi_10, psi_20], np.linspace(0, t_k, 200))
    x_1, x_2, psi_1, psi_2 = ans[-1]
    h = hamilton([x_10, x_20, psi_10, psi_20])
    return [h, a * x_1 ** 2 - b * x_2, psi_1 / (2 * x_1) + psi_2 / b]


def cauchy_problem(t_k, ans):
    t = np.linspace(0, t_k, 200)
    res = odeint(func, ans, t)

    x_1 = np.array(res[:, 0])
    x_2 = np.array(res[:, 1])
    u = np.array(res[:, 3]) / (2 * alpha_2)
    sim = alpha_1 + alpha_2 * u ** 2

    print("u(0) = {0}, u({1}) = {2}".format(u[0], t_k, u[-1]))
    print("I = {}".format(simpson(sim, t)))

    plot_sol(t, x_1, x_2, u)


if __name__ == "__main__":
    psi = opt.fsolve(nev, np.array([t_k0, psi_i0[0], psi_i0[1]]))
    print("t_k = {0}, psi_10 = {1}, psi_20 = {2}".format(psi[0], psi[1], psi[2]))
    cauchy_problem(psi[0], [x_10, x_20, psi[1], psi[2]])
