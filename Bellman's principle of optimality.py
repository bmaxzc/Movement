"""
Task: set the boundary value problem 
of optimal control for a material point
 moving rectilinearly under 
 the action of two forces.
"""
from scipy.interpolate import interp1d
from scipy.integrate import odeint
from scipy.integrate import simps
import numpy as np
import matplotlib.pyplot as plt


t_k = 10
time = np.linspace(t_k, 0, num=50, endpoint=True)
time_new = np.linspace(0, t_k, num=100, endpoint=True)
lam = 1
d = 1
c = 1
m = 1
x_10 = 1
x_20 = 3
s_0 = [0, 0, 0, 0, 0, lam]

def ode_s(s_i0, t):
    s_10, s_20, s_30, s_40, s_50, s_60 = s_i0
    ds_1 = -s_30 ** 2 / 4 + d * s_30 / m 
    ds_2 = -s_30 * s_40 / 2 + d * s_40 / m - c * s_30 / m
    ds_3 = -s_30 * s_60 + 2 * d * s_60 /m + s_20
    ds_4 = -s_40 * s_60 - 2 * c * s_60 / m + 2 * s_50
    ds_5 = -s_40 ** 2 / 4 - c * s_40 / m
    ds_6 = s_40 - s_60 ** 2
    return [ds_1, ds_2, ds_3, ds_4, ds_5, ds_6]


s_i = odeint(ode_s, s_0, time)
s_4 = interp1d(time, s_i[:, 3], axis=0, fill_value="extrapolate")
s_6 = interp1d(time, s_i[:, 5], axis=0, fill_value="extrapolate")

def u_opt(x, t):
    x_1, x_2 = x
    return - (s_4(t) * x_1 + 2 * s_6(t) * x_2) / 2


def ode_f(x, t):
    x_1, x_2 = x
    f_1 = x_2
    f_2 = (d - c * x_1)/m + u_opt(x, t)
    return [f_1, f_2]


def cauchy_problem_f(x_0, t):
    res = odeint(ode_f, x_0, t)
    x_1 = np.array(res[:, 0])
    x_2 = np.array(res[:, 1])
    u = u_opt([x_1, x_2], t)
    sim = u ** 2

    print("J = {}".format(simps(sim, t) + lam * x_2[-1] ** 2))
    print("x_1({0}) = {1}, x_2({2}) = {3}".format(t_k, x_1[-1], t_k, x_2[-1]))

    plt.plot(t, x_1, 'r', linewidth=2, label='coordinates')
    plt.legend(loc='best')
    plt.tick_params(labelsize = 12)
    plt.xlabel('$t$', fontsize=20)
    plt.ylabel('$x$ ', fontsize=20)
    plt.legend()
    plt.grid()
    plt.show()

    plt.plot(t, x_2, 'g', linewidth=2, label='speed')
    plt.legend(loc='best')
    plt.tick_params(labelsize = 12)
    plt.xlabel('$t$', fontsize=20)
    plt.ylabel('$x_2$ ', fontsize=20)
    plt.legend()
    plt.grid()
    plt.show()

    plt.plot(x_1, x_2, 'b', linewidth=2, label='Phase_portrait')
    plt.legend(loc='best')
    plt.tick_params(labelsize = 12)
    plt.xlabel('$t$', fontsize=20)
    plt.ylabel('$u$ ', fontsize=20)
    plt.legend()
    plt.grid()
    plt.show()

    plt.plot(t, u, 'r', linewidth=2, label='Opt_control')
    plt.legend(loc='best')
    plt.tick_params(labelsize = 12)
    plt.xlabel('$x_1$', fontsize=20)
    plt.ylabel('$x_2$ ', fontsize=20)
    plt.legend()
    plt.grid()
    plt.show()
cauchy_problem_f([x_10, x_20], time_new)