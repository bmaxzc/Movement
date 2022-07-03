"""
It is necessary to linearize the equations of motion of the controlled system
 near the equilibrium position and establish whether the zero solution of the
 linearized system is asymptotically stable.
"""
import scipy.optimize as opt
from math import sin, cos
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def func(x): 
  x1, x2 = x
  f1 = 0.6*x0[1]*x1 + 0.6*x0[0]*x2
  f2 = 0.6*x1 + 0.3*x2
  return [f1, f2]
x0 = np.array([1, -2]) 
A = np.array([[0.6*x0[1], 0.6*x0[0]],
[0.6, 0.3]])
lam = np.linalg.eig(A) 
print(lam) 
T = 3
time = np.linspace(0, T, 301)
f_lam = lambda x, t: func(x)
res = odeint(f_lam, np.array([10, 10]), time)
fig = plt.figure(figsize=(10, 6)) 
plt.plot(time, res[:,0], linewidth=1) 
plt.plot(time, res[:,1], linewidth=1, color='green')
plt.axis([0, 4, -50, 80])