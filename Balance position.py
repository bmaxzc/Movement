import scipy.optimize as opt
from math import sin
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

u1 = 4   
u2 = 1 

x = np.linspace(-6, 6, 301)

X11 = -(0.3*4)/(0.6*x)
X12 = -(0.3*x)/(0.7-0.1)

fig = plt.figure(figsize=(10, 6)) 

plt.plot(x,X11, linewidth=1, color='blue') 
plt.plot(x, X12, linewidth=1, color='red')
plt.axis([-7, 7, 7, -7])

def func(x): #left side of the equation
  x1, x2 = x
  f1 = 0.6*x1*x2 + 0.3*u1
  f2 = 0.3*x2 + 0.7*x1 - 0.1*x1*u2
  return [f1, f2] 

x0 = [-3, 5] # Approximate balance points
#x0 = [1.8, -1.1]

ans = opt.fsolve(func, x0) 

print("ans =", ans)
print("pogr =", func(ans)) # error

T = 3 
time = np.linspace(0, T, 31) # segmentation of time
f_lam = lambda x, t: func(x)
res = odeint(f_lam, ans, time) # ODE solution
print("res = ", res)