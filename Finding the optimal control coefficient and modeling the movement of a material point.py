"""
It is necessary to solve the problem numerically
 for α1=0, α2=1, tk=const. Determine how the trajectory
 of a material point will change if at the initial moment
 it moved with a speed of x20.
"""
import scipy.optimize as opt
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

alpha1=0
alpha2=1
x10=1
x20=1
tk=3.4
psi0=np.array([2,15,10])

def u_opt(x):
  x1,x2,psi1,psi2=x
  uopt=psi2/(2*alpha2)
  if abs(uopt)>1:
    uopt=np.sign(uopt)
  return uopt

def faz_sopr(x,t):
  x1,x2,psi1,psi2=x
  f1=x2
  f2=-x2+u_opt(x)
  f3=0
  f4=psi2-psi1
  return [f1,f2,f3,f4]

def H(x):
  x1,x2,psi1,psi2=x
  H=-alpha1-alpha2*u_opt(x)**2+psi1*x2-psi2*x2+psi2*u_opt(x)
  return H

def Nev(psi0):
  tk,psi10,psi20=psi0
  ans=odeint(faz_sopr,[x10,x20,psi10,psi20],np.linspace(0,tk,301))
  pogr1=ans[-1][0]-0
  pogr2=ans[-1][1]-0
  h=H([x10,x20,psi10,psi20])
  return [h,pogr1,pogr2]

def kosh(tk, res):
  t=np.linspace(0, tk, 201)
  ans=odeint(faz_sopr, res, t)
  x1=np.array(ans[:, 0])
  x2=np.array(ans[:, 1])
  u=np.array([u_opt(x) for x in ans])
  return [x1, x2, u]

psio=opt.fsolve(Nev, np.array([tk, -1, -1]))
print(psio)
nevt=Nev(psio)
print(nevt)
gg=kosh(psio[0], [x10, x20, psio[1], psio[2]])
time=np.linspace(0,psio[0],201)

plt.plot(time, gg[0], '-r', linewidth=2, label='$f_1$',color='red')
plt.legend(loc='best')
plt.tick_params(labelsize = 12)
plt.xlabel('$t$', fontsize=20)
plt.ylabel('$x_1$ ', fontsize=20)
plt.legend()
plt.grid()
plt.show()

plt.plot(time, gg[1], '-r', linewidth=2, label='$f_1$',color='green')
plt.legend(loc='best')
plt.tick_params(labelsize = 12)
plt.xlabel('$t$', fontsize=20)
plt.ylabel('$x_2$ ', fontsize=20)
plt.legend()
plt.grid()
plt.show()

plt.plot(time, gg[2], '-r', linewidth=2, label='$f_1$',color='blue')
plt.legend(loc='best')
plt.tick_params(labelsize = 12)
plt.xlabel('$t$', fontsize=20)
plt.ylabel('$u$ ', fontsize=20)
plt.legend()
plt.grid()
plt.show()

plt.plot(gg[0], gg[1], '-r', linewidth=2, label='$f_1$',color='red')
plt.legend(loc='best')
plt.tick_params(labelsize = 12)
plt.xlabel('$x_1$', fontsize=20)
plt.ylabel('$x_2$ ', fontsize=20)
plt.legend()
plt.grid()
plt.show()