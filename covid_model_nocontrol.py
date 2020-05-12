# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 14:58:24 2020

@author: hugo.flores
"""

import numpy as np
import pylab as pl
from scipy import integrate

time = np.linspace(0.0,10.0,100)   ### continuous time
time2 = np.linspace(0.0,10.0,11)   ### discrete time by day

gamma_A = 0.13978
gamma_I = 0.33029
gamma_D = 0.11624
sigma = 1.0/6.4
p = 0.868343
mu = 1.7826e-5
mu_V = 1.0
d = 1.0/9.0
alpha = 0.0
theta = 0.0
### parameter to estimate
beta_A = 0.1e-3
beta_I = 0.1e-3    
beta_V = 0.1e-3
c1 = 5.0
c2 = 1.2

q = np.array([beta_A,beta_I,beta_V,c1,c2])

### Set the dynamical system
def rhs(x,t,q):
    beta_A= q[0]
    beta_I= q[1]
    beta_V= q[2]
    c1 = q[3]
    c2 = q[4]

    fx = np.zeros(6)
    fx[0] = -(beta_A*x[2]+beta_I*x[3]+beta_V*x[5])*x[0]   ###S dot
    fx[1] = (beta_A*x[2]+beta_I*x[3]+beta_V*x[5])*x[0]-sigma*x[1] ###E dot
    fx[2] = (1.0-p)*sigma*x[1]-gamma_A*x[2]  ### A dot
    fx[3] = p*sigma*x[1]-gamma_I*x[3] -mu*x[3] ### I dot
    fx[4] = gamma_A*x[2]+gamma_I*x[3]   ### R dot
    fx[5] = c1*x[2]+c2*x[3]-mu_V*x[5]  ### V dot
    return fx

### Set initial conditions 

S_0 = 1.0e6
E_0 = 5.0
A_0 = 1.0
I_0 = 4.0
R_0 = 0.0
V_0 = 1.0

x0 = np.array([S_0,E_0,A_0,I_0,R_0,V_0])

def soln(q):
    return integrate.odeint(rhs,x0,time,args=(q,))

def soln2(q):
    return integrate.odeint(rhs,x0,time2,args=(q,))


my_soln = soln(q)
my_soln2 = soln2(q)

infected = my_soln[:,3]
infected_day = my_soln2[:,3]

pl.figure()
pl.plot(time,infected)
pl.plot(time2,infected_day,'ro')
pl.xlabel('Time in days')
pl.ylabel('Infected')
pl.show()


Total_cases = np.ones(len(infected_day))
Total_cases[0]=infected_day[0]
for i in np.arange(1,11,1):
    Total_cases[i]=Total_cases[i-1]+infected_day[i]

### from march 11 to march 24
data = np.array([11,15,26,41,53,82,93,118,164,203,251,316,367,405])
data_infected_day = ([4,11,15,12,29,11,25,46,39,48,65,51,38])
Time = np.arange(0,len(data_infected_day),1)
Time2 = np.arange(0,len(data_infected_day)+1,1)

#pl.figure()
#pl.plot(time,infected)
#pl.plot(Time,data_infected_day,'ro')
#pl.xlabel('Time in days')
#pl.ylabel('Infected by day')
#pl.show()
#
#pl.figure()
#pl.plot(time2,Total_cases)
#pl.plot(Time2,data,'ro')
#pl.xlabel('Time in days')
#pl.ylabel('Total Infected')
#pl.show()


###############################################################################

import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def Total_Infected (time,beta_A,beta_I,beta_V,c1,c2,E_0,A_0,V_0):
    N = len(time)
    q = np.ones(8)
    q[0] = beta_A
    q[1] = beta_I 
    q[2] = beta_V 
    q[3] = c1
    q[4] = c2
    q[5] = E_0
    q[6] = A_0
    q[7] = V_0
    I_0 = data[0]
    x0 = np.array([S_0,E_0,A_0,I_0,R_0,V_0])
    soln = integrate.odeint(rhs,x0,time,args=(q,))
    infected_day = soln[:,3]
    Total_cases = np.ones(len(infected_day))
    Total_cases[0]=infected_day[0]
    for i in np.arange(1,N,1):
        Total_cases[i]=Total_cases[i-1]+infected_day[i]
    return Total_cases

ydata = data

#Constrain the optimization to the region of 0 <= a <= 3, 0 <= b <= 1 and 0 <= c <= 0.5:

popt, pcov = curve_fit(Total_Infected, Time2, ydata, bounds=(0, [0.1,0.1,0.1,5.0,5.0,10.0,10.0,10.0]),\
            p0=np.array([0.1e-3,0.1e-3,0.1e-3,5.0,1.2,5.0,1.0,1.0]))

perr = np.sqrt(np.diag(pcov))

plt.plot(Time2, ydata, 'ro', label='data')
plt.plot(Time2, Total_Infected(Time2, *popt), 'g--', label='fit curve')
plt.xlabel('Time in days')
plt.ylabel('Total Infected')
plt.legend(loc=0)
plt.savefig('Parameter_inference_Total_Infected')
plt.show()

##################################################################
### Compute R0 from estimates values
beta_A, beta_I, beta_V, c1, c2, E_0, A_0, V_0 = popt

Term1 = (beta_A/gamma_A + c1*beta_V/(mu_V*gamma_A))
Term2 = (beta_I/(gamma_I+mu) + c2*beta_V/(mu_V*(mu+gamma_I)))
R0 = (Term1*(1-p)+Term2*p)*S_0
print('The basic reproductive number is ',R0)
