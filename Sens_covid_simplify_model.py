# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 14:58:24 2020

@author: hugo.flores
"""

import numpy as np
from scipy import integrate
from SALib.sample import saltelli
from SALib.analyze import sobol
from tempfile import TemporaryFile
import matplotlib.pyplot as plt

N = 11  ### number of days
time2 = np.linspace(0.0,N-1,N)   ### discrete time by day

gamma_A = 0.13978
gamma_I = 0.33029
gamma_D = 0.11624
sigma = 1.0/6.4
p = 0.868343
mu = 1.7826e-5
mu_V = 1.0
d = 1.0/9.0
#alpha = 0.0
#theta = 0.0

def g(q):
    def rhs(x,t,q):
        beta_A= q[0]
        beta_I= q[1]
        beta_V= q[2]
        c1 = q[3]
        c2 = q[4]

        fx = np.zeros(6)
        fx[0] = -(beta_A*x[2]+beta_I*x[3]+beta_V*x[5])*x[0]   ###S dot
        fx[1] = (beta_A*x[2]+beta_I*x[3]+beta_V*x[5])*x[0]-sigma*x[1]  ###E dot
        fx[2] = (1.0-p)*sigma*x[1]-gamma_A*x[2]  ### A dot
        fx[3] = p*sigma*x[1]-gamma_I*x[3] -mu*x[3] ### I dot
        fx[4] = gamma_A*x[2]+gamma_I*x[3] ### R dot
        fx[5] = c1*x[2]+c2*x[3]-mu_V*x[5]  ### V dot
        return fx

    S_0 = 1000.0
#    E_0 = 0.0
    E_0 = q[5]
#    A_0 = 0.0
    A_0 = q[6]
    I_0 = 2.0
    R_0 = 0.0
#    V_0 = 1.0
    V_0 = q[7]

    x0 = np.array([S_0,E_0,A_0,I_0,R_0,V_0])

    def soln(q):
        return integrate.odeint(rhs,x0,time2,args=(q,))

    my_soln2 = soln(q)

    infected_day = my_soln2[:,3]

    Total_cases = np.ones(len(infected_day))
    Total_cases[0]=infected_day[0]
    for i in np.arange(1,N,1):
        Total_cases[i]=Total_cases[i-1]+infected_day[i]

    return Total_cases[-1]


### parameter to estimate
#beta_A = 1.0
#beta_I = 1.0
#beta_V = 1.0
#c1 = 1.0
#c2 = 1.0

def evaluate(values):
    Y = np.empty([values.shape[0]])
    for i, X in enumerate(values):
        Y[i] = g(X)
    return Y

problem = {
#'num_vars': 5,
#'names': ['beta_A', 'beta_I', 'beta_V','c1','c2'],
#'bounds': [[0, 1e-1], [0, 1e-1],[0, 1e-1],[0, 1],[0, 1]]
'num_vars': 8,
'names': ['beta_A', 'beta_I', 'beta_V','c1','c2','E0','A0','V0'],
#'bounds': [[0, 1e-1], [0, 1e-1],[0, 1e-1],[0, 1],[0, 1],[0, 10],[0, 10],[0, 10]]
'bounds': [[0, 1e-1], [0, 1e-1],[0, 1e-1],[1, 5],[1, 5],[0, 10],[0, 10],[0, 10]]
}

# Generate samples
param_values = saltelli.sample(problem, 50000, calc_second_order=True)

# Run model (example)
Y = evaluate(param_values)

# Perform analysis
Si = sobol.analyze(problem, Y, print_to_console=True)
#Si = sobol.analyze(problem, Y, print_to_console=False)
# Returns a dictionary with keys 'S1', 'S1_conf', 'ST', and 'ST_conf'
# (first and total-order indices with bootstrap confidence intervals)

print(Si['S1'])
print(Si['ST'])
print(Si['S2'])

Index=TemporaryFile()
np.save('indicesN'+str(N)+'/Indices1',Si['S1'])
np.save('indicesN'+str(N)+'/Indices1_conf',Si['S1_conf'])
np.save('indicesN'+str(N)+'/Indices2',Si['S2'])
np.save('indicesN'+str(N)+'/Indices2_conf',Si['S2_conf'])
np.save('indicesN'+str(N)+'/IndicesT',Si['ST'])
np.save('indicesN'+str(N)+'/IndicesT_conf',Si['ST_conf'])


### the histogram of the data
n, bins, patches = plt.hist(Y, 20, density=True, facecolor='green', alpha=0.75)
plt.xlabel('Total infected')
plt.grid(True)
#plt.axis([50, 200, 0, 0.3])
plt.savefig('indicesN'+str(N)+'/HistN'+str(N))
#plt.show()

