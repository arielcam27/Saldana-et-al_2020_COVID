# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 14:58:24 2020

@author: hugo.flores
"""

import numpy as np
import pylab as pl
import pytwalk
#import sys
import os
from tempfile import TemporaryFile
from scipy import integrate

if not os.path.exists("covid"):
    os.makedirs("covid")

gamma_A = 0.13978
gamma_I = 0.33029
gamma_D = 0.11624
sigma = 1.0/6.4
p = 0.868343
mu = 1.7826e-5
mu_V = 1.0
d = 1.0/9.0

### parameter to estimate
#beta_A = 1.0e-9
#beta_I = 0.1e-7    
#beta_V = 0.1e-7
#c1 = 1.0e-3
#c2 = 1.0e-5
#
#q = np.array([beta_A,beta_I,beta_V,c1,c2])

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

S_0 = 1.28e8
E_0 = 4.0
A_0 = 1.0
I_0 = 4.0
R_0 = 0.0
V_0 = 10.0

x0 = np.array([S_0,E_0,A_0,I_0,R_0,V_0])


### from march 11 to march 25
data = np.array([11,15,26,41,53,82,93,118,164,203,251,316,367,405,475])
N = len(data)

Time2 = np.arange(0,N,1)

def soln2(q):
    return integrate.odeint(rhs,x0,Time2,args=(q,))

var = 5.0**2

k0 = 1.0e-8
theta0 = 1.0 
k1 = 1.0e-3
theta1 = 1.0 

def energy(q): # -log of the posterior
    my_soln2 = soln2(q)
    infected_day = my_soln2[:,3]
    Total_cases = np.ones(len(infected_day))
    Total_cases[0]=data[0]
    for i in np.arange(1,N,1):
        Total_cases[i]=Total_cases[i-1]+infected_day[i]

    log_likelihood = -0.5*(np.linalg.norm(data - Total_cases))**2/var # Gaussian
    a0 = (k0-1)*np.log(q[0])- (q[0]/theta0)   ## gamma distribution for gamma_A
#    print(a0)
    a1 = (k1-1)*np.log(q[3])- (q[3]/theta1)   ## gamma distribution for c1
#    a2 = (k2-1)*np.log(p[2])- (p[2]/theta2)
#    a3 = (k3-1)*np.log(p[3])- (p[3]/theta3)
    log_prior = a0+a1#+a0+a2
    return -log_likelihood - log_prior

def support(q):
    rt = True
    rt &= 0.0 < q[0] <0.1
    rt &= 0.0 < q[1] <0.1
    rt &= 0.0 < q[2] <0.1
    rt &= 0.0 < q[3] <1.0
    rt &= 0.0 < q[4] <1.0
    return rt

def init():
    q = np.zeros(5)
    q[0] = np.random.uniform(low=0.0, high=1.0e-5)
#    q[0] = np.random.gamma(k0, theta0)
    q[1] = np.random.uniform(low=0.0, high=0.1)
    q[2] = np.random.uniform(low=0.0, high=0.1)
#    q[3] = np.random.gamma(k1, theta1)
    q[3] = np.random.uniform(low=0.0, high=1.0e-3)
    q[4] = np.random.uniform(low=0.0, high=1.0)
    return q

burnin = 1000000
T = 2000000

covid = pytwalk.pytwalk(n=5,U=energy,Supp=support)
y0=init()
yp0=init()
covid.Run(T,y0,yp0)

    
cadena=TemporaryFile()
np.save('covid/cadena',covid.Output)


chain = covid.Output

energy = chain[:,-1]
#############################################
### Computing the MAP estimate
energy_MAP = min(energy)
loc_MAP = np.where(energy==energy_MAP)[0]
MAP = chain[loc_MAP[-1]]
MAP = MAP[:-1]

### Computing the posterior mean
Post_mean = np.ones(5)
Post_mean[0] = np.mean(chain[burnin:,0])
Post_mean[1] = np.mean(chain[burnin:,1])
Post_mean[2] = np.mean(chain[burnin:,2])
Post_mean[3] = np.mean(chain[burnin:,3])
Post_mean[4] = np.mean(chain[burnin:,4])
#############################################

pl.figure()
pl.plot(range(T+1),energy)
##pl.title('Energy Patient '+str(p_number))
pl.savefig('covid/energy.png')
pl.show()
#
pl.figure()
pl.plot(range(T+1-burnin),energy[burnin:])
##pl.title('Energy Patient '+str(p_number))
pl.savefig('covid/energy_burn.png')
pl.show()

pl.figure()
pl.hist(chain[burnin:,0],bins=30)
#pl.title('Theta0 Patient '+str(p_number))
pl.savefig('covid/beta_A.png')
pl.show()

pl.figure()
pl.hist(chain[burnin:,1],bins=30)
#pl.title('Theta0 Patient '+str(p_number))
pl.savefig('covid/beta_I.png')
pl.show()

pl.figure()
pl.hist(chain[burnin:,2],bins=30)
#pl.title('Theta0 Patient '+str(p_number))
pl.savefig('covid/beta_V.png')
pl.show()

pl.figure()
pl.hist(chain[burnin:,3],bins=30)
#pl.title('Theta0 Patient '+str(p_number))
pl.savefig('covid/c1.png')
pl.show()

pl.figure()
pl.hist(chain[burnin:,4],bins=30)
#pl.title('Theta0 Patient '+str(p_number))
pl.savefig('covid/c2.png')
pl.show()


###############################################################################
beta_A = chain[burnin:,0]
beta_I = chain[burnin:,1]
beta_V = chain[burnin:,2]
c1 = chain[burnin:,3]
c2 = chain[burnin:,4]
Term1 = (beta_A/gamma_A + c1*beta_V/(mu_V*gamma_A))
Term2 = (beta_I/(gamma_I+mu) + c2*beta_V/(mu_V*(mu+gamma_I)))
R0 = (Term1*(1-p)+Term2*p)*S_0

pl.figure()
pl.hist(R0,bins=70)
pl.xlim((0.0,6.0))
#pl.title('Theta0 Patient '+str(p_number))
pl.savefig('covid/R0.png')
pl.show()

###########################################################################
beta_A = MAP[0]
beta_I = MAP[1]
beta_V = MAP[2]
c1 = MAP[3]
c2 = MAP[4]
Term1 = (beta_A/gamma_A + c1*beta_V/(mu_V*gamma_A))
Term2 = (beta_I/(gamma_I+mu) + c2*beta_V/(mu_V*(mu+gamma_I)))
R0 = (Term1*(1-p)+Term2*p)*S_0
print('The value of R0 at the MAP estimate is ',R0)

###########################################################################
beta_A = Post_mean[0]
beta_I = Post_mean[1]
beta_V = Post_mean[2]
c1 = Post_mean[3]
c2 = Post_mean[4]
Term1 = (beta_A/gamma_A + c1*beta_V/(mu_V*gamma_A))
Term2 = (beta_I/(gamma_I+mu) + c2*beta_V/(mu_V*(mu+gamma_I)))
R0 = (Term1*(1-p)+Term2*p)*S_0
print('The value of R0 at the posterior mean is ',R0)
