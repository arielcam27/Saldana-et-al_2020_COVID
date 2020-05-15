"""
Manuscript:
Modeling the transmission dynamics and the impact of the control
interventions for the COVID-19 epidemic outbreak

Authors:
Fernando Saldana, Hugo Flores-Arguedas, Jose Ariel Camacho-Gutiérrez, and Ignacio
Barradas

Date:
May, 2020

To perform the Bayesian Inference, we run an MCMC using t-walk:

Christen, J. A., & Fox, C. (2010). 
A general purpose sampling algorithm for continuous distributions (the t-walk).
Bayesian Analysis, 5(2), 263-281.


To run this script, download the pytwalk script from 

https://www.cimat.mx/~jac/twalk/

"""

import numpy as np
import pytwalk
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
#beta_A = 1.0e-8
#beta_I = 0.1e-8    
#beta_V = 0.1e-8
#c1 = 1.0e-3
#c2 = 1.0e-2
#

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

### numerical solution of the ODE system
def soln2(q):
    return integrate.odeint(rhs,x0,Time2,args=(q,))

### likelihood variance
var = 5.0**2

### prior parameters

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
    a1 = (k1-1)*np.log(q[3])- (q[3]/theta1)   ## gamma distribution for c1

    log_prior = a0+a1

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
    q[0] = np.random.uniform(low=0.0, high=0.1)
    q[1] = np.random.uniform(low=0.0, high=0.1)
    q[2] = np.random.uniform(low=0.0, high=0.1)
    q[3] = np.random.uniform(low=0.0, high=1.0)
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


###########################################################################
### Computing the R0 value for the MAP estimate
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
### Computing the R0 value for the posterior mean
beta_A = Post_mean[0]
beta_I = Post_mean[1]
beta_V = Post_mean[2]
c1 = Post_mean[3]
c2 = Post_mean[4]
Term1 = (beta_A/gamma_A + c1*beta_V/(mu_V*gamma_A))
Term2 = (beta_I/(gamma_I+mu) + c2*beta_V/(mu_V*(mu+gamma_I)))
R0 = (Term1*(1-p)+Term2*p)*S_0
print('The value of R0 at the posterior mean is ',R0)
