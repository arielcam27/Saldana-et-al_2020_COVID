# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 14:58:24 2020

@author: hugo.flores
"""

import numpy as np
import pylab as pl
from scipy import integrate
import seaborn as sns
sns.set(style="ticks")
pl.rc('font', size=7)

gamma_A = 0.13978
gamma_I = 0.33029
gamma_D = 0.11624
sigma = 1.0/6.4
p = 0.868343
mu = 1.7826e-5
mu_V = 1.0
d = 1.0/9.0

### parameter to estimate

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

    a1 = (k1-1)*np.log(q[3])- (q[3]/theta1)   ## gamma distribution for c1

    log_prior = a0+a1
    return -log_likelihood - log_prior


burnin = 1000000
T = 2000000


#chain = np.load('covid2/cadena.npy')
chain = np.load('cadena.npy')

Energy = chain[:,-1]

#############################################

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime as dt
#
###############################################################################
N_future = 30
Future_Time = np.arange(0,N_future,1)

def soln(q):
    return integrate.odeint(rhs,x0,Future_Time,args=(q,))

###############################################################################
Total_cases_qq = np.load('Tota_cases_distribution.npy')

###############################################################################

Predictive_values = np.ones(N_future)
Predictive_var = np.ones(N_future)
for i in range(N_future):
    Predictive_values[i] = np.mean(Total_cases_qq[:,i])
    Predictive_var[i] = np.var(Total_cases_qq[:,i])

Predictive_std = np.sqrt(Predictive_var)

###############################################################################

dates = ['11/03','12/03','13/03','14/03','15/03','16/03','17/03','18/03','19/03','20/03', \
        '21/03','22/03','23/03','24/03','25/03','26/03','27/03','28/03','29/03','30/03','31/03', \
        '01/04','02/04','03/04','04/04','05/04','06/04','07/04','08/04','09/04']

data_new = np.array([11,15,26,41,53,82,93,118,164,203,\
                    251,316,367,405,475,585,717,848,993,1094,1215,\
                     1378,1510,1688,1890,2143,2439,2785,3181,3441])

def interval_low(pos):
    SS = np.sort(Total_cases_qq[:,-pos])
    suma = np.sum(SS)
    partial_sum = 0.0
    for i in range(len(SS)):
        partial_sum = partial_sum+SS[i]
        if partial_sum/suma > 1.0/100.0:
            return SS[i]
        
def interval_high(pos):
    SS = np.sort(Total_cases_qq[:,-pos])
    suma = np.sum(SS)
    partial_sum_inv = 0.0
    for i in range(len(SS)):
        partial_sum_inv = partial_sum_inv+SS[-1-i]
        if partial_sum_inv/suma > 1.0/100.0:
            return SS[-1-i]

inter_low = np.ones(N_future-1)
inter_high = np.ones(N_future-1)

for i in range(N_future-1):
    inter_low[-1-i] = interval_low(1+i) 
    inter_high[-1-i] = interval_high(1+i) 

pl.figure(figsize=(6,3))
pl.hist(Total_cases_qq[:,-10],bins=30)
pl.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
pl.axvline(x = Predictive_values[-10],color='c',label='Prediction')
pl.axvline(x = data_new[-10],color='r',label='Real ')
pl.xlabel('Predictive Cumulative Cases for 31/03',fontsize=14)
pl.ylabel('Frequency',fontsize=14)
pl.xticks(fontsize=14)
pl.yticks(fontsize=14)
pl.legend(loc=0)
pl.tight_layout()
pl.savefig('covid2/pred10-sea.pdf')
pl.show()



pl.figure(figsize=(4,3))
pl.hist(Total_cases_qq[:,-9],bins=30)
pl.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
pl.axvline(x = Predictive_values[-9],color='c',label='Prediction')
pl.axvline(x = data_new[-9],color='r',label='Real ')
pl.title('01/04')
#pl.xlabel('', fontsize=20)
pl.xlabel('Predictive Cumulative Cases',fontsize=14)
pl.ylabel('Frequency',fontsize=14)
pl.legend(loc=0)
pl.tight_layout()
pl.savefig('covid2/pred9-sea.pdf')
pl.show()
#
#
pl.figure(figsize=(4,3))
pl.hist(Total_cases_qq[:,-5],bins=30)
pl.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
pl.axvline(x = Predictive_values[-5],color='c',label='Prediction')
pl.axvline(x = data_new[-5],color='r',label='Real ')
pl.title('05/04')
pl.legend(loc=0)
pl.xlabel('Predictive Cumulative Cases',fontsize=14)
pl.ylabel('Frequency',fontsize=14)
pl.tight_layout()
pl.savefig('covid2/pred5-sea.pdf')
pl.show()
#
#
pl.figure(figsize=(4,3))
pl.hist(Total_cases_qq[:,-1],bins=30)
pl.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
pl.axvline(x = Predictive_values[-1],color='c',label='Prediction')
pl.axvline(x = data_new[-1],color='r',label='Real ')
pl.title('09/04')
pl.legend(loc=0)
pl.ylabel('Frequency',fontsize=14)
pl.xlabel('Predictive Cumulative Cases',fontsize=14)
pl.tight_layout()
pl.savefig('covid2/pred1-sea.pdf')
pl.show()

x = [dt.datetime.strptime(d,'%d/%m').date() for d in dates]
y = Predictive_values 


pl.figure(figsize=(6,3))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=3))
pl.plot(x[15:], y[15:], 'k^', label='prediction')
pl.plot(x[:15], data, 'ro', label='data used')
pl.plot(x[15:], inter_low[14:], '--', color='gray')
pl.plot(x[15:], inter_high[14:], '--', color='gray')
pl.fill_between(x[15:], inter_low[14:], inter_high[14:], facecolor='gray', alpha=0.3)
pl.plot(x[15:], data_new[15:], 'go', label='new data')
pl.ylabel('Infected Individuals', fontsize=14)
pl.legend(loc=0)
pl.gcf().autofmt_xdate()
pl.tight_layout()
pl.savefig('covid2/predictive_dates-sea.pdf')
pl.show()

'''

pl.figure(figsize=(6,3))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=3))
pl.plot(x[15:], y[15:], 'g^', label='prediction')
pl.plot(x[15:], inter_low[14:], 'k--')
pl.plot(x[15:], inter_high[14:], 'k--')
pl.plot(x[15:], data_new[15:], 'm*', label='new data')
pl.ylabel('Total Infected', fontsize=14)
pl.legend(loc=0)
pl.tight_layout()
#pl.savefig('covid2/predictive_dates_nodata-sea.pdf')
pl.show()


'''