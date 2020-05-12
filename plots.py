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

#if not os.path.exists("covid"):
#    os.makedirs("covid")

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

burnin = 1000000
T = 2000000


chain = np.load('covid/cadena.npy')

Energy = chain[:,-1]

#############################################
### Computing the MAP estimate
energy_MAP = min(Energy)
loc_MAP = np.where(Energy==energy_MAP)[0]
MAP = chain[loc_MAP[-1]]
MAP = MAP[:-1]

sol_MAP = soln2(MAP)
infected_day_MAP = sol_MAP[:,3]
Total_cases_MAP = np.ones(len(infected_day_MAP))
Total_cases_MAP[0]=data[0]
for i in np.arange(1,N,1):
    Total_cases_MAP[i]=Total_cases_MAP[i-1]+infected_day_MAP[i]


### Computing the posterior mean
Post_mean = np.ones(5)
Post_mean[0] = np.mean(chain[burnin:,0])
Post_mean[1] = np.mean(chain[burnin:,1])
Post_mean[2] = np.mean(chain[burnin:,2])
Post_mean[3] = np.mean(chain[burnin:,3])
Post_mean[4] = np.mean(chain[burnin:,4])

sol_CM = soln2(Post_mean)
infected_day = sol_CM[:,3]
Total_cases_CM = np.ones(len(infected_day))
Total_cases_CM[0]=data[0]
for i in np.arange(1,N,1):
    Total_cases_CM[i]=Total_cases_CM[i-1]+infected_day[i]

#############################################
#
pl.figure(figsize=(3,2.5))
pl.hist(chain[burnin:,0],bins=30)
pl.xlabel(r"$\beta_A$", fontsize=14)
pl.ylabel('Frequency',fontsize=14)
pl.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
pl.tight_layout()
pl.savefig('covid/beta_A-sea.pdf')
pl.show()
#
pl.figure(figsize=(3,2.5))
pl.hist(chain[burnin:,1],bins=30)
pl.xlabel(r"$\beta_I$", fontsize=14)
pl.ylabel('Frequency',fontsize=14)
pl.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
pl.tight_layout()
pl.savefig('covid/beta_I-sea.pdf')
pl.show()

pl.figure(figsize=(3,2.5))
pl.hist(chain[burnin:,2],bins=30)
pl.xlabel(r"$\beta_V$", fontsize=14)
pl.ylabel('Frequency',fontsize=14)
pl.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
pl.tight_layout()
pl.savefig('covid/beta_V-sea.pdf')
pl.show()

pl.figure(figsize=(3,2.5))
pl.hist(chain[burnin:,3],bins=30)
pl.xlabel(r"$c_1$", fontsize=14)
pl.ylabel('Frequency',fontsize=14)
pl.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
pl.tight_layout()
pl.savefig('covid/c1-sea.pdf')
pl.show()

pl.figure(figsize=(3,2.5))
pl.hist(chain[burnin:,4],bins=30)
pl.xlabel(r"$c_2$", fontsize=14)
pl.ylabel('Frequency',fontsize=14)
pl.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
pl.tight_layout()
pl.savefig('covid/c2-sea.pdf')
pl.show()


###########################################################################

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime as dt


dates = ['11/03','12/03','13/03','14/03','15/03','16/03','17/03','18/03','19/03','20/03', \
        '21/03','22/03','23/03','24/03','25/03']

x = [dt.datetime.strptime(d,'%d/%m').date() for d in dates]
y = data 


pl.figure(figsize=(5,3))
pl.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))
pl.gca().xaxis.set_major_locator(mdates.DayLocator(interval=3))
pl.plot(x, y, 'go', markersize=8, label='data')
pl.gcf().autofmt_xdate()
pl.plot(x, Total_cases_MAP, 'r-', linewidth=3,label='MAP')
pl.plot(x, Total_cases_CM, 'b--', linewidth=3, label='CM')
pl.ylabel('Infected individuals', fontsize=14)
pl.legend(loc=0)
pl.tight_layout()
#pl.savefig('data_and_fitted_dates-sea.pdf')
pl.show()


###########################################################################
beta_A = MAP[0]
beta_I = MAP[1]
beta_V = MAP[2]
c1 = MAP[3]
c2 = MAP[4]
Term1 = (beta_A/gamma_A + c1*beta_V/(mu_V*gamma_A))
Term2 = (beta_I/(gamma_I+mu) + c2*beta_V/(mu_V*(mu+gamma_I)))
R0_MAP = (Term1*(1-p)+Term2*p)*S_0
#print('The value of R0 at the MAP estimate is ',R0_MAP)
#
############################################################################
beta_A = Post_mean[0]
beta_I = Post_mean[1]
beta_V = Post_mean[2]
c1 = Post_mean[3]
c2 = Post_mean[4]
Term1 = (beta_A/gamma_A + c1*beta_V/(mu_V*gamma_A))
Term2 = (beta_I/(gamma_I+mu) + c2*beta_V/(mu_V*(mu+gamma_I)))
R0_mean = (Term1*(1-p)+Term2*p)*S_0
#print('The value of R0 at the posterior mean is ',R0_mean)

###############################################################################
beta_A = chain[burnin:,0]
beta_I = chain[burnin:,1]
beta_V = chain[burnin:,2]
c1 = chain[burnin:,3]
c2 = chain[burnin:,4]
Term1 = (beta_A/gamma_A + c1*beta_V/(mu_V*gamma_A))
Term2 = (beta_I/(gamma_I+mu) + c2*beta_V/(mu_V*(mu+gamma_I)))
R0 = (Term1*(1-p)+Term2*p)*S_0

pl.figure(figsize=(5,3))
pl.hist(R0,bins=70, normed=True)
pl.xlim((0.0,6.0))
pl.axvline(x = R0_MAP,color='r',label='MAP')
pl.axvline(x = R0_mean,color='c', label='Posterior mean')
pl.xlabel('R0', fontsize=14)
pl.ylabel('Frequency', fontsize=14)
pl.legend(loc=0)
pl.tight_layout()
pl.savefig('R0-sea.pdf')
pl.show()

###############################################################################
Time = np.arange(0,300,1)

def soln(q):
    return integrate.odeint(rhs,x0,Time,args=(q,))
    
sol_MAP = soln(MAP)
infected_day_MAP = sol_MAP[:,3]

sol_CM = soln(Post_mean)
infected_day_CM = sol_CM[:,3]

pl.figure(figsize=(5,3))
#pl.plot(Time2, data, 'ro', label='data')
pl.plot(Time, infected_day_MAP, 'r', label='MAP')
pl.fill_between(Time, infected_day_MAP,color='r', alpha=0.30)
pl.plot(Time, infected_day_CM, 'b', label='CM')
pl.fill_between(Time, infected_day_CM,color='b', alpha=0.30)
pl.xlabel('Time in days', fontsize=14)
pl.ylabel('Infected individuals', fontsize=14)
pl.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
pl.legend(loc=0)
pl.tight_layout()
#pl.savefig('infected_proj-sea.pdf')
pl.show()

###############################################################################

data_new = np.array([11,15,26,41,53,82,93,118,164,203,251,316,367,405,475,585,717,848,993,1094,1215])
N_new = len(data_new)
Time_new = np.arange(0,N_new,1)

Time_extra = np.arange(0,N_new,1)

def soln3(q):
    return integrate.odeint(rhs,x0,Time_extra,args=(q,))


### Computing the MAP estimate
sol_MAP = soln3(MAP)
infected_day_MAP = sol_MAP[:,3]
Total_cases_MAP = np.ones(len(infected_day_MAP))
Total_cases_MAP[0]=data[0]
for i in np.arange(1,N_new,1):
    Total_cases_MAP[i]=Total_cases_MAP[i-1]+infected_day_MAP[i]


### Computing the posterior mean

sol_CM = soln3(Post_mean)
infected_day = sol_CM[:,3]
Total_cases_CM = np.ones(len(infected_day))
Total_cases_CM[0]=data[0]
for i in np.arange(1,N_new,1):
    Total_cases_CM[i]=Total_cases_CM[i-1]+infected_day[i]


#import matplotlib.pyplot as plt
#import matplotlib.dates as mdates
#import datetime as dt


dates = ['11/03','12/03','13/03','14/03','15/03','16/03','17/03','18/03','19/03','20/03', \
        '21/03','22/03','23/03','24/03','25/03','26/03','27/03','28/03','29/03','30/03','31/03']

data_new = np.array([11,15,26,41,53,82,93,118,164,203,251,316,367,405,475,585,717,848,993,1094,1215])

x = [dt.datetime.strptime(d,'%d/%m').date() for d in dates]
y = data_new # many thanks to Kyss Tao for setting me straight here


pl.figure(figsize=(5,3))
pl.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))
pl.gca().xaxis.set_major_locator(mdates.DayLocator(interval=3))
#A = np.zeros((25000,21))
#for k in range(25000):
#    if k%1000==0:
#        print(k)
#    qq = chain[-1-k][:-1]
#    my_soln = soln3(qq)
#    infected_dayqq = my_soln[:,3]
#    Total_cases_qq = np.ones(len(infected_dayqq))
#    Total_cases_qq[0]=data[0]
#    for i in np.arange(1,N_new,1):
#        Total_cases_qq[i]=Total_cases_qq[i-1]+infected_dayqq[i]
#    A[k][:]=Total_cases_qq
##    pl.plot(x,Total_cases_qq,'k',color='0.75')
#mu1 = A.mean(axis=0)
#sigma1 = A.std(axis=0)
pl.plot(x, Total_cases_MAP, 'r', linewidth=2, label='MAP')
pl.plot(x, Total_cases_CM, 'b--', linewidth=2, label='CM')
pl.plot(x, mu1+3*sigma1, '--', color='gray')
pl.plot(x, mu1-3*sigma1, '--', color='gray')
pl.fill_between(x, mu1+3*sigma1, mu1-3*sigma1, facecolor='gray', alpha=0.3)
pl.plot(x[0:15], data, 'ro', markersize=7, label='data')
pl.plot(x[15:], y[15:], 'go', markersize=7,label='data new')
pl.ylabel('Infected individuals', fontsize=14)
pl.gcf().autofmt_xdate()
pl.legend(loc=0)
pl.tight_layout()
#pl.savefig('UQ_Total_Infected_proj_dates-sea.pdf')
pl.show()
#
