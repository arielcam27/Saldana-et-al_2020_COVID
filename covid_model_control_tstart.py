"""
Manuscript:
Modeling the transmission dynamics and the impact of the control
interventions for the COVID-19 epidemic outbreak

Authors:
Fernando Saldana, Hugo Flores-Arguedas, Jose Ariel Camacho-Gutierrez, and Ignacio
Barradas

Date:
May, 2020
"""

import numpy as np
import pylab as pl
from scipy import integrate
import seaborn as sns
sns.set(style="ticks")
pl.rc('font', size=7)

#---------#
# initial time
t0 = 0.0
# final time
tf = 300
# sub-intervals
N = 10
# time windows
time = np.linspace(t0, tf, int((tf-t0)*N))

#---------#
def control(tstart, alpha0, theta0, d0, m0):
    # TIME
    time1 = np.linspace(t0, tstart, int((tstart-t0)*N))
    time2 = np.linspace(tstart, tf, int((tf-tstart)*N))
    
    # INITIAL CONDITIONS
    E_0  = 4.0
    A_0  = 1.0
    I_0  = 4.0
    R_0  = 0.0
    V_0  = 10.0
    S_0  = 128.0e6 - E_0 - A_0 - I_0 - R_0
    Sc_0 = 0.0
    D_0  = 0.0
    
    # FIXED PARAMETERS
    gamma_A = 0.13978
    gamma_I = 0.33029
    gamma_D = 0.11624
    sigma   = 1.0/6.4
    p       = 0.868343
    mu      = 1.7826e-5
    mu_V    = 1.0
    
    # ESTIMATED PARAMETERS
    beta_A = 1.9e-9
    beta_I = 4.52e-9
    beta_V = 4.88e-8
    c1     = 2.54e-2
    c2     = 5.31e-2
    
    q = np.array([beta_A,beta_I,beta_V,c1,c2])
    
    # MODEL
    def rhs1(x,t,q):
        # CONTROL PARAMETERS
        alpha = 0.0
        theta = 1.0
        d     = 0.0
        m     = 0.0
        
        beta_A= q[0]
        beta_I= q[1]
        beta_V= q[2]
        
        c1 = q[3]
        c2 = q[4]
        
        S  = x[0]
        Sc = x[1]
        E  = x[2]
        A  = x[3]
        I  = x[4]
        D  = x[5]
        R  = x[6]
        V  = x[7]
        
        lamb = beta_A*A + beta_I*I + beta_V*V
        
        Sdot  = -lamb*S - alpha*S
        Scdot = -lamb*theta*Sc + alpha*S
        Edot  = lamb*(S + theta*Sc) - sigma*E - 0.1*d*E
        Adot  = (1.0-p)*sigma*E - 0.1*d*A - gamma_A*A
        Idot  = p*sigma*E - d*I - gamma_I*I - mu*I
        Ddot  = d*(0.1*E+0.1*A+I) - gamma_D*D - mu*D
        Rdot  = gamma_A*A + gamma_I*I + gamma_D*D
        Vdot  = c1*A + c2*I - (mu_V+m)*V
        
        fx = np.array([Sdot, Scdot, Edot, Adot, Idot, Ddot, Rdot, Vdot])
        return fx
    
    # MODEL
    def rhs2(x,t,q):
        # CONTROL PARAMETERS
        alpha = alpha0
        theta = theta0
        d     = d0
        m     = m0
        
        beta_A= q[0]
        beta_I= q[1]
        beta_V= q[2]
        
        c1 = q[3]
        c2 = q[4]
        
        S  = x[0]
        Sc = x[1]
        E  = x[2]
        A  = x[3]
        I  = x[4]
        D  = x[5]
        R  = x[6]
        V  = x[7]
        
        lamb = beta_A*A + beta_I*I + beta_V*V
        
        Sdot  = -lamb*S - alpha*S
        Scdot = -lamb*theta*Sc + alpha*S
        Edot  = lamb*(S + theta*Sc) - sigma*E - 0.1*d*E
        Adot  = (1.0-p)*sigma*E - 0.1*d*A - gamma_A*A
        Idot  = p*sigma*E - d*I - gamma_I*I - mu*I
        Ddot  = d*(0.1*E+0.1*A+I) - gamma_D*D - mu*D
        Rdot  = gamma_A*A + gamma_I*I + gamma_D*D
        Vdot  = c1*A + c2*I - (mu_V+m)*V
        
        fx = np.array([Sdot, Scdot, Edot, Adot, Idot, Ddot, Rdot, Vdot])
        return fx
    
    x0 = np.array([S_0, Sc_0, E_0, A_0, I_0, D_0, R_0, V_0])
    
    mySol = integrate.odeint(rhs1,x0,time1,args=(q,))
    
    x0 = np.array([mySol[-1, 0],
                   mySol[-1, 1], 
                   mySol[-1, 2], 
                   mySol[-1, 3], 
                   mySol[-1, 4], 
                   mySol[-1, 5], 
                   mySol[-1, 6], 
                   mySol[-1, 7]])
    
    mySol1 = integrate.odeint(rhs2,x0,time2,args=(q,))
    
    mySol2 = np.concatenate((mySol, mySol1))
    return mySol2

def plot_SandI(susceptible, infected, Title):
    pl.figure(figsize=(5,3))
    pl.plot(time, susceptible,label="Susceptible")
    pl.fill_between(time, susceptible, 0, alpha=0.30)
    pl.plot(time, infected,label="Infected")
    pl.fill_between(time, infected, 0, alpha=0.30)
    pl.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    pl.xlabel('Time in days', fontsize=14)
    pl.ylabel('Individuals', fontsize=14)
    pl.title(Title)
    pl.legend()

def plot_I(susceptible, infected, Title):
    pl.figure(figsize=(5,3))
    pl.plot(time, infected,label="Infected", color=(0.8666666666666667, 0.5176470588235295, 0.3215686274509804))
    pl.fill_between(time, infected, 0, alpha=0.30, color=(0.8666666666666667, 0.5176470588235295, 0.3215686274509804))
    pl.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    pl.xlabel('Time in days', fontsize=14)
    pl.ylabel('Infected individuals', fontsize=14)
    pl.title(Title)
    
#-------------------#
#
# control(StartingDay, alpha, theta, d2, m)
#
#-------------------#
#
#my_soln1     = control(1, 0.01, 0.3, 0.0, 0.0)
#susceptible1 = my_soln1[:,0] + my_soln1[:,1]
#infected1    = my_soln1[:,4] + my_soln1[:,5]
#
#my_soln2     = control(15, 0.01, 0.3, 0.0, 0.0)
#susceptible2 = my_soln2[:,0] + my_soln2[:,1]
#infected2    = my_soln2[:,4] + my_soln2[:,5]
#
#my_soln3     = control(30, 0.01, 0.3, 0.0, 0.0)
#susceptible3 = my_soln3[:,0] + my_soln3[:,1]
#infected3    = my_soln3[:,4] + my_soln3[:,5]
#
#pl.figure(figsize=(5,3))
#pl.plot(time, infected1, label=r"From $t=1$")
#pl.fill_between(time, infected1, 0, alpha=0.30)
#pl.plot(time, infected2, label=r"From $t=15$")
#pl.fill_between(time, infected2, 0, alpha=0.30)
#pl.plot(time, infected3, label=r"From $t=30$")
#pl.fill_between(time, infected3, 0, alpha=0.30)
## pl.title("CM")
#pl.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
#pl.plot([0,tf],[1e6, 1e6],'k--')
#pl.legend()
#pl.xlabel('Time in days', fontsize=14)
#pl.ylabel('I+D', fontsize=14)
#pl.title(r"$\alpha=0.01, \theta=0.3$")
#pl.tight_layout()

#pl.savefig("control-pm-tstart.pdf")
#
#-------------------#
#
my_soln1     = control(1, 0.0, 0.0, 0.2, 0.0)
susceptible1 = my_soln1[:,0] + my_soln1[:,1]
infected1    = my_soln1[:,4] + my_soln1[:,5]

my_soln2     = control(15, 0.0, 0.0, 0.2, 0.0)
susceptible2 = my_soln2[:,0] + my_soln2[:,1]
infected2    = my_soln2[:,4] + my_soln2[:,5]

my_soln3     = control(30, 0.0, 0.0, 0.2, 0.0)
susceptible3 = my_soln3[:,0] + my_soln3[:,1]
infected3    = my_soln3[:,4] + my_soln2[:,5]

pl.figure(figsize=(5,3))
pl.plot(time, infected1, label=r"From $t=1$")
pl.fill_between(time, infected1, 0, alpha=0.30)
pl.plot(time, infected2, label=r"From $t=15$")
pl.fill_between(time, infected2, 0, alpha=0.30)
pl.plot(time, infected3, label=r"From $t=30$")
pl.fill_between(time, infected3, 0, alpha=0.30)
# pl.title("CM")
pl.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
pl.plot([0,tf],[1e6, 1e6],'k--')
pl.legend()
pl.xlabel('Time in days', fontsize=14)
pl.ylabel('I+D', fontsize=14)
pl.title(r"$d_2= 0.2$")
pl.tight_layout()

#pl.savefig("control-pm-tstart-d2.pdf")
#
#-------------------#
#
#my_soln1     = control(1, 0.00, 0.0, 0.0, 15.0)
#susceptible1 = my_soln1[:,0] + my_soln1[:,1]
#infected1    = my_soln1[:,4] + my_soln1[:,5]
#
#my_soln2     = control(15, 0.00, 0.0, 0.0, 15.0)
#susceptible2 = my_soln2[:,0] + my_soln2[:,1]
#infected2    = my_soln2[:,4] + my_soln2[:,5]
#
#my_soln3     = control(30, 0.00, 0.0, 0.0, 15.0)
#susceptible3 = my_soln3[:,0] + my_soln3[:,1]
#infected3    = my_soln3[:,4] + my_soln3[:,5]
#pl.savefig("control-pm-tstart-m.pdf")
#
#pl.figure(figsize=(5,3))
#pl.plot(time, infected1, label=r"From $t=1$")
#pl.fill_between(time, infected1, 0, alpha=0.30)
#pl.plot(time, infected2, label=r"From $t=15$")
#pl.fill_between(time, infected2, 0, alpha=0.30)
#pl.plot(time, infected3, label=r"From $t=30$")
#pl.fill_between(time, infected3, 0, alpha=0.30)
## pl.title("CM")
#pl.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
#pl.plot([0,tf],[1e6, 1e6],'k--')
#pl.legend()
#pl.xlabel('Time in days', fontsize=14)
#pl.ylabel('I+D', fontsize=14)
#pl.title(r"$m=15$")
#pl.tight_layout()
#pl.savefig("control-pm-tstart-m.pdf")
#
#-------------------#
#
#my_soln1     = control(1, 0.01, 0.5, 0.1, 1.0)
#susceptible1 = my_soln1[:,0] + my_soln1[:,1]
#infected1    = my_soln1[:,2]*IsE + my_soln1[:,3]*IsA + my_soln1[:,4]
#
#my_soln2     = control(15, 0.01, 0.5, 0.1, 1.0)
#susceptible2 = my_soln2[:,0] + my_soln2[:,1]
#infected2    = my_soln2[:,2]*IsE + my_soln2[:,3]*IsA + my_soln2[:,4]
#
#my_soln3     = control(30, 0.01, 0.5, 0.1, 1.0)
#susceptible3 = my_soln3[:,0] + my_soln3[:,1]
#infected3    = my_soln3[:,2]*IsE + my_soln3[:,3]*IsA + my_soln3[:,4]

#pl.figure(figsize=(5,3))
#pl.plot(time, infected1, label=r"From $t=1$")
#pl.fill_between(time, infected1, 0, alpha=0.30)
#pl.plot(time, infected2, label=r"From $t=15$")
#pl.fill_between(time, infected2, 0, alpha=0.30)
#pl.plot(time, infected3, label=r"From $t=30$")
#pl.fill_between(time, infected3, 0, alpha=0.30)
## pl.title("CM")
#pl.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
#pl.plot([0,tf],[1e6, 1e6],'k--')
#pl.legend()
#pl.xlabel('Time in days', fontsize=14)
#pl.ylabel('I+D', fontsize=14)
#pl.title(r"$\alpha=0.01, \theta=0.5, d_2= 0.1, m=1$")
#pl.tight_layout()
#pl.savefig("control-pm-tstart-multiple.pdf")
#
pl.show()

