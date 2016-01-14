# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

""" Numerically solve Kuramoto with Lorentzian coupling"""


from scipy.stats import cauchy
from scipy.integrate import odeint 
import prettyplotlib as pp  #for pretty plotting
import numpy as np
import math


#Define equations -- this is used by the ODE solver and represents the RHS of the Eq: d(theta)/dt = f(theta, t)
def g(theta, t, omega, K):
    """ theta = list of oscillators
        t = time
        K = list of coupling strengths
    """
    
    eqns = []
    for i in range(len(theta)):
        temp = 0.0
        for j in range(len(theta)):
            temp += K[i]*sin(theta[j] - theta[i])
        eqns.append(omega[i] + temp / len(theta))
    return eqns



def make_positive_cauchy_rv(N, mean, var):
    """ Makes a list of N POSITIVE random variates drawn from a cauchy dist
        with given mean and var (or whatever pars define the dist).
        
    """
    
    omega = []
    for i in range(N):
        temp = -1
        while temp < 0:
            temp =  float(cauchy.rvs(size = 1, scale = var_omega,  loc = mean_omega))
        omega.append(temp)
    return omega




def make_bif_diagram(num_oscillators, t_max, average_k_range):
    
    """ theta = list of time series of each oscillator
        average_k = list of average coupling strengths
        
        I've defined some parameters inside the function 
        instead of making them arguements.
        
    """
    
    #Define Parameters
    mean_omega, var_omega, var_k = 20, 1, 5
    omega = make_positive_cauchy_rv(num_oscillators, mean_omega, var_omega)
    IC = [np.random.uniform(0,2*np.pi) for i in range(num_oscillators)]   #initial conditions


    #ODE solver parameters
    num_time_steps = 100.0
    dt = t_max / num_time_steps
    t = [i*dt for i in range(int(num_time_steps) + 1)]
    
    
    #Solve ODE's
    bif_data = []    #bifurcation data

    for mean_k in average_k_range:
        K = make_positive_cauchy_rv(num_oscillators, mean_k, var_k)
        sols = odeint(lambda theta, t: g(theta,t,omega,K), IC, t)
        print 'Started <k> = ' + str(mean_k)
        
    
    #Take last 20% of R, and average to find R_inf
        temp = [abs(sum(exp(sols[-i,:]*1j)) / num_oscillators) for i in range((len(sols))/20)]
        R_inf = np.mean(temp)
        bif_data.append([mean_k, R_inf])
    
    
    #Plot Solution
    temp1 = np.array(bif_data)
    pp.plot(temp1[:,0], temp1[:,1], '-o')
    pp.plt.title('For N = ' + str(num_oscillators) + ' oscillators')
    pp.plt.ylim([0,1])
    pp.plt.xlabel('<k>')
    pp.plt.ylabel('$ R_{\infty} $')
    
    return 0


#--------------------- PARAMETERS BELOW --------------------------


#Define Parameters
num_oscillators, mean_omega, var_omega, mean_k, var_k = 100, 20, 1, 20, 10
omega = make_positive_cauchy_rv(num_oscillators, mean_omega, var_omega)
IC = [np.random.uniform(0,2*np.pi) for i in range(num_oscillators)]   #initial conditions


#ODE solver parameters
t_max = 5
num_time_steps = 200.0
dt = t_max / num_time_steps
t = [i*dt for i in range(int(num_time_steps) + 1)]


t_max , average_k_range = 5, [5]
make_bif_diagram(num_oscillators, t_max, average_k_range)

# <codecell>


