# -*- Coding: UTF-8 -*-
#coding: utf-8

### Eduardo Ometto Gebara -  April/2020 ###

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

#References

# 1) https://scipython.com/book/chapter-8-scipy/additional-examples/the-sir-epidemic-model/
# 2) https://towardsdatascience.com/social-distancing-to-slow-the-coronavirus-768292f04296

# Total population, N

N = 12.18e6 # São Paulo

# Initial number of infected, recovered and exposed individuals

I0 = 2666
R0 = 1800
E0 = 10000
D0 = 260 

# Everyone else, S0, is susceptible to infection initially

S0 = N - E0 - I0 - R0- D0

# Parameters 

alpha = 1/5.1 # 1/days of incubation period 
R_0 = 2.27 # how quickly the disease can spread 
gamma = 1/1.61 # mean 1/days to recover
beta = gamma*R_0 # mean contact rate in population
rho = 0.5 # social distancing from 0 (everyone in querentine) to 1 (no one in querentine)
omega = 0.008


# A grid of time points (in days)

t = np.linspace(0, 300, 300)

# The SEIR model differential equations

def deriv(y, t, N, beta, gamma,alpha, omega, rho):
    
    S, E, I, R, D = y
    
    dSdt = -rho*beta*S*I/N
    dEdt = rho*beta*S*I/N - alpha*E
    dIdt = alpha*E - gamma*I - omega*I
    dRdt = gamma*I
    dDdt = omega*I

    return dSdt, dEdt, dIdt, dRdt, dDdt

# Initial conditions vector

y0 = S0, E0, I0, R0, D0

# Integrate the SEIR equations over the time grid, t

ret = odeint(deriv, y0, t, args=(N, beta, gamma, alpha, omega, rho))
S, E, I, R, D= ret.T

# Plot

fig = plt.figure(facecolor='w')
ax = fig.add_subplot(111, axis_bgcolor='#dddddd', axisbelow=True)
#ax.plot(t, S, 'b', alpha=0.5, lw=2, label='Susceptible')
#ax.plot(t, E, 'm', alpha=0.5, lw=2, label='Exposed')
ax.plot(t, I, 'r', alpha=0.5, lw=2, label='Infected')
#ax.plot(t, R, 'g', alpha=0.5, lw=2, label='Recovered')
ax.plot(t, D, 'black', alpha=0.5, lw=2, label='Dead')
ax.set_xlabel('Time (days)')
ax.set_ylabel('Number')
ax.grid(b=True, which='major', c='w', lw=2, ls='-')
legend = ax.legend()
legend.get_frame().set_alpha(0.5)

tmax = t[np.argmax(I)] # date when there is the max number of infected people at once
Imax = I.max() # max number of infected people at once 
#Imax_mi = Imax/1e6 # Imax in millions 

text= "t={:.3f} days, Imax={:.3f} people".format(tmax, Imax)
ax.annotate(text, xy=(tmax, Imax)) # generating annotation in plot

#ylim = N  # upper limit of the y axis 
#tmax_norm = tmax/t[-1] # normalizing to plot vertical dashed line
#Imax_norm = Imax/(ylim) # normalizing to plot horizontal dashed line 

#ax.axvline(x=tmax, ymin=0, ymax=Imax_norm, linestyle='--', color='black') # generating vertical line 
#ax.axhline(y=Imax, xmin=0, xmax=tmax_norm, linestyle='--', color='black') # generating horizontal line 
         

#ax.set_ylim(0,ylim)

plt.show()

print(max(D))
