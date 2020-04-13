# -*- Coding: UTF-8 -*-
#coding: utf-8


### Eduardo Ometto Gebara -  April/2020 ###

import numpy as np
from scipy.integrate import odeint
import math
import matplotlib.pyplot as plt


#References

# 1) https://scipython.com/book/chapter-8-scipy/additional-examples/the-sir-epidemic-model/
# 2) https://towardsdatascience.com/social-distancing-to-slow-the-coronavirus-768292f04296

# Total population, N

N = 12.18e6 # São Paulo

# Initial number of infected, recovered and exposed individuals

E0 = 1000
I0 = 2000
Q0 = 2000
R0 = 1000
D0 = 400
C0 = 0
Ic0 = 7000

# Everyone else, S0, is susceptible to infection initially

S0 = N - E0 - I0 - Q0 - R0 - D0 - C0 

# Parameters 

mu = (10/(10*365))
alpha = 0.03
beta = 1.288
gamma = 0.870
omega = 0.585
lambda0 = 0.057
lambda1 = 0.026
k0 = 0.070
k1 = 0.144

# A grid of time points (in days)

t = np.linspace(0, 300, 300)

# parameters lambda and k are time dependent

def lambdat(t):

	return lambda0*(1-math.exp(-lambda1*t))
				

def k(t):

	return k0*math.exp(-k1*t) 
	

# The SEIR model differential equations

def deriv(y, t, N, mu, alpha, beta, gamma, omega):
    
    S, E, I, Q, R, D, C, Ic = y

    dSdt = mu*N - alpha*S - beta*S*I/N - mu*S
    dEdt = beta*S*I/N - gamma*E - mu*E
    dIdt = gamma*E - omega*I - mu*I
    dQdt = omega*I - lambdat(t)*Q - k(t)*Q - mu*Q
    dRdt = lambdat(t)*Q - mu*R
    dDdt = k(t)*Q
    dCdt = alpha*S - mu*C
    dIcdt = gamma*E
    


    return dSdt, dEdt, dIdt, dQdt, dRdt, dDdt, dCdt, dIcdt

# Initial conditions vector

y0 = S0, E0, I0, Q0, R0, D0, C0, Ic0

# Integrate the SEIR equations over the time grid, t

ret = odeint(deriv, y0, t, args=(N, mu, alpha, beta, gamma, 
	omega))

S, E, I, Q, R, D, C, Ic = ret.T

# Plot

fig = plt.figure(facecolor='w')
ax = fig.add_subplot(111, axis_bgcolor='#dddddd', axisbelow=True)
#ax.plot(t, S, 'b', alpha=0.5, lw=2, label='Susceptible')
ax.plot(t, E, 'm', alpha=0.5, lw=2, label='Exposed')
ax.plot(t, I, 'r', alpha=0.5, lw=2, label='Infected')
ax.plot(t, Q, 'indigo', alpha=0.5, lw=2, label='Quarantined')
ax.plot(t, R, 'g', alpha=0.5, lw=2, label='Recovered')
ax.plot(t, D, 'black', alpha=0.5, lw=2, label='Dead')
ax.plot(t, C, 'c', alpha=0.5, lw=2, label='Confined Susceptible')
ax.plot(t, Ic, 'y', alpha=0.5, lw=2, label='Cumulative Infected')
ax.set_xlabel('Time (days)')
ax.set_ylabel('Number')
ax.grid(b=True, which='major', c='w', lw=2, ls='-')
legend = ax.legend()
legend.get_frame().set_alpha(0.5)

tmax = t[np.argmax(I)] # date when there is the max number of infected people at once
Imax = I.max() # max number of infected people at once 
Imax_mi = Imax/1e6 # Imax in millions 

text= "t={:.3f} days, Imax={:.3f} millions".format(tmax, Imax_mi)
ax.annotate(text, xy=(tmax, Imax)) # generating annotation in plot

ylim = N + 1e6 # upper limit of the y axis 
tmax_norm = tmax/t[-1] # normalizing to plot vertical dashed line
Imax_norm = Imax/(ylim) # normalizing to plot horizontal dashed line 

ax.axvline(x=tmax, ymin=0, ymax=Imax_norm, linestyle='--', color='black') # generating vertical line 
ax.axhline(y=Imax, xmin=0, xmax=tmax_norm, linestyle='--', color='black') # generating horizontal line 
         

ax.set_ylim(0,ylim)

plt.show()

print(max(D))
