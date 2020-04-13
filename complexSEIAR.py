### Eduardo Ometto Gebara -  April/2020 ###

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

#References

# 1) https://scipython.com/book/chapter-8-scipy/additional-examples/the-sir-epidemic-model/
# 2) https://towardsdatascience.com/social-distancing-to-slow-the-coronavirus-768292f04296

# Total population, N

N = 12.18e6 # São Paulo

# Initial number 

E0 = 0
I0 = 0
A0 = 0
J0 = 0
R0 = 0
D0 = 0

# Everyone else, S0, is susceptible to infection initially

S0 = N - E0 - I0 - A0 - J0 - R0 - D0

# Parameters 

mu = 0 #natality and mortality rates in population 
q = 0 #reduction (0 to 1) in transmissibility of Asymptomatics
beta = 0 #transmission rate 
kappa = 0 #rate in which exposed become infected 
rho = 0 #fraction (0 to 1) of exposed that become infected
gamma1 = 0 # A goes to R at this rate 
alpha = 0 #I goes to J at this rate 
omega = 0 #J goes to D at this rate 
gamma2 = 0 #J goes to R at this rate 


# A grid of time points (in days)

t = np.linspace(0, 300, 300)

# The SEIR model differential equations

def deriv(y, t, N, mu, q, beta, kappa, rho, gamma1, 
	alpha, omega, gamma2):
    
    S, E, I, A, J, R, D = y
    
    dSdt = mu*N - beta*S*(I + J + q*A)/N - mu*S
    dEdt = beta*S*(I + J + q*A)/N - (kappa + mu)*E
    dAdt = kappa*(1-rho)*E - (gamma1 + mu)*A
    dIdt = kappa*rho*E - (alpha + gamma1 + mu)*I
    dJdt = alpha*I - (omega + gamma2 + mu)*J
    dRdt = gamma1*(A + I) + gamma2*J - mu*R
    dDdt = omega*J

    return dSdt, dEdt, dAdt, dIdt, dJdt, dRdt, dDdt

# Initial conditions vector

y0 = S0, E0, I0, A0, J0, R0, D0

# Integrate the SEIR equations over the time grid, t

ret = odeint(deriv, y0, t, args=(N, mu, q, beta, 
	kappa, rho, gamma1, alpha, omega, gamma2 ))

S, E, I, A, J, R, D = ret.T

# Plot

fig = plt.figure(facecolor='w')
ax = fig.add_subplot(111, axis_bgcolor='#dddddd', axisbelow=True)
ax.plot(t, S, 'b', alpha=0.5, lw=2, label='Susceptible')
ax.plot(t, E, 'm', alpha=0.5, lw=2, label='Exposed')
ax.plot(t, I, 'r', alpha=0.5, lw=2, label='Infected')
ax.plot(t, R, 'g', alpha=0.5, lw=2, label='Recovered')
ax.set_xlabel('Time (days)')
ax.set_ylabel('Number')
ax.yaxis.set_tick_params(length=0)
ax.xaxis.set_tick_params(length=0)
ax.grid(b=True, which='major', c='w', lw=2, ls='-')
legend = ax.legend()
legend.get_frame().set_alpha(0.5)
for spine in ('top', 'right', 'bottom', 'left'):
    ax.spines[spine].set_visible(False)


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


