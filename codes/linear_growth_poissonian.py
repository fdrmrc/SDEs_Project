import numpy as np
import matplotlib.pyplot as plt
from StochasticIntegrators import Integrators_LSO #load 
num_traj = 1000
T = 10
dt = 0.01
ts = int(T/dt + 1)
Matr1 = np.zeros([num_traj, ts])
Matr2 = np.zeros([num_traj, ts])
y0=np.array([1,0])
t = np.linspace(0,T,ts)  
integr = Integrators_LSO(met_name='TRIG',dt=dt,y0=y0,alpha=0.6,eta=0.6,omega=3,theta = 0.5, T= T, lamb=3)
for i in range(num_traj):
    y = integr.evolve()
    Matr1[i,:] = y[0,:]
    Matr2[i,:] = y[1,:]
E = np.zeros(ts)
omega = 3
for k in range(ts):
    for j in range(num_traj):
        E[k] += 0.5*(omega**2 * Matr1[j,k]**2 + Matr2[j,k]**2)
E = (1/num_traj)*E
plt.plot(t,E, label='Second moment $E[x_t^2 + y_t^2]$')
plt.xlabel('t')
plt.legend()