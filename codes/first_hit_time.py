#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 17:38:09 2020

@author: marco
"""


import numpy as np
import matplotlib.pyplot as plt
from StochasticIntegrators import Integrators_LSO


y0 = np.array([1,0])
dt = 0.001

T = 8
integr = Integrators_LSO(met_name='EX',dt=dt,y0=y0,alpha=1,eta=1,theta = 0.5, T=T, lamb=2)
ts = int(T/dt + 1)
t = np.linspace(0,T,ts)
num_simu = 100
expect = np.zeros(num_simu)
for n in range(num_simu):
    y = integr.evolve()
    for i in range(ts-1):
        if (y[0,i]*y[0,i+1]<0): 
            expect[n] = (t[i]+t[i+1])/2 #take average 
            break
            
print('Expected value first hitting time is: ',np.sum(expect)/num_simu)
                
'''            
plt.figure(figsize=(8,5))
for i in range(6):
    y = integr.evolve()
    plt.plot(t,y[0,:],'-')
    plt.xlabel('t')
    plt.ylabel('x(t)')
    plt.axhline(y=0, color= 'r', linestyle = '-.')
    plt.axvline(x=2*np.pi,color='g',linestyle='-.')
    plt.title('Realizations')
plt.legend()
'''