import numpy as np
import random
import pandas as pd
from matplotlib import pyplot as plt
import time

k_B = 1.380649e-23 #m^2 kg s^-2 K^-1

J = 1
T = 1

n = 5

state = 2*np.random.randint(2, size=(n,n))-1
E_t = 1e-26

plt.imshow(state, origin='upper')

def change_spin(state, E_t):
    x = random.randint(0,n-1)
    y = random.randint(0,n-1)
    s_i = -state[x][y]
    s_j = [state[(x+1)%n][y], state[(x-1)%n][y], state[x][(y+1)%n], state[x][(y-1)%n]]
    E = 0
    for elem in s_j:
        E = E_t + E - J * s_i * s_j[elem]  
        D_E = E - E_t
    if D_E < 0:
        state[x][y] = -state[x][y]
    else:
        R = random.random()
        p_i = np.exp(-D_E/(k_B*T))
        if R < p_i:
            state[x][y] = -state[x][y]
            print('Success')
        else: 
            print('Fail')
            pass
    plt.imshow(state, origin='upper')
    time.sleep(0.2)


for i in range(10):
    change_spin(state, E_t)

#%%

R = random.random()

x, y = np.meshgrid(np.linspace(0, n, n), np.linspace(0, n, n))
s = random.randint(1,2)

fig, ax = plt.subplots(s)


plt.imshow()
#%%

T_red = k_B*T/J

E = -J*np.sum(s_i*s_j)

Z = np.sum(np.exp(-E_i/(k_B*T)))

p_i = np.exp(-D_E/(k_B*T))

#%%

