import numpy as np
import random
import pandas as pd
from matplotlib import pyplot as plt

k_B = 1.380649e-23 #m^2 kg s^-2 K^-1

J = 1
T = 1

n = 5

state = 2*np.random.randint(2, size=(n,n))-1

plt.imshow(state, origin='upper')

def change_spin(state):
    x = random.randint(0,n-1)
    y = random.randint(0,n-1)
    state[x][y] = -state[x][y]
    print(state)
    plt.imshow(state)


change_spin(state)

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

