import numpy as np
import random
from matplotlib import pyplot as plt

k_B = 1.380649e-23 #m^2 kg s^-2 K^-1

J = 1
T = 300

n = 5

state = 2*np.random.randint(2, size=(n,n))-1

def total_energy():
    E_t = 0
    for x in range(n):
        for y in range(n):
            neighbours = [state[(x + 1)%n, y], state[(x - 1)%n, y], state[x, (y + 1)%n], state[x, (y - 1)%n]]
            s_i = state[x,y]
            for neighbour in neighbours:
                E_t += - J * s_i * neighbours[neighbour]
                E_t *= 0.5
    return E_t
    
            

plt.imshow(state, origin='upper')
plt.pause(0.2)

def change_spin(state):
    x = random.randint(0,n-1)
    y = random.randint(0,n-1)
    s_i = -state[x,y]
    neighbours = [state[(x + 1)%n, y], state[(x - 1)%n, y], state[x, (y + 1)%n], state[x, (y - 1)%n]]
    E = E_t
    for elem in neighbours:
        E += - J * s_i * neighbours[elem]  
        D_E = E
    if D_E < 0:
        state[x,y] = s_i
        print('Success')
    else:
        R = random.random()
        p_i = np.exp(-D_E/(k_B*T))
        if R < p_i:
            state[x,y] = -state[x,y]
            print('Success Accept')
        else: 
            print('Fail')
    print(state, x, y)
    plt.imshow(state, origin='upper')
    plt.pause(0.1)

total_energy()
for i in range(10):
    change_spin(state)

#%%

T_red = k_B*T/J

E = -J*np.sum(s_i*s_j)

Z = np.sum(np.exp(-E_i/(k_B*T)))

p_i = np.exp(-D_E/(k_B*T))

#%%

