import numpy as np
import random
from matplotlib import pyplot as plt


#Use seed for report

k_B = 1.380649e-23 #m^2 kg s^-2 K^-1

T_red = 0.5
J_red = 1
J = 1/T_red
T = T_red*J/k_B

n = 20


#state = 2*np.random.randint(2, size=(n,n))-1
state = np.random.choice([-1, 1], size=(n, n), p=[0.2, 0.8])


def total_energy():
    global E_t
    E_t = 0
    for x in range(n):
        for y in range(n):
            neighbours = [state[(x+1)%n, y], state[(x-1)%n, y], state[x, (y+1)%n], state[x, (y-1)%n]]
            s_i = state[x,y]
            for neighbour in neighbours:
                E_t += - J * s_i * neighbour
    E_t *= 0.5
    print(E_t)
    return E_t
    
        

plt.imshow(
    state, 
    origin='upper',
    cmap='Grays'
)
plt.pause(0.2)

def change_spin(state):
    global E_t
    x = random.randint(0,n-1)
    y = random.randint(0,n-1)
    neighbours = [state[(x+1)%n, y], state[(x-1)%n, y], state[x, (y+1)%n], state[x, (y-1)%n]]
    D_E = 2 * J * state[x,y] * np.sum(neighbours)
    if D_E < 0:
        state[x,y] *= -1
        print('Success')
        E_t += D_E
    elif random.random() < np.exp(-D_E / (k_B * T)):
        state[x, y] *= -1
        print('Success Accept')
        E_t += D_E
    else:
        print('Fail')
    plt.imshow(
        state, 
        origin='upper', 
        cmap='Grays'
    )
    plt.pause(0.00001)

total_energy()
for i in range(1e3):
    change_spin(state)
print(E_t)

m = (1/(n**2)) * np.sum(state)
print(m)
