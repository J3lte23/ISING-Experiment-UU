import numpy as np
import random
from matplotlib import pyplot as plt


#Use seed for report

k_B = 1.380649e-23 #m^2 kg s^-2 K^-1

T_red = 2.27
J_red = 1/T_red
J = 1
T = T_red*J/k_B

beta = 1/(k_B*T)

n = 100


#state = 2*np.random.randint(2, size=(n,n))-1
#state = np.random.choice([-1, 1], size=(n, n), p=[0.2, 0.8])

E_t_list = np.array([])
m_list = np.array([])
av_E_t = np.array([])
av_E_t_2 = np.array([])
av_m = np.array([])
av_abs_m = np.array([])
av_cap = np.array([])

def total_energy():
    global E_t
    E_t = 0
    for x in range(n):
        for y in range(n):
            s_i = state[x,y]
            neighbours = [state[(x+1)%n, y], state[x, (y+1)%n]]
            for neighbour in neighbours:
                E_t += - J * s_i * neighbour     
    return E_t

def change_spin(state):
    global E_t
    x = random.randint(0,n-1)
    y = random.randint(0,n-1)
    neighbours = [state[(x+1)%n, y], state[(x-1)%n, y], state[x, (y+1)%n], state[x, (y-1)%n]]
    #Trust
    D_E = 2 * J * state[x,y] * np.sum(neighbours)
    if D_E < 0:
        state[x,y] *= -1
        E_t += D_E
    elif random.random() < np.exp(-D_E / (k_B * T)):
        state[x, y] *= -1
        E_t += D_E
    else:
        pass

for run in range(3):
    state = 2*np.random.randint(2, size=(n,n))-1
    total_energy()
    plt.imshow(
        state, 
        origin='upper',
        cmap='Grays'
    )
    plt.pause(0.001)
    for i in range(100):
        for i in range(10000):
            change_spin(state)
        E_t_list = np.append(E_t_list, E_t)
        m_list = np.append(m_list, np.mean(state))
        
        plt.imshow(
            state, 
            origin='upper',
            cmap='Grays'
        )
        plt.pause(0.001)
        
    av_E_t = np.append(av_E_t, np.mean(E_t_list))
    av_E_t_2 = np.append(av_E_t_2, np.mean(E_t_list**2))
    av_m = np.append(av_m, np.mean(m_list))   
    av_abs_m = np.append(av_abs_m, np.mean(np.abs(m_list)))
    av_cap = np.append(av_cap, k_B*beta**2*(np.mean(E_t_list**2)-np.mean(E_t_list)**2))
    
    E_t_list = np.array([])
    
print(av_E_t)
print(av_E_t_2)
print(av_m)
print(av_abs_m)
#%%
#determine T_red/T_c
t_red_arr= np.linspace(0,10, num=50)/2.27
 
c_arr = 8*k_B/np.pi*(1/8*J)**2*np.log(np.abs(1/(t_red_arr-1)))
plt.plot(t_red_arr,c_arr)
