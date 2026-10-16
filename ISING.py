import numpy as np
from matplotlib import pyplot as plt

#np.random.seed(100)

#Use seed for report

steps = 21
mask = 50

T_c = 2/np.log(1+np.sqrt(2)) 

T_red_space = np.linspace(0.7, 1.3, steps)*T_c

n = 50

E_t_list = np.array([])
m_list = np.array([])
av_E_t = np.array([])
av_E_t_unc = np.array([])
av_E_t_2 = np.array([])
av_m = np.array([])
av_m_unc = np.array([])
av_m_2 = np.array([])
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
    x = np.random.randint(0,n)
    y = np.random.randint(0,n)
    neighbours = [state[(x+1)%n, y], state[(x-1)%n, y], state[x, (y+1)%n], state[x, (y-1)%n]]
    D_E = 2 * J * state[x,y] * np.sum(neighbours)
    if D_E < 0:
        state[x,y] *= -1
        E_t += D_E
    elif np.random.random() < np.exp(-1 * D_E / T):
        state[x, y] *= -1
        E_t += D_E
    else:
        pass

for run in range(len(T_red_space)):
    J = 1 #J
    T = T_red_space[run]

    print(T_red_space[run]/T_c)
    
    state = 2*np.random.randint(2, size=(n,n))-1
    
# =============================================================================
#     state = np.ones_like(state)
# =============================================================================

    total_energy()
    
# =============================================================================
#     plt.imshow(
#         state, 
#         origin='upper',
#         cmap='Grays'
#     )
#     plt.pause(0.001)
# =============================================================================

    for i in range(100):
        for i in range(10000):
            change_spin(state)
        E_t_list = np.append(E_t_list, E_t)
        m_list = np.append(m_list, np.mean(state))
        
# =============================================================================
#         plt.imshow(
#             state, 
#             origin='upper',
#             cmap='Grays'
#         )
#         plt.pause(0.001)
#         
# =============================================================================
    av_E_t = np.append(av_E_t, np.mean(E_t_list[mask:]))
    av_E_t_2 = np.append(av_E_t_2, np.mean(E_t_list[mask:]**2))
    av_m = np.append(av_m, np.mean(m_list[mask:])) 
    av_m_2 = np.append(av_m_2, np.mean(m_list[mask:]**2))
    av_abs_m = np.append(av_abs_m, np.mean(np.abs(m_list[mask:])))
    av_cap = np.append(av_cap, (np.log(1 + np.sqrt(2)) / 2)**2*(np.mean(E_t_list[mask:]**2)-np.mean(E_t_list[mask:])**2))
    
    av_E_t_unc = np.append(av_E_t_unc, np.std(E_t_list[mask:]))
    av_m_unc = np.append(av_m_unc, np.std(m_list[mask:]))
    
    E_t_list = np.array([])
    m_list = np.array([])

unc_list = np.sqrt(av_E_t_2 - av_E_t**2)

print(av_E_t)
print(av_E_t_2)
print(av_m)
print(av_abs_m)
print(av_cap)

print(unc_list)

C_unc = np.sqrt(np.mean(av_cap**2) - np.mean(av_cap)**2)/np.sqrt(len(av_cap))

#%%

T_red_plt = np.linspace(0.6, 1.4, num=100000)

beta_c = np.log(1 + np.sqrt(2)) / 2

c_arr = (
    8 / np.pi * (beta_c * J)**2
    * np.log(1 / np.abs(T_red_plt - 1))
)

plt.axvline(1, color='red', linestyle='--', label='Critical Temperature T_c')

for run in range(len(T_red_space)):
    plt.scatter(T_red_space[run] / T_c, av_cap[run] / n**2)
    plt.errorbar(T_red_space[run] / T_c, av_cap[run] / n**2, yerr=np.std(av_cap)/n**2)

plt.plot(T_red_plt, 
         c_arr, 
         label='Theoretical Model'
)
plt.xlabel(r'$T/T_c$')
plt.ylabel(r'$C/(Nk_B)$')
plt.legend()
plt.show()

#%%

T_red_plt = np.linspace(0.01, 1.4, 1000)

beta_mag = 1/8

M_theory = np.zeros_like(T_red_plt)

below_Tc = T_red_plt < 1
T_actual = T_red_plt[below_Tc] * T_c

M_theory = (
    1 - T_red_plt
)**beta_mag

plt.errorbar(
    T_red_space / T_c,
    av_abs_m,
    yerr=av_m_unc,
    fmt='o',
    label='Monte Carlo'
)

plt.plot(
    T_red_plt,
    M_theory,
    label='Exact theoretical model'
)

plt.axvline(1, color='red', linestyle='--', label=r'$T_c$')
plt.xlabel(r'$T/T_c$')
plt.ylabel(r'$|M|/N$')
plt.xlim(0, 1.4)
plt.ylim(bottom=0)
plt.legend()
plt.tight_layout()
plt.show()

#%%

T_red_plt = np.linspace(0.01, 1.4, 1000)

beta_mag = 1/8

M_theory = np.zeros_like(T_red_plt)

below_Tc = T_red_plt < 1
T_actual = T_red_plt[below_Tc] * T_c

M_theory = (
    1 - T_red_plt
)**beta_mag

plt.errorbar(
    T_red_space / T_c,
    av_m,
    yerr=av_m_unc,
    fmt='o',
    label='Monte Carlo'
)

plt.plot(
    T_red_plt,
    M_theory,
    label='Exact theoretical model'
)

#%%

plt.plot(
    T_red_space/T_c,
    av_E_t,
    label='Exact theoretical model'
)

plt.errorbar(
    T_red_space / T_c,
    av_E_t,
    yerr=av_E_t_unc
)

plt.legend()
plt.tight_layout()
plt.show()
