# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
#from numba import jit

# normalization constant
omega_D = 1                                                                                                     # DEFINE 1 / s
Debye_length = 1                                                                                                # DEFINE m
elementary_charge = 1.602 * 1e-19                                                                               # C
elementary_mass = 9.109 * 1e-31                                                                                 # kg
epsilon0 = 8.854 * 1e-12                                                                                        # F / m

n0 = omega_D * omega_D * elementary_mass * epsilon0 / elementary_charge / elementary_charge                     # 1 / m^3
v_nor = Debye_length * omega_D                                                                                  # m / s
rho_nor = elementary_charge * n0                                                                                # C / m^-3
phi_nor = Debye_length * Debye_length * n0 * elementary_charge / epsilon0
E_nor = Debye_length * n0 * elementary_charge / epsilon0
energy_nor =  Debye_length **3 * omega_D **4 * elementary_mass **2 * epsilon0 / elementary_charge **2
momentum_nor = Debye_length **2 * omega_D **3 * elementary_mass **2 * epsilon0 / elementary_charge **2

# fdmt constant
epsilon = 8.854 * 1e-12 / epsilon0 
q_e = -1.602 * 1e-19 / elementary_charge      
m_e = 9.109 * 1e-31 / elementary_mass     
q_i = -q_e                           
m_i = 1836 * m_e                      

# initial condition
GN = int(2 * np.pi /0.6124/0.1)        
ppc = 500                            
PN = ppc * GN 
dx = 0.1 / Debye_length          
dt = 0.1 * omega_D      
x_electron = GN * dx * np.random.rand(PN) 
t_max = 1000 * dt    
rho_macro = ppc / dx            # 巨粒子密度

N_real = n0 * dx / n0          # 一個格點實際粒子數量(n0*dx*GN)，除以n0表示對n0做歸一化
weight = N_real / ppc           # 一個巨粒子所代表實際粒子的數量

v_electron = np.zeros(PN) 
for i in range(PN):
    rand_num = random.gauss(1, 0.3)
    if i < 0.5 * PN:
        v_electron[i] = 1 * rand_num / v_nor
    else:
        v_electron[i] = -1 * rand_num / v_nor

U_energy = []
K_energy = []
Momentum = []
# @jit
def first_order_weighting(x_electron):
    tmp = x_electron // dx                             
    tmp1 = (x_electron - (tmp) * dx) / dx                                          
    rho = np.zeros(GN)
    rho_tmp = np.zeros(GN+1)
    for i in range(GN+1):
        for j in range(PN):
            if i == tmp[j]:
                rho_tmp[i] += q_e * (1 - tmp1[j])           
                rho_tmp[i+1] += q_e * tmp1[j]       
                                                            # q_e: 平行帶電板，單位面積的庫倫C/m/m
    rho = rho_tmp[0:GN]                                     # represent q * number of particle
    rho[0] += rho_tmp[GN]                                
    #print(np.sum(rho))                               
    rho -= np.sum(rho) / GN                      
    rho = rho / ppc                                   # 要考慮每個particle的weighting
                                                        
    '''
    print('x_electron', x_electron)
    print('temp',temp)
    print('temp1', temp1)
    print('temp2', temp2)
    print('rho_temp', rho_temp)
    print('rho', rho)   
    print('rho', rho)
    print(np.sum(rho)) 
    '''
    return rho, tmp, tmp1
# @jit
def Poisson(rho):
    phi = np.zeros(GN)
    phi[GN-1] = 0   # reference point

    phi[0] = 0
    sum = 0
    for i in range(GN):
        sum = sum + rho[i] * (i + 1)             
    phi[0] = - sum / GN / epsilon * dx * dx       

    phi[1] = -rho[0] * dx * dx / epsilon + 2 * phi[0] - phi[GN-1]

    for i in range(2, GN-1):
        phi[i] = -rho[i-1] * dx * dx / epsilon + 2 * phi[i-1] - phi[i-2]

    return phi                        
# @jit
def electric_field(phi): 
    E_field = np.zeros(GN)
    E_field[0] = phi[1] - phi[GN-1]
    for i in range(1, GN-1):
        E_field[i] = phi[i+1] - phi[i-1]
    E_field[GN-1] = phi[0] - phi[GN-2]
    E_field = - E_field / 2 / dx   
    return E_field                
# @jit
def weighing_to_particle(E_field, tmp, tmp1): 

    E_nx_electron = np.zeros(PN)
    E_field_tmp = np.zeros(GN+1)
    E_field_tmp[0:GN] = E_field                   
    E_field_tmp[GN] = E_field[0]                    
    
    for i in range(PN):
        for j in range(GN):
            if tmp[i] == j: #j是哪個格點 i是哪個栗子
                E_nx_electron[i] = E_field_tmp[j] * (1 - tmp1[i]) + E_field_tmp[j+1] * tmp1[i] 
    return E_nx_electron
# @jit
def Euler_velocity(dot_a, initial):  
    a = np.zeros(np.size(initial))
    a = initial + dot_a * dt      
    return a
# @jit
def Euler_position(dot_a, initial): 
    a = np.zeros(np.size(initial))
    a = initial + dot_a * dt       
    for i in range(PN):
        if a[i] > GN * dx:
            a[i] = a[i] - GN * dx
        elif a[i] < 0:
            a[i] = a[i] + GN * dx
    return a
    
plt.style.use('dark_background')
fig = plt.figure(figsize=(14, 6), dpi=200)
ax = fig.subplots(1,2)

step = 0
def run(data):
    global x_electron, v_electron, step

    rho, tmp, tmp1 = first_order_weighting(x_electron)  
    phi = Poisson(rho)                              
    
    E_field = electric_field(phi) 
    E_nx_electron = weighing_to_particle(E_field, tmp, tmp1) 
    v_electron = Euler_velocity(q_e / m_e * E_nx_electron, v_electron) 
    x_electron = Euler_position(v_electron, x_electron)

    ax[0].clear()
    ax[0].set_xlabel('position')
    ax[0].set_ylabel('velocity')
    ax[0].set_xlim(0, GN * dx)
    ax[0].set_ylim(-2.5, 2.5)
    ax[0].set_title('phase space')
    ax[0].plot(x_electron[0:int(0.5*PN)], v_electron[0:int(0.5*PN)], '*', markersize = 0.5, color = 'lightcoral') 
    ax[0].plot(x_electron[int(0.5*PN):-1], v_electron[int(0.5*PN):-1], '*', markersize = 0.5, color = 'cornflowerblue') 

    U_energy_sum = 0
    for i in range(GN):
        U_energy_sum = U_energy_sum + 0.5 * E_field[i] * E_field[i] * epsilon * dx

    K_energy_sum = 0
    Momentum_sum = 0
    for i in range(PN):
        K_energy_sum = K_energy_sum + 0.5 * m_e * v_electron[i] * v_electron[i] * weight
        Momentum_sum = Momentum_sum + m_e * v_electron[i] * weight

    U_energy.append(U_energy_sum)
    K_energy.append(K_energy_sum)
    Momentum.append(Momentum_sum)

    ax[1].clear()
    ax[1].set_xlabel('time')
    ax[1].set_ylabel('energy')
    ax[1].set_xlim(0, 1000)
    # ax[1].set_ylim(-2.5, 2.5)
    ax[1].set_title('energy')
    ax[1].plot(np.array(U_energy), label = 'field energy', color = 'yellowgreen')
    ax[1].plot(np.array(K_energy), label = 'kinetic energy', color = 'gold')
    ax[1].plot((np.array(U_energy) + np.array(K_energy)), label = 'total energy', color = 'plum')
    ax[1].legend(loc = 'center right')

    step += 1

ani = animation.FuncAnimation(fig, run, frames=1000, interval=30)
ani.save('/Users/sandy1114/Desktop/ani/animation.mov', fps=30)
plt.show()

# while t < t_max and step < 300:

#     rho, tmp, tmp1 = first_order_weighting(x_electron)  
#     phi = Poisson(rho)                              
#     '''
#     rho_test = np.zeros(GN)
#     rho_test[0] = phi[1] - 2 * phi[0] + phi[GN-1]
#     for i in range(1, GN-1):
#         rho_test[i] = phi[i+1] - 2 * phi[i] + phi[i-1]
#     rho_test[GN-1] = phi[0] - 2 * phi[GN-1] + phi[GN-2]
#     rho_test = rho_test / dx / dx * epsilon * (-1)
#     plt.plot(np.linspace(0, dx * (GN-1), GN), rho_test, '.', label = 'rho test')
#     '''
#     E_field = electric_field(phi) 
#     E_nx_electron = weighing_to_particle(E_field, tmp, tmp1) 
#     v_electron = Euler_velocity(q_e / m_e * E_nx_electron, v_electron) 
#     x_electron = Euler_position(v_electron, x_electron)
    
#     if step % 10 == 0:
#         plt.style.use('dark_background')
#         fig, ax = plt.subplots()

#         plt.plot(x_electron[0:int(0.5*PN)], v_electron[0:int(0.5*PN)], '*', markersize = 0.5) 
#         plt.plot(x_electron[int(0.5*PN):-1], v_electron[int(0.5*PN):-1], '*', markersize = 0.5) 
#         plt.xlabel('x')
#         plt.ylabel('v')
#         plt.xlim(0, GN * dx)
#         plt.ylim(-2.5, 2.5)
#         plt.title('phase space')
#         if step < 10:
#             plt.savefig('/Users/sandy1114/Desktop/ani/Phase_Space_0000'+str(step)+'.png', dpi=300)
#         elif step < 100: 
#             plt.savefig('/Users/sandy1114/Desktop/ani/Phase_Space_000'+str(step)+'.png', dpi=300)
#         elif step < 1000:
#             plt.savefig('/Users/sandy1114/Desktop/ani/Phase_Space_00'+str(step)+'.png', dpi=300)
#         elif step < 10000:
#             plt.savefig('/Users/sandy1114/Desktop/ani/Phase_Space_0'+str(step)+'.png', dpi=300)
#         plt.close()
    
#     for i in range(GN):
#         U_energy[step] = U_energy[step] + 0.5 * E_field[i] * E_field[i] * epsilon * dx

#     for i in range(PN):
#         K_energy[step] = K_energy[step] + 0.5 * m_e * v_electron[i] * v_electron[i] * weight
#         Momentum[step] = Momentum[step] + m_e * v_electron[i] * weight

#     print(step, t,  U_energy[step], K_energy[step], U_energy[step] + K_energy[step])
    
#     t = t + dt
#     step = step + 1
    
    
# # the following values back to real number (by multiply normolization values)

# #U_energy = U_energy * energy_nor
# #K_energy = K_energy * energy_nor
# #Momentum = Momentum * momentum_nor
# T_energy =  U_energy + K_energy
# #np.save('Momentum_t='+str(dt)+'_x='+str(dx), Momentum)
# #np.save('T_error_t='+str(dt)+'_x='+str(dx), (T_energy - T_energy[0])/T_energy[0])
# #np.save('U_energy_t='+str(dt)+'_x='+str(dx), U_energy)
# #np.save('K_energy_t='+str(dt)+'_x='+str(dx), K_energy)
# plt.plot(U_energy[0:int(t_max / dt)+1], label = 'field energy')
# plt.plot(K_energy[0:int(t_max / dt)+1], label = 'kinetic energy')
# plt.plot(T_energy[0:int(t_max / dt)+1], label = 'total energy')
# #plt.plot(Momentum[0:int(t_max / dt)+1], label = 'momentum')
# #plt.yscale("log")#,basey=2)
# plt.xlabel('time')
# plt.ylabel('energy')
# plt.savefig('/Users/sandy1114/Desktop/ani/energy.png')
# plt.legend()
# plt.close()

# plt.plot(Momentum[0:int(t_max / dt)+1])
# plt.xlabel('time')
# plt.ylabel('energy')
# plt.savefig('/Users/sandy1114/Desktop/ani/momentum.png')
# plt.close()