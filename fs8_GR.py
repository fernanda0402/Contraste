#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 10:25:29 2024

@author: usuario
"""

# Bibliotecas:
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.integrate import cumtrapz

# Parâmetros:

O_r0 = 0
O_L0 = 0.7
w0 = -0.957
wa = -0.29
O_k0 = -0.056

p = 0.17 #0.049787068
t = np.linspace(p, 1, 1000)
z = (1/t) - 1




# Modelo LCDM:
def Den_RG(t, y, H0, O_m0):
    D_RG  = y[0]
    dD_RG = y[1]

    O_m = O_m0*t**(-3)
    O_r = O_r0*t**(-4)
    O_L = 1 - O_m0

    H_RG = H0*np.sqrt(O_m + O_r + O_L)
    dH_RG = - (H_RG/t) - 0.5*(H0/t)*(H0/H_RG)*(O_m + 2*O_r - 2*O_L)

    faux1 = 3*H0**2
    faux2 = 2*(t**2)*(H_RG**2)

    ddD_RG = - ((3/t) + (dH_RG/H_RG))*dD_RG + (faux1/faux2)*O_m*D_RG
    return [dD_RG, ddD_RG]

def solD(H0, O_m0):
    # Espaço de integração:
    t_span = [p, 1]

    # Condições iniciais:
    y0 = [p, 1]

    # Solução:
    sol = solve_ivp(Den_RG, t_span, y0, t_eval=t, method='LSODA', args=(H0, O_m0))
    D = sol.y[0]
    return D

def solfs8L(H0, O_m0, sig8):
    # Espaço de integração:
    t_span = [p, 1]

    # Condições iniciais:
    y0 = [p, 1]

    # Solução:
    sol = solve_ivp(Den_RG, t_span, y0, t_eval=t, method='LSODA', args=(H0, O_m0))
    dD_RG = sol.y[1]
    fs8L = sig8*sol.t*(dD_RG/solD(H0, O_m0)[999])
    return fs8L
#print(solfs8L(67.4, 0.315, 0.811))



#####################################################################################



# Modelo wCDM
def Densidade_w1(t, y, H0, O_m0):    # t é o fator de escala e y é o delta (contraste)
    D_RG_w1  = y[0]     # contraste
    dD_RG_w1 = y[1]    # primeira derivada do contraste

    O_m = O_m0*t**(-3)
    O_L = O_L0*(t**(-3*(1+w0) ) )

    H_RG_w1 = H0*np.sqrt(O_m + O_L)    # H(z)
    dH_RG_w1 = - (H_RG_w1/t) - 0.5*(H0/t)*(H0/H_RG_w1)*(O_m - 2*O_L)        # derivada do H(z)

# funções definidas somente para facilitar a escrita da derivada segunda do contraste
    faux1 = 3*H0**2
    faux2 = 2*(t**2)*(H_RG_w1**2)

    ddD_RG_w1 = - ((3/t) + (dH_RG_w1/H_RG_w1))*dD_RG_w1 + (faux1/faux2)*O_m*D_RG_w1   #  eq. da derivada segunda do contraste
    return [dD_RG_w1, ddD_RG_w1]

def solD_w1(H0, O_m0):
    # Espaço de integração:
    t_span = [p, 1]

    # Condições iniciais:
    y0 = [p, 1]

    # Solução:
    sol_w1 = solve_ivp(Densidade_w1, t_span, y0, t_eval=t, method='LSODA', args=(H0, O_m0))
    D_w1 = sol_w1.y[0]
    return D_w1

def solfs8L_w1(H0, O_m0, sig8):
    # Espaço de integração:
    t_span = [p, 1]

    # Condições iniciais:
    y0 = [p, 1]

    # Solução:
    sol_w1 = solve_ivp(Densidade_w1, t_span, y0, t_eval=t, method='LSODA', args=(H0, O_m0))
    dD_w1 = sol_w1.y[1]
    fs8L_w1 = sig8*sol_w1.t*(dD_w1/solD_w1(H0, O_m0)[999])
    return fs8L_w1



###################################################################################33

# Modelo w0waCDM

def Densidade_w(t, y, H0, O_m0): # t é o fator de escala e y é o delta (contraste)
   
    D_RG_w  = y[0]     # contraste
    dD_RG_w = y[1]    # primeira derivada do contraste

    O_m = O_m0*t**(-3)
    O_L = O_L0*(t**(-3*(1 + w0 + wa*(1-t)) ) )

    H_RG_w = H0*np.sqrt(O_m + O_L)    # H(z)
    dH_RG_w = - (H_RG_w/t) - 0.5*(H0/t)*(H0/H_RG_w)*(O_m - 2*O_L)        # derivada do H(z)

# funções definidas somente para facilitar a escrita da derivada segunda do contraste
    faux1 = 3*H0**2
    faux2 = 2*(t**2)*(H_RG_w**2)

    ddD_RG_w = - ((3/t) + (dH_RG_w/H_RG_w))*dD_RG_w + (faux1/faux2)*O_m*D_RG_w   #  eq. da derivada segunda do contraste
    return [dD_RG_w, ddD_RG_w]

def solD_w(H0, O_m0):
    # Espaço de integração:
    t_span = [p, 1]

    # Condições iniciais:
    y0 = [p, 1]

    # Solução:
    sol_w = solve_ivp(Densidade_w, t_span, y0, t_eval=t, method='LSODA', args=(H0, O_m0))
    D_w = sol_w.y[0]
    return D_w

def solfs8L_w(H0, O_m0, sig8):
    # Espaço de integração:
    t_span = [p, 1]

    # Condições iniciais:
    y0 = [p, 1]

    # Solução:
    sol_w = solve_ivp(Densidade_w, t_span, y0, t_eval=t, method='LSODA', args=(H0, O_m0))
    dD_w = sol_w.y[1]
    fs8L_w = sig8*sol_w.t*(dD_w/solD_w(H0, O_m0)[999])
    return fs8L_w



#####################################################################################

# Modelo Ok-CDM - arxiv 0903.0001

t = np.linspace(p, 1, 1000)
z = (1/t) - 1

def growth(O_m0, O_K0):
    
  w = -1
  
  O_L = 1 - O_m0 - O_K0
  O_K = O_K0*t / ( O_m0 + O_K0*t + O_L*(t**(-3*w)) )
  O_M = O_m0/( O_m0 + O_K0*t + O_L*(t**(-3*w)) )
  gamma = 0.55
  f_K = O_M**(gamma) + (gamma - 4/7)*O_K
  return f_K


def ln_delta(O_m0, O_K0):
  y = growth(O_m0, O_K0)/t
  ln_delta = cumtrapz(y, t, initial=0)
  return ln_delta

def delta(O_m0, O_K0):
  delta = np.exp(ln_delta(O_m0, O_K0))
  delta = delta/delta[999]
  return delta

def fs8(O_m0, O_K0, sig80):
      fs8 = sig80*growth(O_m0, O_K0)*delta(O_m0, O_K0)
      return fs8



#######################################################################################


plt.plot(z, solfs8L(70, 0.3, 0.8), color='blue', linewidth = 2.5, label='$\Lambda$CDM', linestyle = '--')
plt.plot(z, solfs8L_w1(70, 0.3, 0.8), color='red', linewidth = 2, label='$\omega$CDM')
plt.plot(z, solfs8L_w(70, 0.3, 0.8), color='darkgoldenrod', linewidth = 2, label='$\omega_0 \omega_a$CDM')
plt.plot(z, fs8(0.3, -0.056, 0.811), color='green', label='$\Omega_k$-CDM')

plt.legend()
plt.xlabel('$z$')
plt.ylabel('$[f\sigma_8](z)$')
#plt.xlim(-0.1, 2)
#plt.ylim(0.315, 0.47)
#plt.savefig('fs8_GR.pdf', dpi=520, format='pdf', bbox_inches='tight')
plt.show()


