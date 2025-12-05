#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 14:33:40 2025

@author: usuario
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy import loadtxt, savetxt
plt.rcParams['text.usetex'] = True
from gapp import gp
from scipy.integrate import solve_ivp
from scipy import integrate

plt.rcParams['text.usetex'] = True


# baixando os dados
data_Hz = np.genfromtxt('/home/usuario/Documentos/Dados/CC_Hz_data (cópia).csv', delimiter=', ')

z_gapp = data_Hz[:, 0]
H = data_Hz[:, 1]

sig_H = data_Hz[:, 2]

#print(max(z_gapp))


# definindo constantes
c = 2.9*10**5 # velocidade da luz
h0 = 67.27 #km/s/Mpc

##################### PROCESSO GAUSSIANO GAPP ###########################

# nomeando
x_gapp = z_gapp
y_gapp = H
e = sig_H

# xmin, xmax and nstar are interpreted as two-dimensional vectors
xmin = 0
xmax = 5.0
nstar = 1000

# initial values of the hyperparameters of the squared-exponential covariance function
initheta = [2.0, 2.0]

# initialization of the Gaussian Process
g = gp.GaussianProcess(x_gapp, y_gapp, e, cXstar=(xmin, xmax, nstar))

# training of the hyperparameters and reconstruction of the function
(rec, theta) = g.gp(theta=initheta)

xi = rec[:, 0]

y_pred = rec[:, 1]
sigma  = rec[:, 2]

y_pred_95_less = y_pred - 1.9600*sigma
y_pred_95_plus = y_pred + 1.9600*sigma


# salvando os dados reconstruídos

F = xi, y_pred, sigma
#np.savetxt('Hz_recon_gapp_new.csv', np.transpose(F), delimiter=', ')



ti = 0.17
t = np.linspace(ti, 1, 1000)
t_span = [ti, 1]
z = (1/t) - 1
O_r0 = 0

# Modelo LCDM:
def H_LCDM(H0, O_m0):    # t é o fator de escala e y é o delta (contraste)

    O_m = O_m0*t**(-3)  
    O_L = 1 - O_m0

    H_RG = H0*np.sqrt(O_m + O_L)    # H(z)
    
    return H_RG


G1 = z, H_LCDM(67.2, 0.339)
#np.savetxt('Hz_lcdm_mcmc.csv', np.transpose(G1), delimiter=', ')



# Modelo wCDM
def H_wCDM(H0, O_m0, w):
  O_m = O_m0*t**(-3)
  O_L0 = 1 - O_m0
  O_L = O_L0*t**(-3*(1 + w))
  H_wCDM = H0*np.sqrt(O_m + O_L)
  return H_wCDM


G2 = z, H_wCDM(67.3, 0.306, -0.887)
#np.savetxt('Hz_wcdm_mcmc.csv', np.transpose(G2), delimiter=', ')




# w0waCDM model:
def H_w0wa(H0, O_m0, w0, wa):
  O_m = O_m0*t**(-3)
  O_L0 = 1 - O_m0
  #w_aux = 1 + (w0 + wa*(1 - t))
  O_L = O_L0*np.exp(-3*wa*(1-t))*(t**(-3*(1 + w0 + wa) ) )
  H_w0wa = H0*np.sqrt(O_m + O_L)
  return H_w0wa


G3 = z, H_w0wa(67.4, 0.320, -0.876, -0.07)
#np.savetxt('Hz_w0wacdm_mcmc.csv', np.transpose(G3), delimiter=', ')



# H(t) O_KCDM model:
def H_OKCDM(H0, O_m0, O_K0):
  O_K = O_K0*t**(-2)
  O_m = O_m0*t**(-3)
  O_L = 1 - O_m0 
  H_OK = H0*np.sqrt(O_K + O_m + O_L)
  return H_OK


G4 = z, H_OKCDM(61.3, 0.317, 0.20)
#np.savetxt('Hz_okcdm_mcmc.csv', np.transpose(G4), delimiter=', ')



Delta = 10**(-7)
# H(t) para o modelo R2-AB:
def Hubble_AB(t, y, H0, O_m0, b):
  H  =  y[0]
  dH  = y[1]
  ddH = y[2]

  O_m = O_m0*t**(-3)
  O_r = O_r0*t**(-4)

  R_vac = 12*H0**2
  e_AB = R_vac/np.log(1 + np.exp(2*b))
  M2 = e_AB/Delta

  R = 6*H*(2*H + t*dH)
  alpha = (R/e_AB) - b
  if abs(alpha) < 15:
    s = 1/np.cosh(alpha)
  else:
    s = 0

  A = H0*H0*(3*O_m + 4*O_r)
  B = t*H*dH*(np.tanh(alpha) + 1)
  C = 4*t*H*H*dH*(t*dH + 2*H)/M2
  if abs(alpha) < 15:
    D = ((6*t*t*H*H)/e_AB)*(((5*dH/t) + (dH*dH/H) +ddH)**2)*np.tanh(alpha)*s*s
    E = 6*(t**3)*(H**3)*((1/(3*M2)) + (0.5*s*s/e_AB))
  else:
    D = 0
    E = 6*(t**3)*(H**3)/(3*M2)

  F = (11*dH*dH/(t*H)) + (dH*dH*dH/(H*H)) + (6*ddH/t) + (4*dH*ddH/H)

  dddH = - ((A + B + C - D)/E) - F
  return [dH, ddH, dddH]

def solH_AB(H0, O_m0, b):
  t_span = [0.17, 1]
  ti = 0.17
  O_mi = O_m0*ti**(-3)
  O_ri = O_r0*ti**(-4)
  O_L = 1 - O_m0
  Hi = H0*np.sqrt(O_mi + O_ri + O_L)
  dHi = - ((H0*H0)/(2*ti*Hi))*(3*O_mi + 4*O_ri)
  ddHi = 0.5*(H0/(ti*Hi))**2*(Hi + ti*dHi)*(3*O_mi + 4*O_ri) + 0.5*(H0/(ti*Hi))**2*Hi*(9*O_mi + 16*O_ri)
  y0 = [Hi, dHi, ddHi]

  sol = solve_ivp(Hubble_AB, t_span, y0, t_eval=t, method='LSODA', rtol = 10**(-6), args=(H0, O_m0, b))
  H = sol.y[0]
  return H



G5 = z, solH_AB(67.3, 0.332, 1.98)
#np.savetxt('Hz_ab_mcmc.csv', np.transpose(G5), delimiter=', ')




# FQ1

def H_Q3(H0, O_m0):
    O_m = O_m0*t**(-3)
    O_L = 1 - O_m0 - O_r0
    H_Q3 = H0*np.sqrt(O_m + O_L)
    return H_Q3


G6 = z, H_Q3(66.9, 0.347)
#np.savetxt('Hz_fq3_mcmc.csv', np.transpose(G6), delimiter=', ')





# Modelo HS-f(R):
def yH_HS1(t, y, H0, O_m0, mu):
    yH = y[0]
    YH = y[1]        # YH = dyH/da
    
    n = 1

    O_m = O_m0*t**(-3)
    O_r = O_r0*t**(-4)

    Lbd = 3*(H0**2)*(1 - O_m0)

    chi = O_r0/O_m0
    ms = (H0**2)*O_m0

    R = 3*ms*(4*yH + t*YH + t**(-3))

    A = (R**n) + (mu**(2*n))
    B = (R**n)
    C = (n+1)*(R**(2*n - 2)) - (n- 1)*(mu**(2*n))*(R**(n-2))
    D = A**3

    f = R - 2*Lbd*(B/A)
    fR = 1 - 2*n*Lbd*(mu**(2*n))*(B/A)*(1/(R*A))
    fRR = 2*n*Lbd*(mu**(2*n))*(C/D)

    yaux1 = yH + t**(-3) + chi*t**(-4)
    yaux2 = t**(-3) + 2*chi*t**(-4)

    j1 = 4 + (1/yaux1)*(1-fR)/(6*ms*fRR)
    j2 = (1/yaux1)*(2-fR)/(3*ms*fRR)
    j3 = - 3*t**(-3) - (((1-fR)*yaux2 + (R-f)/(3*ms))/yaux1)*(1/(6*ms*fRR))
    J1 = (1/t)*(1 + j1)
    J2 = (1/t)*(j2/t)
    J3 = (1/t)*(j3/t)

    dYH = - J1*YH - J2*yH - J3
    return [YH, dYH]

def solyH_HS1(H0, O_m0, mu):
    t_span = [0.17, 1]
    y0 = [(1-O_m0)/O_m0, 0]
    sol = solve_ivp(yH_HS1, t_span, y0, t_eval=t, method='LSODA', rtol = 10**(-6), args=(H0, O_m0, mu))
    return sol.y[0]

def H_HS1(H0, O_m0, mu):
    O_r0 = 0
    chi = O_r0/O_m0
    ms = (H0**2)*O_m0
    yH = solyH_HS1(H0, O_m0, mu)
    H = np.sqrt(ms*(yH + (1/t)**3 + chi*(1/t)**4))
    return H



G7 = z, H_HS1(69.7, 0.269, 82)
#np.savetxt('Hz_hs1_mcmc.csv', np.transpose(G7), delimiter=', ')




# Modelo HS-f(R) n=2:
def yH_HS2(t, y, H0, O_m0, mu):
    yH = y[0]
    YH = y[1]        # YH = dyH/da
    
    n=2

    O_m = O_m0*t**(-3)
    O_r = O_r0*t**(-4)

    Lbd = 3*(H0**2)*(1 - O_m0)

    chi = O_r0/O_m0
    ms = (H0**2)*O_m0

    R = 3*ms*(4*yH + t*YH + t**(-3))

    A = (R**n) + (mu**(2*n))
    B = (R**n)
    C = (n+1)*(R**(2*n - 2)) - (n- 1)*(mu**(2*n))*(R**(n-2))
    D = A**3

    R0 = ms*((12/O_m0) - 9)
    M2 = R0/Delta

    f = R - 2*Lbd*(B/A) + (R**2/(6*M2))
    fR = 1 - 2*n*Lbd*(mu**(2*n))*(B/A)*(1/(R*A)) + (R/(3*M2))
    fRR = 2*n*Lbd*(mu**(2*n))*(C/D) + (1/(3*M2))

    yaux1 = yH + t**(-3) + chi*t**(-4)
    yaux2 = t**(-3) + 2*chi*t**(-4)

    j1 = 4 + (1/yaux1)*(1-fR)/(6*ms*fRR)
    j2 = (1/yaux1)*(2-fR)/(3*ms*fRR)
    j3 = - 3*t**(-3) - (((1-fR)*yaux2 + (R-f)/(3*ms))/yaux1)*(1/(6*ms*fRR))
    J1 = (1/t)*(1 + j1)
    J2 = (1/t)*(j2/t)
    J3 = (1/t)*(j3/t)

    dYH = - J1*YH - J2*yH - J3
    return [YH, dYH]

def solyH_HS2(H0, O_m0, mu):
    t_span = [0.17, 1]
    y0 = [(1-O_m0)/O_m0, 0]
    sol = solve_ivp(yH_HS2, t_span, y0, t_eval=t, method='LSODA', rtol = 10**(-6), args=(H0, O_m0, mu))
    return sol.y[0]

def H_HS2(H0, O_m0, mu):
  O_r0 = 0
  chi = O_r0/O_m0
  ms = (H0**2)*O_m0
  yH = solyH_HS2(H0, O_m0, mu)
  H = np.sqrt(ms*(yH + (1/t)**3 + chi*(1/t)**4))
  return H


G8 = z, H_HS2(66.8, 0.327, 104)
#np.savetxt('Hz_hs2_mcmc.csv', np.transpose(G8), delimiter=', ')



# Starobinsky n=1

def yH_S1(t, y, H0, O_m0, mu):
    yH = y[0]
    YH = y[1]

    O_m = O_m0*t**(-3)
    O_r = O_r0*t**(-4)
    chi = O_r0/O_m0
    ms = (H0**2)*O_m0
    R = 3*ms*(4*yH + t*YH + t**(-3))

    Lbd = 3 * H0**2 * (1 - O_m0)
    Rs = 2*Lbd*mu
    star = 1 + (R/Rs)**2

    lmu2 = Lbd * mu**2

    f = R - 2*Lbd*(1 - (star**(-1)))
    fR = 1 - (R/lmu2)*(star**(-2))
    fRR = (1/lmu2)*(4 * (R/Rs)**2 * star**(-1) - 1) * star**(-2)

    yaux1 = yH + t**(-3) + chi*t**(-4)
    yaux2 = t**(-3) + 2*chi*t**(-4)
    j1 = 4 + (1/yaux1)*(1-fR)/(6*ms*fRR)
    j2 = (1/yaux1)*(2-fR)/(3*ms*fRR)
    j3 = - 3*t**(-3) - (((1-fR)*yaux2 + (R-f)/(3*ms))/yaux1)*(1/(6*ms*fRR))
    J1 = (1/t)*(1 + j1)
    J2 = (1/t)*(j2/t)
    J3 = (1/t)*(j3/t)
    dYH = - J1*YH - J2*yH - J3
    return [YH, dYH]

def solyH_S1(H0, O_m0, mu):
    y0 = [(1-O_m0)/O_m0, 0]
    sol_yH_S = solve_ivp(yH_S1, t_span, y0, t_eval=t, method='LSODA', rtol = 10**(-10), args=(H0, O_m0, mu))
    yH = sol_yH_S.y[0]
    return yH

def H_S1(H0, O_m0, mu):
    yH = solyH_S1(H0, O_m0, mu)
    chi = O_r0/O_m0
    ms = H0**2 * O_m0
    H_S = np.sqrt(ms*(yH + (1/t)**3 + chi*(1/t)**4))
    return H_S


G9 = z, H_S1(67.1, 0.328, 0.589)
#np.savetxt('Hz_s1_mcmc.csv', np.transpose(G9), delimiter=', ')



#Starobinsky n=2:

def yH_S2(t, y, H0, O_m0, mu):
    yH = y[0]
    YH = y[1]

    O_m = O_m0*t**(-3)
    O_r = O_r0*t**(-4)
    chi = O_r0/O_m0
    ms = (H0**2)*O_m0
    R = 3*ms*(4*yH + t*YH + t**(-3))

    Lbd = 3 * H0**2 * (1 - O_m0)
    Rs = 2*Lbd*mu
    star = 1 + (R/(2*Lbd*mu))**2

    M2 = Rs/Delta

    lmu2 = Lbd * (mu**2)
    star2 = (2*Lbd*mu)**2 + R**2

    f = R - 2*Lbd*(1 - (star**(-2))) + ((R**2)/(6*M2))
    fR = 1 - (2*R/lmu2)*(star**(-3)) + (R/(3*M2))
    fRR = ((2**7)*(Lbd**5)*(mu**4)/(star2**3))*(6*((R**2)/star2)- 1) + (1/(3*M2))

    yaux1 = yH + t**(-3) + chi*t**(-4)
    yaux2 = t**(-3) + 2*chi*t**(-4)
    j1 = 4 + (1/yaux1)*(1-fR)/(6*ms*fRR)
    j2 = (1/yaux1)*(2-fR)/(3*ms*fRR)
    j3 = - 3*t**(-3) - (((1-fR)*yaux2 + (R-f)/(3*ms))/yaux1)*(1/(6*ms*fRR))
    J1 = (1/t)*(1 + j1)
    J2 = (1/t)*(j2/t)
    J3 = (1/t)*(j3/t)
    dYH = - J1*YH - J2*yH - J3
    return [YH, dYH]

def solyH_S2(H0, O_m0, mu):
    y0 = [(1-O_m0)/O_m0, 0]
    sol_yH_S = solve_ivp(yH_S2, t_span, y0, t_eval=t, method='LSODA', rtol = 10**(-10), args=(H0, O_m0, mu))
    yH = sol_yH_S.y[0]
    return yH

def H_S2(H0, O_m0, mu):
    yH = solyH_S2(H0, O_m0, mu)
    chi = O_r0/O_m0
    ms = H0**2 * O_m0
    H_S = np.sqrt(ms*(yH + (1/t)**3 + chi*(1/t)**4))
    return H_S


G10 = z, H_S2(66.4, 0.350, 1.047)
#np.savetxt('Hz_s2_mcmc.csv', np.transpose(G10), delimiter=', ')



# Plot the function, the prediction and the 95% confidence interval 
plt.figure()
plt.ylim(0, 180)
plt.tick_params(labelsize=14, color='purple')

#plt.errorbar(x_gapp, y_gapp, e, fmt='r.', color='red', markersize=10, label='Data')

plt.plot(z, H_LCDM(67.2, 0.339), color='blue', linewidth = 1, linestyle='--', label='$\Lambda$CDM')
plt.plot(z, H_wCDM(67.3, 0.306, -0.887), color='red', linewidth = 1, label='$\omega$CDM')
plt.plot(z, H_w0wa(67.4, 0.320, -0.876, -0.07), color='saddlebrown', linewidth = 1, label='$\omega_0 \omega_a$CDM')
plt.plot(z, H_OKCDM(61.3, 0.317, 0.20), color='green', linewidth = 1, label='$\Omega_k$-CDM')


#plt.plot(z, H_S1(67.1, 0.328, 0.589), color='orange', linewidth=1, label='Starobinski (n=1)', linestyle = '-.')
#plt.plot(z, H_S2(66.4, 0.350, 1.047), color='deeppink', linewidth=1, label='Starobinski (n=2)', linestyle = '-.')
#plt.plot(z, H_HS1(69.7, 0.269, 82), color='purple', linewidth=1, label='Hu-Sawicki (n=1)')
#plt.plot(z, H_HS2(66.8, 0.327, 104), color='magenta', linewidth=1, label='Hu-Sawicki (n=2)')
#plt.plot(z, solH_AB(67.3, 0.332, 1.98), color='black', linewidth = 1, label='$R^2$_AB')

#plt.plot(z, H_Q3(66.9, 0.347), color='maroon', label='$F(Q)$', linewidth = 1)


plt.plot(xi, y_pred, color = 'darkblue', linewidth=2, label='GP', linestyle="dotted")


plt.plot()
plt.fill(np.concatenate([xi, xi[::-1]]),
         np.concatenate([y_pred - 1.9600 * sigma,
                        (y_pred + 1.9600 * sigma)[::-1]]),
         alpha=.5, color = 'lightblue', ec='None')
plt.fill(np.concatenate([xi, xi[::-1]]),
         np.concatenate([y_pred - 1.00 * sigma,
                        (y_pred + 1.00 * sigma)[::-1]]),
         alpha=.5, color = 'dodgerblue', ec='None')


# legenda, label e título
plt.ylim(30,250)
plt.xlim(0,2)
plt.xlabel('$z$', fontsize=15)
plt.ylabel('$H(z)$', fontsize=15)
plt.legend(loc='best')
plt.savefig('Hz_comparison_RG.pdf', format='pdf', bbox_inches='tight')
plt.show()


