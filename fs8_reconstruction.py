#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 11:10:28 2024

@author: usuario
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy import loadtxt, savetxt
import pyccl as ccl
plt.rcParams['text.usetex'] = True
from gapp import gp
from scipy.special import gamma
from numpy import exp
from scipy.integrate import solve_ivp


# BAIXANDO O ARQUIVO DE fs8

data = np.genfromtxt('/home/usuario/Documentos/Códigos/GaPP SNIa/fsig8_bold_data.dat')

z_gapp = data[:, 0]
fs8 = data[:, 1]
sig_fs8 = data[:, 2]




# DEFININDO INVGAMMA
def invgamma(x, a, b):
    x = x[1]
    p = b**a/gamma(a) * x**(-1 - a) * exp(-b/x)

    return p

# nomeando
x_gapp = z_gapp[z_gapp<2]
y_gapp = fs8[z_gapp<2]
e = sig_fs8[z_gapp<2]

# xmin, xmax and nstar are interpreted as two-dimensional vectors
xmin = 0
xmax = 5.0
nstar = 1000

# initial values of the hyperparameters of the squared-exponential covariance function
initheta = [0.5, 0.3]

# initialization of the Gaussian Process
#g = gp.GaussianProcess(x_gapp, y_gapp, e, cXstar=(xmin, xmax, nstar))
g = gp.GaussianProcess(x_gapp,y_gapp,e,cXstar=(xmin, xmax, nstar),
                        prior=invgamma, priorargs=(4, 1.5),
                        grad='False')

# training of the hyperparameters and reconstruction of the function
(rec, theta) = g.gp(theta=initheta)

xi = rec[:, 0]

y_pred = rec[:, 1]
sigma  = rec[:, 2]

y_pred_95_less = y_pred - 1.9600*sigma
y_pred_95_plus = y_pred + 1.9600*sigma


######################################################################################

O_r0 = 0
O_L0 = 0.7
mu = 10**(-11)
k = 0.125
w0 = -0.957
wa = -0.29
O_k0 = -0.056

p = 0.17 #0.049787068
t = np.linspace(p, 1, 1000)
z = (1/t) - 1

# MODELO LCDM

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




# Plot the function, the prediction and the 95% confidence interval 
plt.figure()
plt.tick_params(labelsize=14, color='purple')
plt.plot(z, solfs8L(70, 0.3, 0.8), color='blue', linewidth = 2.5, label='$\Lambda$CDM', linestyle = '--')
plt.errorbar(x_gapp, y_gapp, e, fmt='o', markersize=5, color='red', label='Data')
plt.plot(xi, y_pred, color = 'navy', label='GP', linestyle="dotted")
plt.fill(np.concatenate([xi, xi[::-1]]),
         np.concatenate([y_pred - 1.9600 * sigma,
                        (y_pred + 1.9600 * sigma)[::-1]]),
         alpha=.5, color = 'lightblue', ec='None')
plt.fill(np.concatenate([xi, xi[::-1]]),
         np.concatenate([y_pred - 1.00 * sigma,
                        (y_pred + 1.00 * sigma)[::-1]]),
         alpha=.5, color = 'dodgerblue', ec='None')

# legenda, label e título
plt.xlim(0,1.0)
plt.ylim(0.15,0.65)
plt.xlabel('$z$', fontsize=15)
plt.ylabel('$f\sigma_8(z)$', fontsize=15)
plt.legend(loc='lower left')
#plt.savefig('fs8_reconstruction.pdf', format='pdf', bbox_inches='tight')
plt.show()

















