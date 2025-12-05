#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 10:17:46 2024

@author: usuario
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy import loadtxt, savetxt
import pyccl as ccl
plt.rcParams['text.usetex'] = True
from gapp import gp
from scipy.integrate import solve_ivp

# Parâmetros:
H0 = 67.4
O_m0 = 0.315
O_r0 = 0
k = 0.125
c2 = 1  # Valor de exemplo para c2
n2 = 1     # Valor de exemplo para n
n3 = 2
O_L0 = 0.7
w0 = -0.957
wa = -0.29
O_k0 = -0.056 
b = 2.18
mu = 10**(-7)
c = 299792.458

p = 0.17 #0.049787068
t = np.linspace(p, 1, 1000)
zlcdm = (1/t) - 1


data_fz = np.genfromtxt('/home/usuario/Documentos/Dados/fz_data.csv', delimiter=', ')

z = data_fz[:,0]
fz = data_fz[:,1]
sig_fz = data_fz[:,2]


# nomeando
x_gapp = z
y_gapp = fz
e = sig_fz

# xmin, xmax and nstar are interpreted as two-dimensional vectors
xmin = 0
xmax = 1.0
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
#np.savetxt('fz_recon_gapp_novo.csv', np.transpose(F), delimiter=', ')



# MODELO LCDM


def fgRG(t, y):
    fg_RG  = y[0]

    O_m = O_m0*t**(-3)
    O_r = O_r0*t**(-4)
    O_L = 1 - O_m0

    H_RG = H0*np.sqrt(O_m + O_r + O_L)
    dH_RG = - (H_RG/t) - 0.5*(H0/t)*(H0/H_RG)*(O_m + 2*O_r - 2*O_L)

    faux1 = 3*H0**2
    faux2 = 2*(t**2)*(H_RG**2)

    dfg_RG = - ((dH_RG/H_RG) + ((2+fg_RG)/t))*fg_RG + (3/2)*((H0/H_RG)**2)*(O_m/t)  # equação diferencial para o f
    return dfg_RG


# Integração:
t_span =[p, 1]
t = np.linspace(p, 1, 1000)


# Condições iniciais:
ti = p
Hi = H0*np.sqrt(O_m0*ti**(-3) + (1 - O_m0))
O_mi = O_m0*ti**(-3)
O_ri = O_r0*ti**(-4)
O_L = 1 - O_m0
fgi = ((H0/Hi)**2 * O_mi)**(6/11)
y0 = [fgi]


# Solução RG:
solRG = solve_ivp(fgRG, t_span, y0, t_eval=t, method='LSODA')
fg_RG = solRG.y[0]


# definindo o redshift
z = 1/solRG.t - 1



# Plot the function, the prediction and the 95% confidence interval 
plt.figure()
plt.tick_params(labelsize=14, color='purple')

plt.plot(zlcdm, fg_RG, color='blue', linewidth = 3, linestyle='--', label='$\Lambda$CDM')
plt.plot(xi, y_pred, color = 'navy', label='GP', linestyle="dotted", linewidth = 2)
plt.errorbar(x_gapp, y_gapp, e, fmt='r.', color='red', markersize=10, label='Data')


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
plt.xlabel('$z$', fontsize=15)
plt.ylabel('$f(z)$', fontsize=15)
plt.legend(loc='upper left')
#plt.savefig('fz_reconstruction.pdf', format='pdf', bbox_inches='tight')
plt.show()



