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

# primeira derivada

from gapp import dgp


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
#np.savetxt('fz_recon_gapp.csv', np.transpose(F), delimiter=', ')



# MODELO LCDM


cosmo = ccl.Cosmology(
    Omega_c=0.2656, Omega_b=0.0494, h=0.6727, sigma8=0.8120, n_s=0.9649,
    matter_power_spectrum='linear')


zlcdm = np.linspace(0.001, 1.0, 1000)

a = 1. / (1. + zlcdm)

flcdm = ccl.background.growth_rate(cosmo, a)



# Plot the function, the prediction and the 95% confidence interval 
plt.figure()
plt.tick_params(labelsize=14, color='purple')
plt.errorbar(x_gapp, y_gapp, e, fmt='r.', color='red', markersize=10, label='Data')
plt.plot(xi, y_pred, color = 'navy', label='Prediction', linestyle="dotted", linewidth = 2)
plt.plot(zlcdm, flcdm, label='$\Lambda$CDM', color='blue', linestyle='--', linewidth = 3)
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



