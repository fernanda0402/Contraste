#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 21 15:05:23 2025

@author: usuario
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy import loadtxt, savetxt
from gapp import gp
plt.rcParams['text.usetex'] = True


data = np.genfromtxt('/home/usuario/Documentos/Dados/dlc_snia.dat', delimiter='\t')

zCMB = data[:, 0]
dl = data[:, 1]
dlerr = data[:, 2]



# removing repeated values of zCMB
unique_zCMB, unique_indices = np.unique(zCMB, return_index=True)

# Filters the data to keep only rows matching unique zCMB values
unique_dl = dl[unique_indices]
unique_dlerr = dlerr[unique_indices]

print(len(unique_zCMB))
print(len(unique_dl))
print(len(unique_dlerr))


# constants
c = 2.9*10**5 # velocidade da luz
h0 = 70 #km/s/Mpc


######## GAUSSIAN PROCESS #################

# naming
x_gapp = unique_zCMB
y_gapp = unique_dl
e = unique_dlerr


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



# saving the reconstructed data

N =  xi, y_pred, sigma

#np.savetxt('dlz_recon_gapp_novo.csv', np.transpose(N), delimiter=', ')


# Plot the function, the prediction and the 95% confidence interval
plt.figure()
plt.tick_params(labelsize=14, color='purple')
plt.errorbar(x_gapp, y_gapp, e, fmt='r.', color='purple', markersize=10, label='Data')
plt.plot(xi, y_pred, color = 'green', label='Prediction', linestyle="--")
plt.fill(np.concatenate([xi, xi[::-1]]),
         np.concatenate([y_pred - 1.9600 * sigma,
                        (y_pred + 1.9600 * sigma)[::-1]]),
         alpha=.5, color = 'lightgreen', ec='None')

# legenda, label e título
plt.xlabel('$z$', fontsize=15)
plt.ylabel('$D_L(z)$', fontsize=15)
plt.legend(loc='best')
#plt.savefig('dl_recon.pdf', format='pdf', bbox_inches='tight')
plt.show()



