from __future__ import absolute_import, division, print_function, unicode_literals

import os
import sys
sys.path.append("..")
import math
import numpy as np
import scipy
import matplotlib.pyplot as plt
from time import process_time
import torch

from truebayes import network
from truebayes import geometry
from truebayes import roman
from truebayes import loss
from truebayes import like
from truebayes import plot

print(sys.path)

Net_roman_G2 = network.makenet([241*2] + [1024] * 8 + [1*6], softmax=False)

nrg2 = Net_roman_G2()
print('nrg2', nrg2)

print('training started (', process_time(), ')')

trainingset = lambda: roman.syntrain(snr=[8,16], size=100000, varx=['Mc','nu'], region=[[0.26,0.47], [0.2,0.25]])
roman.syntrainer(nrg2, trainingset, lossfunction=loss.kllossGn2, iterations=100, initstep=1e-4, finalv=1e-8)

print('training end (', process_time(), ')')

nrg2.load_state_dict(torch.load(roman.datadir + 'percival/Mc-nu_l1024x8_g1_SNR8-16_2d.pt', map_location=torch.device('cpu')))
nrg2.eval()

print('training started (', process_time(), ')')
mutest = roman.syntrain(snr=[8,16], size=5000, varx='Mc', region=[[0.26,0.47], [0.2,0.25]], noise=1, varall=True, seed=2)
print('training started (', process_time(), ')')

Mc = mutest[0][:,0]
nu = mutest[0][:,1]
snr = mutest[0][:,4]

# we will be sorting our plots by chirp mass
idx = np.argsort(Mc)

print('training started (', process_time(), ')')
likeset_Mc = lambda: roman.syntrain(snr=[8,16], size=100000, varx='Mc', region=[[0.26,0.47], [0.2,0.25]], noise=0)
likeset_nu = lambda: roman.syntrain(snr=[8,16], size=100000, varx='nu', region=[[0.26,0.47], [0.2,0.25]], noise=0)
likeset_2 = lambda: roman.syntrain(snr=[8,16], size=100000, varx=['Mc','nu'], region=[[0.26,0.47], [0.2,0.25]], noise=0)
print('training started (', process_time(), ')')

## Plot 1-D histograms from 1-D and 2-D networks
sl_Mc = like.synlike(mutest[2][:24,:], likeset_Mc, iterations=10000000)
sl_nu = like.synlike(mutest[2][:24,:], likeset_nu, iterations=10000000)

plot.plotgauss(*mutest, net=nrg2, varx='Mc', like=sl_Mc, twodim=True, istart=6)
plot.plotgauss(*mutest, net=nrg2, varx='nu', like=sl_nu, twodim=True, istart=6)

## Plot 2-D histogram from 2-D network
sl2 = like.synlike(mutest[2][:24,:], likeset_2, iterations=10000000)
plot.makecontour(*mutest, net=nrg2, like=sl2, istart=6)

## Compare 1D and 2D means and variances

lm, le = like.synmean(mutest[2], likeset_Mc, iterations=10000000)
lm_nu, le_nu  = like.synmean(mutest[2], likeset_nu, iterations=10000000)
lm2, le2, lc2 = like.synmean(mutest[2], likeset_2, iterations=10000000)
nm2, ne2, nc2 = loss.netmeanGn2(mutest[2], net=nrg2)

plt.figure(figsize=(12,5))

idx = np.argsort(Mc)

plt.subplot(1,2,1)
plt.plot(Mc[idx], nm2[idx,0]  - lm2[idx,0], 'x', label='net2D - like', alpha=0.2)
plt.xlabel('Mc')
plt.ylabel('delta Mc')
plt.legend()
plt.axis(ymin=-0.1,ymax=0.1)

plt.subplot(1,2,2)
plt.plot(Mc[idx], nm2[idx,1]  - lm2[idx,1], 'x', label='net2D - like', alpha=0.2)
plt.xlabel('Mc')
plt.ylabel('delta nu')
plt.legend()
plt.axis(ymin=-0.4,ymax=0.4)

plt.tight_layout()


plt.figure(figsize=(12,5))

idx = np.argsort(Mc)

plt.subplot(1,2,1)
plt.plot(Mc[idx], (ne2[idx,0]  - le2[idx,0])/le2[idx,0], 'x', label='net2D - like', alpha=0.2)
plt.xlabel('Mc')
plt.ylabel('frac delta stderr Mc')
plt.legend()
plt.axis(ymax=4)

plt.subplot(1,2,2)
plt.plot(Mc[idx], (ne2[idx,1]  - le2[idx,1])/le2[idx,1], 'x', label='net2D - like', alpha=0.2)
plt.xlabel('Mc')
plt.ylabel('frac delta stderr nu')
plt.legend()
plt.axis(ymax=4)

plt.tight_layout()

ncov = nc2 / ne2[:,0] / ne2[:,1]
lcov = lc2 / le2[:,0] / le2[:,1]

plt.plot(Mc[idx], ncov[idx] - lcov[idx], '.', label='net2D - like', alpha=0.2)
plt.xlabel('Mc')
plt.ylabel('frac delta cov Mc eta')
plt.legend()
plt.axis(ymin=-0.5, ymax=0.5)