# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http:www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This file is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY, to the extent permitted by law; without
# even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE.

import numpy as np
#from .constants import *
import time
import multiprocessing
import os
from functools import partial
import h5py
import scipy
from FunctionXY import FunctionXY


try:
    from pyfftw.interfaces import numpy_fft as fft
    import pyfftw
    pyfftw.interfaces.cache.enable()
    print('using pyfftw for fft')
except ImportError:
    from numpy import fft
    print('using numpy for fft')


const_boltzmann = 8.6173303e-5
const_eV2kk = 1/2.072124652399821e-3
const_planck = 4.13566769692386e-15  #(source: NIST/CODATA 2018)
const_hbar = const_planck*0.5/np.pi

#default units:
#energy eV
#wavelength angstrom
#time second
#wavenumber angstrom^-1
#angle degree
from scipy.signal import kaiserord, lfilter, firwin, freqz
from scipy import interpolate

def takfft(input, dt=1., fftsize=None, conversion=2*np.pi):
    return fft.fftshift(fft.fft(fft.fftshift(input), n=fftsize))*dt/conversion

def takifft(input, dt=1., fftsize=None, conversion=2*np.pi):
    return fft.fftshift(fft.ifft(fft.fftshift(input), n=fftsize))/dt*conversion


def eKin2k(eV):
    return np.sqrt(eV*const_eV2kk)
#
def k2eKin(wn):
    return wn*wn/const_eV2kk
#
def q2Alpha(Q, kt):
    return Q*Q/(kt*const_eV2kk)
#
def alpha2Q(alpha,kt):
    return np.sqrt(alpha*kt*const_eV2kk)
#
def angle2Q(angle_deg, enin_eV, enout_eV):
    ratio = enout_eV/enin_eV
    k0=eKin2k(enin_eV)
    scale = np.sqrt(1.+ ratio - 2*np.cos(angle_deg*const_deg2rad) *np.sqrt(ratio) )
    return k0*scale


def expand(input, axis=0, neg_factor=1., pos_factor=1.):
    s = [slice(None)]*input.ndim
    s[axis] = slice(-2,0,-1)
    return np.concatenate((input[tuple(s)].conjugate()*neg_factor,input*pos_factor),axis=axis)

#
def nd2str(arr):
    return ' '.join(map(str,arr))


def minMaxQ(enin_eV, enout_eV):
    if enout_eV.min()<0:
        raise RuntimeError('Negative energy')
    ratio = enout_eV/enin_eV
    k0=eKin2k(enin_eV)
    qmin = k0*np.sqrt(1.+ ratio - 2*np.sqrt(ratio) )
    qmax = k0*np.sqrt(1.+ ratio + 2*np.sqrt(ratio) )
    return qmin, qmax


def writeDynInfo(fo, element, alpha, beta,  fraction, temperature, knl):
    dynInfo_str = """\n@DYNINFO
              \nelement  {element}
              \nfraction {fraction}
              \ntype     scatknl
              \ntemperature {temperature}
              \nalphagrid {alpha}
              \nbetagrid {beta}
              \nsab """
    fo.write(dynInfo_str.format(alpha=nd2str(alpha), beta=nd2str(beta),
            element=element, fraction=fraction, temperature=temperature))

    for line in knl:
        fo.write(nd2str(line))
        fo.write('\n')
    fo.write('\n')
#
def gaussSqw2ncmat(fname, density, elements, Qs, fres, sqws, fractions, temperature=293., plot=False):
    kt = temperature*const_boltzmann

    density_str = """@DENSITY
      {density} atoms_per_aa3\n
      """
    fo = open(fname, "w")
    fo.write('NCMAT v2\n')
    fo.write(density_str.format(density=density))

    for Q, element, fre, sqw, fraction in zip(Qs, elements, fres, sqws, fractions):
        alpha = q2Alpha(Q, kt)
        beta = (fre*const_hbar/kt)
        knl = sqw.swapaxes(0,1)*0.5*kt*kt*const_eV2kk/(2*np.pi)

        if plot:
            import matplotlib.pyplot as plt
            plt.figure()
            plt.plot(alpha, np.trapz(knl.T, beta))
            plt.title('zeroth of sab')
            plt.show()

        print('Zeroth momentum for element', element, ':', np.trapz(knl.T, beta))

        writeDynInfo(fo, element, alpha, beta, fraction, temperature, knl)

    fo.close()

def getOmegaFromTime(tsize, dt):
    #angular freqency
    fre = fft.fftshift(fft.fftfreq(tsize, dt))*2*np.pi
    return fre[1]-fre[0], fre

def takconv(input1, input2, fast=True):
    if input1.x.size > input2.x.size:
        a1=input1
        a2=input2
    else:
        a1=input2
        a2=input1

    deltaX1 = a1.getDeltaX()
    deltaX2 = a2.getDeltaX()

    if not (a1.distortFact==0. and a2.distortFact==0.):
        np.testing.assert_almost_equal(a1.distortFact/a2.distortFact, 1.) #functions with different distortFact
    if not (a1.asymExponent==0. and a2.asymExponent==0.):
        np.testing.assert_almost_equal(a1.asymExponent/a2.asymExponent, 1.) #functions with different asymExponent
    np.testing.assert_almost_equal(deltaX1/deltaX2, 1.) #functions with different spacing
    if fast:
        y= scipy.signal.convolve(a1.y, a2.y)*deltaX1
    else:
        y= np.convolve(a1.y, a2.y)*deltaX1
    zOfY = a1.getZ()+a2.x.size-a2.getZ()-1
    minx = -zOfY*deltaX1
    maxx = minx + (y.size-1)*deltaX1
    f = FunctionXY(np.linspace(minx, maxx, y.size), y, a1.distortFact, a1.asymExponent)
    return f


def seed(begin, num=10):
    x=np.logspace(np.log10(begin),np.log10(begin*np.sqrt(2.)), num)
    #skip the last one, as it can be calc using x[0]**sqrt(2.)
    return x[:-1]

# this function finds a suitable Q to calculate exp(-0.5*Q*Q*gamma)
# by trying to get the maxima of the exponent to a certain value
def findSeed4Gamma(gamma, targetExponent=20., num=10):
    exponentMax = np.max(0.5*np.abs(gamma))
    Qpow = targetExponent/exponentMax
    return seed(np.sqrt(Qpow), num)

def brewSeed(seed, power):
    return seed*np.sqrt(2.)**power

def calBrewNum(seed,qvalue):
    if qvalue<seed[0]:
        s = seed[0]
    else:
        s = seed[-1]
    return int(np.floor(np.log2(qvalue/s))*2)
