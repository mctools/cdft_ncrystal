#!/usr/bin/env python3

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
import sys
import h5py
import matplotlib.pyplot as plt
from helper import getOmegaFromTime,takfft, takifft, takfft, takconv,findSeed4Gamma,calBrewNum
from FunctionXY import FunctionXY

##const value
const_planck = 4.13566769692386e-15  #(source: NIST/CODATA 2018)
const_hbar = const_planck*0.5/np.pi
const_boltzmann = 8.6173303e-5
const_neutron_mass_evc2 = 1.0454075098625835e-28  #eV/(Aa/s)^2  #fixme: why not calculated from other constants).#<EXCLUDE-IN-NC1BRANCH>

class HDRFT():
    def __init__(self, tVec, exponent):
        self.dt = tVec[1]-tVec[0]
        if tVec[0]+tVec[-1] > 2*self.dt:
            raise RuntimeError("tVec[0]+tVec[-1] > 2*dt  ")
        if not np.all(np.diff(tVec)) > 0.:
            raise RuntimeError("tVec is expected to be monotonically increasing ")
        self.t=tVec
        self.deltaf, self.freq = getOmegaFromTime(self.t.size, self.dt)
        self.exponent = exponent

    def getExponent(self):
        return self.exponent

    def calcFFT(self,window=None):
        y=np.exp(self.getExponent())
        if window is not None:
            if window=="kaiser":
                y*=np.kaiser(y.size,20)
            if window=="hanning":
                y*=np.hanning(y.size)
        directFFT = np.abs(takfft(y, self.dt))
        directFre = self.freq
        return directFre, directFFT

    def calHdRFT(self, order, firstOrderCutoff , distortFact,
                asymExponent, offset=0,window=None,isNormalise=False, xBoundary=10./const_hbar,autodistort=False):
        hdrftFre = self.freq
        ft=self.getExponent()
        x=ft.min()+offset
        rt=ft-x
        # print(f"x={x}")
        r0=np.interp(0., self.t, rt)
        f0=np.ones(self.t.size)
        f1=rt/r0
        if window is not None:
            if window=="kaiser":
                f0*=np.kaiser(f0.size,20)
                f1*=np.kaiser(f1.size,20)
            if window=="hanning":
                f0*=np.hanning(f0.size)
                f1*=np.hanning(f1.size)
        g0=takfft(f0, self.dt)
        g1=takfft(f1, self.dt)

        deltaOmega = abs(self.freq[-1]-self.freq[0])/(self.freq.size-1)

        totalxy = FunctionXY(self.freq, np.abs(r0*g1+g0), distortFact,asymExponent, autodistort)
        g1xy = FunctionXY(self.freq, np.copy(g1), distortFact,asymExponent, autodistort)
        gnxy=FunctionXY(self.freq, np.copy(g1), distortFact, asymExponent, autodistort)

        if isNormalise:
            g1xy.normalise()
            gnxy.normalise()

        totalxy.crop(-firstOrderCutoff, firstOrderCutoff)
        g1xy.crop(-firstOrderCutoff, firstOrderCutoff)
        gnxy.crop(-firstOrderCutoff, firstOrderCutoff)
        g1xy.distort()
        g1xy.flipNeg2Pos()


        gnxy.distort()
        gnxy.flipNeg2Pos()

        coef = r0

        for n in range(2, order+1):
            gnxy = takconv(gnxy, g1xy)
            gnxy.flipNeg2Pos()
            gnxy.restort()
            # if isNormalise:
            #     gnXYRecoveredAear = np.trapz(np.abs(gnxy.f), gnxy.x)
            #     print(f'gnxy order {n}, recovered area {gnXYRecoveredAear}')
            #     #making sure the area is always unity
            #     gnxy.f *= 1./gnXYRecoveredAear

            gnXYRecovered=np.abs(gnxy.y)
            gnXYRecoveredAear = np.trapz(gnXYRecovered, gnxy.x)
            print(f'gnxy order {n}, recovered area {gnXYRecoveredAear}')
            if gnXYRecoveredAear>1.1 or gnXYRecoveredAear< 0.9:
                plt.figure()
                plt.semilogy(gnxy.x*const_hbar, gnXYRecovered, label= 'recovered, order={n} ')
                gnxy.distort()
                plt.semilogy(gnxy.x*const_hbar, np.abs(gnxy.y), label= 'distorted, order={n} ')
                plt.title('Debug info, recovered result is not close to unity')
                plt.legend()
                plt.show()
                raise RuntimeError('recovered result is not close to unity')

            if autodistort:
                gnxy.calDistortFact()

            coef *= r0/n
            gnxy.scaleY(coef)
            totalxy.accumulate(gnxy)
            gnxy.scaleY(1./coef)

            gnxy.distort()


            if isNormalise:
                gnxy.scaleY(1./gnXYRecoveredAear)

            if gnxy.x[-1]> xBoundary:
                gnxy.crop(-xBoundary,xBoundary)
        totalxy.scaleY(np.exp(x))
        totalxy.flipNeg2Pos()
        if isNormalise:
            totalxy.normalise()

        return totalxy.x, totalxy.y


class Conv_Sw(HDRFT):
    def __init__(self,tVec,gamma,distortFact, asymExponent,firstOrderCutoff=0.9,maxEnergy=10.):
        # calculate sqw from gamma: small q use calHdrft(), large q use seedQ to conv
        # calSw(qvale), calSqw(qVec)
        super().__init__(tVec, None)
        self.gamma=gamma
        self.cutoffValue = firstOrderCutoff/const_hbar
        self.distortFact=distortFact
        self.asymExponent=asymExponent
        self.seedQ = findSeed4Gamma(gamma, targetExponent=20., num=10)
        # self.seedQ = np.logspace(-0.23, -0.1, 4)/np.sqrt(2.)
        self.startQ= self.seedQ[-1]*1.0001 #np.sqrt(500./abs(gamma.max()))
        self.ord=100

    def calLargeQ(self,q):
        cutQ=self.seedQ[-1]

        conv_times=calBrewNum(self.seedQ,q)
        print("calculate conv times=",conv_times)
        self.exponent = -0.5*cutQ*cutQ*self.gamma
        omega, baseSW = self.calcFFT(window="kaiser")
        deltaOmega = omega[1]-omega[0]
        data=FunctionXY(np.copy(omega),np.copy(baseSW),
                        self.distortFact, self.asymExponent)
        data.distort()
        data.flipNeg2Pos()
        data.normalise()
        if data.x[-1]>5./const_hbar:
            data.crop(-5./const_hbar,5./const_hbar)
        data.restort()

        for n in range(1,conv_times+1):
            qeff = np.sqrt(2.**n)*cutQ
            data.distort()
            if qeff<2.:
                print(f"start qeff between 0.5~2 conv:{n}, qeff={qeff},cutQ={cutQ}")
                data = takconv(data,data,True)
                data.flipNeg2Pos()
                data.restort()
                data.normalise()
                if data.x[-1]>3./const_hbar:
                    data.crop(-3./const_hbar, 3./const_hbar)
                data = FunctionXY(np.copy(data.x),np.copy(data.y),
                                self.distortFact, self.asymExponent)

            if qeff >=2 and qeff<10.:
                print(f"start qeff between 2~10, conve:{n}, qeff={qeff},cutQ={cutQ}")
                data = takconv(data,data,True)
                data.flipNeg2Pos()
                data.restort()
                data.normalise()
                if data.x[-1]>5./const_hbar:
                    data.crop(-5./const_hbar, 5./const_hbar)
                data = FunctionXY(np.copy(data.x),np.copy(data.y),
                                     self.distortFact*3./4, self.asymExponent)
            if qeff>=10.:
                print(f"start qeff between 10~ conv:{n}, qeff={qeff},cutQ={cutQ}")
                data.restort()
                jump=3
                zeropos = int(round(data.getZ()%jump))
                print(f'zero offset {zeropos}')
                data=FunctionXY(np.copy(data.x[zeropos::jump]),np.copy(data.y[zeropos::jump]),
                        0., self.asymExponent)
                data = takconv(data,data,False)
                data.normalise()

        return qeff,data.x, np.abs(data.y)


class Freegas():
    def __init__(self,fre,dos,temperature,massNum=1):
        #super().__init__(fre,dos)
        self.temperature=temperature
        self.mass=massNum*const_neutron_mass_evc2
        self.fre=fre
        self.dos=dos
    def calEffTemp(self):
        detbal = const_hbar/(const_boltzmann*self.temperature)
        fw_1 = np.array([self.dos[0]])
        fw_2 = self.dos[1:]*self.fre[1:]*detbal/2/np.tanh(self.fre[1:]*detbal/2)
        fw = np.concatenate((fw_1,fw_2))
        return np.trapz(fw,self.fre)*self.temperature


    def freeScattering(self, omega, Q):
        T = self.calEffTemp()
        beta = 1./(T*const_boltzmann)
        Er = (const_hbar*Q)**2./(2.*self.mass)
        sw = np.sqrt(beta/(4*np.pi*Er))*np.exp(-beta/(4*Er)*(const_hbar*omega-Er)**2)
        print(sw)
        return sw*const_hbar*np.exp(-const_hbar*omega*beta)



def expandGammaFromH5(filename):
    f=h5py.File(filename,'r')
    t_old=f["time_vec"][()]
    g_cls = f['gamma_cls'][()]
    g_qtm_r=f["gamma_qtm_real"][()]
    g_qtm_i=f["gamma_qtm_imag"][()]
    f.close()
    #print(t_old.size)
    t_neg = -np.flip(t_old)
    # print("t neg=",t_neg)
    time=np.concatenate((t_neg,t_old[1:]))
    g_neg = np.flip(g_cls)
    gamma_cls=np.concatenate((g_neg,g_cls[1:]))
    g_neg = np.flip(g_qtm_r)
    gamma_qtm_r=np.concatenate((g_neg,g_qtm_r[1:]))
    g_neg = -np.flip(g_qtm_i)
    gamma_qtm_i=np.concatenate((g_neg,g_qtm_i[1:]))

    gamma_qtm=np.zeros(time.size,dtype=complex)
    for i in range(time.size):
        gamma_qtm[i]=complex(gamma_qtm_r[i],gamma_qtm_i[i])
    return time,gamma_cls,gamma_qtm


q=float(sys.argv[1]) #input 1 and 80

temperature=293
element="H"
gammafile="./data/last_"+element+".gamma"
#gammafile="test.gamma"

detbal=-const_hbar/(temperature*const_boltzmann)
distortFact=detbal/3
asymExponent=detbal

time,_,gamma_qtm=expandGammaFromH5(gammafile)

firstCut=0.9 #unit = eV
if q<2.:
    order=120
    isNorm=True
    window="kaiser"
    exponent = -0.5*q*q*gamma_qtm

    hdr = HDRFT(time,exponent)
    xhdr,yhdr = hdr.calHdRFT(order, firstCut/const_hbar, distortFact,
                        asymExponent,window=window, isNormalise=isNorm,autodistort=False)
    xfft, yfft = hdr.calcFFT()
    plt.semilogy(xhdr,np.abs(yhdr),label="cdft")
    plt.semilogy(xfft,np.abs(yfft),label="fft")
    plt.title(f"q={q}")
    plt.legend()
    plt.show()
else:
    filename="./data/last.sqw"
    f=h5py.File(filename,'r')
    fre=f["inc_omega_"+element][()]
    dos=f["inc_vdos_"+element][()]
    scale = np.trapz(dos,fre)
    dos /= scale
    f.close()
    fg = Freegas(fre,dos,temperature)
    hdr = Conv_Sw(time,gamma_qtm,distortFact,asymExponent,firstOrderCutoff=firstCut)
    qeff, xhdr, yhdr = hdr.calLargeQ(q)
    ygas = fg.freeScattering(xhdr,qeff)

    plt.plot(xhdr*const_hbar,yhdr,label="cdft")
    plt.plot(xhdr*const_hbar,ygas,label="stc")
    plt.legend()
    plt.title(f"q={qeff}")
    plt.show()
