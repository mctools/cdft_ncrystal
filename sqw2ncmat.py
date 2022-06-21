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
import matplotlib.pyplot as plt
#from Sqw import Sqw

from helper import *


fres=[]
sqws=[]
fractions=[]
Qs=[]


input=["./data/last_H.sqwhdrft","./data/last_O.sqwhdrft"]
count=[2,1]
elements=["H","O"]
density=1.
temperature=293
outfName="./data/last.ncmat"

for fn, weight, ele in zip(input, count, elements):
    f=h5py.File(fn,'r')
    s=f['inc_sqw_'+ele][()]
    q=f['qVec_'+ ele][()]
    w=f['inc_omega_'+ele][()]
    f.close()
    #sqw=Sqw(s,q,w,temperature)
    fractions.append(float(weight))
    # sqws.append(sqw.s/const_hbar)
#     Qs.append(sqw.q)
#     fres.append(sqw.w)
    sqws.append(s/const_hbar)
    Qs.append(q)
    fres.append(w)


fractions=np.array(fractions)
fractions=fractions/fractions.sum()
gaussSqw2ncmat(outfName, density, elements, Qs, fres, sqws, fractions, temperature, True)
