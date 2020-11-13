# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 18:06:12 2018

@author: Chandrasen.Wadikar
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import pylab as pl

h=[186, 176, 158, 180, 186, 168, 168, 164, 178, 170, 189, 195, 172,
     187, 180, 186, 185, 168, 179, 178, 183, 179, 170, 175, 186, 159,
     161, 178, 175, 185, 175, 162, 173, 172, 177, 175, 172, 177, 180]


fit = stats.norm.pdf(h,np.mean(h),np.std(h))


You may try using hist to put your data info along with the fitted curve as below:

import numpy as np
import scipy.stats as stats
import pylab as pl

h = sorted([186, 176, 158, 180, 186, 168, 168, 164, 178, 170, 189, 195, 172,
     187, 180, 186, 185, 168, 179, 178, 183, 179, 170, 175, 186, 159,
     161, 178, 175, 185, 175, 162, 173, 172, 177, 175, 172, 177, 180])  #sorted

fit = stats.norm.pdf(h, np.mean(h), np.std(h))  #this is a fitting indeed

pl.plot(h,fit,'-o')

pl.hist(h,normed=True)

pl.show()



import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

h = [186, 176, 158, 180, 186, 168, 168, 164, 178, 170, 189, 195, 172,
     187, 180, 186, 185, 168, 179, 178, 183, 179, 170, 175, 186, 159,
     161, 178, 175, 185, 175, 162, 173, 172, 177, 175, 172, 177, 180]
h.sort()
hmean = np.mean(h)
hstd = np.std(h)
pdf = stats.norm.pdf(h, hmean, hstd)
plt.plot(h, pdf) # including h here is crucial