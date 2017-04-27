# -*- coding: utf-8 -*-
"""
Created on Sun Apr 02 21:04:16 2017

@author: Kevin
"""

import time as time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pylab import e
from pylab import pi
from scipy.optimize import curve_fit

"""Kroupa integrals. NK refers to the integral over all space to determine normalisation constant k.
N refers to the probabilties of aqcuiring a star in the mass range (so that '1/(1-Ns)' gives
the expected number of stars for a high mass star). M is effectively the total mass contribution
per star (M*N_expected = M_required)."""

MaxMass = 315
#SFE = 0.045 #Proportion of stars in a cloud.
SFEarray = np.linspace(0.001,1,1000)
popt = np.zeros([1000,3])
error = np.zeros([1000,3])

NK1 = (10./7)*0.08**0.7-(10./7)*0.01**0.7
NK2 = (-10./3)*0.5**-0.3-(-10./3)*0.08**-0.3
NK3 = (-10./13)*MaxMass**-1.3-(-10./13)*0.5**-1.3

k = 1./(NK1+NK2+NK3)

N1 = (k*10./7)*0.08**0.7-(k*10./7)*0.01**0.7
N2 = (-k*10./3)*0.5**-0.3-(-k*10./3)*0.08**-0.3
N3 = (k*10./13)*0.5**-1.3 #Double negative -> actually only half of the third integral, as it depends on star mass.

M1 = (k*10./17)*0.08**1.7-(k*10./17)*0.01**1.7
M2 = (k*10./7)*0.5**0.7-(k*10./7)*0.08**0.7
M3 = (k*10./3)*0.5**-0.3 #Double negative

def M_req(M_high,SFE):
    """Function used to determine expected cloud mass from a given star mass."""
    M = (M1 + M2 + M3 +(k*10./3)*M_high**-0.3)/((1 - N1 - N2 - N3 -(-k*10./13)*M_high**-1.3)*SFE)
    return M

def f(m,a,b,c):
    """Relationship used to model the mass function."""
    return a*m**b+c

def GraphFmt(Title,Ylabel,Xlabel,xmin=None,xmax=None,ymin=None,ymax=None,Legend=False):
    """ A short graph formatting function. Serves no other purpose than to save space."""
    plt.title(Title)
    plt.ylabel(Ylabel)
    plt.xlabel(Xlabel)
    plt.ylim(ymin,ymax)
    plt.xlim(xmin,xmax)
    if Legend != False:
        plt.legend(loc="best")
    return None
    
for i in range(0,1000):
    #Aqcuiring values for curvefit...
    y = np.linspace(6,40,1000)
    x = M_req(y,SFEarray[i])
    guess = [2,0.9,0.01]
    popt[i,:],pcov = curve_fit(f,x,y,guess)
    error[i,:] = np.sqrt(np.diag(pcov))
 

plt.plot(SFEarray,popt[:,0])

apopt,apcov = curve_fit(f,SFEarray,popt[:,0],[0.25,0.9,0])
aerror = np.sqrt(np.diag(apcov))
plt.plot(SFEarray,f(SFEarray,*apopt))
plt.show()

def Graph(i):
    y = np.linspace(6,40,1000)
    x = M_req(y,SFEarray[i])
    m = np.linspace(min(x),max(x),1000)
    fm = f(m,*popt[i,:])
    plt.plot(x[::20],y[::20],'kx',label="Tabulated values")
    plt.plot(m,fm,label="Curvefit")
    GraphFmt("","Highest Star Mass (M*)","Total Cloud Mass (M*)",Legend=True)
    print(SFEarray[i])
    plt.show()

print(popt)
print(error)

"""Actual functions used in program:"""

def SNF(CloudMass,SFE):
    """Relationship used to model supernova masses. a*CloudMass^b + c. 'a' is a function of SFE, in the form of a'*SFE^b'.
    SFE in this context refers to the fraction/precentage of cloud mass that is stars.
    a' = 0.24337061
    b' = 0.74382654
    b  = 0.74382654
    c  = -1.00611354
    All values were determined using a curvefit function on the 'required solar mass' integral."""
    return ((0.24337061*SFE**0.7438265)*CloudMass**0.74382654 -1.00611354)
    
""" Column density note to self:
Required CD is 10^21 particles per cm-2. We use 100 particles per cm^-3. In order to achieve the
minimum CD the column width required is 10^19 centimeters. This corresponds to 3.2407793 parsecs.
From 4/3*pi*r^3*rho (4./3*pi*3.2407793**3*clouddensity), can aqcuire a minimum cloud mass of 1057
solar masses before star formation. """

