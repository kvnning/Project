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
SFE = 0.045 #Proportion of stars in a cloud.

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

def M_req(M_high):
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
    
#Aqcuiring values for curvefit...
y = np.linspace(6,40,1000)
x = M_req(y)
guess = [2,0.9,0.01]
popt,pcov = curve_fit(f,x,y,guess)
m = np.linspace(min(x),max(x),1000)
fm = f(m,*popt)
error = np.sqrt(np.diag(pcov))

plt.plot(x[::20],y[::20],'kx',label="Tabulated values")
plt.plot(m,fm,label="Curvefit")
GraphFmt("","Highest Star Mass (M*)","Total Cloud Mass (M*)",Legend=True)
plt.show()

print(popt)
print(error)

"""Actual functions used in program:"""

def SNF(x):
    """Relationship used to model supernova masses. a*x^b+c with a=0.24337062, b=0.74382654 & c=-1.00611354.
    All values were determined using a curvefit function on the 'required solar mass' integral."""
    return (0.02423769*x**0.74382668 -1.00610948)
    
test = 1337
f = open("Test.txt","w")
f.write("Line1")
f.write("Line2 %s \n" %test)
np.savetxt(f,popt)

np.savetxt(f,popt)
np.savetxt(f,pcov)
f.close()
f= open("Test.txt","r")
print(f.readline())
x = np.loadtxt(f,skiprows=6)
