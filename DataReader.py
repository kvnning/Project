# -*- coding: utf-8 -*-
"""
Created on Fri Apr 07 10:41:17 2017

@author: Kevin
"""
import time as time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import curve_fit

st = 14909319.84
n = 12264  #Total number of iterations +1 - MUST BE EQUAL OR LOWER

#Data file dumps
f = open("ParticleBoxDump.txt","r")
f2 = open("PBSupernovaHist.txt","r")
#f2 - Supernova data. Loop Num, Supernova particle num, collision history, Mass, SFE
#f3 - Initial high mass star data. loop no, particle no, Collision history, SFE, Mass.

#Arrays
Time = np.zeros(n)
Active = np.zeros(n)
Countdown = np.zeros(n)
Bondi = np.zeros(n)
EGain = np.zeros(n)
Collision = np.zeros(n)
ZLoss = np.zeros(n)
SN = np.zeros(n)
BH = np.zeros(n)
SFMGain = np.zeros(n)
SFE = np.zeros(n)
NumCollision = np.zeros(n)
NumSupernova = np.zeros(n)
TotalM = np.zeros(n) # Now average mass
ParticleM = np.zeros(n) #For specfic particle tracking

b= 100 #Supernova data
snn = np.zeros(b) #Loop no.
sni = np.zeros(b) #Particle no.
snc = np.zeros(b) #Collision no.
snm = np.zeros(b) #Mass
sne = np.zeros(b) #SFE
snsm = np.zeros(b) #Highest star mass

p = 1300
SFM = np.zeros(p)
x = np.zeros(p)
y = np.zeros(p)
z = np.zeros(p)
m = np.zeros(p)
CollidedBefore = np.zeros(p)

Active[0] = 863 #Starting active particle number.

#Functions
def linear(x,a,b):
    """For curvefitting"""
    return a*x + b

def GraphFmt(Title,Ylabel,Xlabel,xmin=None,xmax=None,ymin=None,ymax=None,Legend=False):
    """ A short graph formatting function. Serves no other purpose than to save space."""
    plt.title(Title)
    plt.ylabel(Ylabel)
    plt.xlabel(Xlabel)
    plt.ylim(ymin,ymax)
    plt.xlim(xmin,150)
    if Legend != False:
        plt.legend(loc="best")
    return None
    
def SNF(CloudMass,SFE):
    """Relationship used to model supernova masses. a*CloudMass^b + c. 'a' is a function of SFE, in the form of a'*SFE^b'.
    SFE in this context refers to the fraction/precentage of cloud mass that is stars.
    a' = 0.24337061
    b' = 0.74382654
    b  = 0.74382654
    c  = -1.00611354
    All values were determined using a curvefit function on the 'required solar mass' integral."""
    return ((0.24337061*SFE**0.7438265)*CloudMass**0.74382654 -1.00611354)
    
    
Animation = False
if Animation == True: #3D plotting
    fig = plt.figure()
    fig.clf()
    ax = fig.add_subplot(111,projection='3d')

#Data reading
for i in range(1,n):
    f.readline()
    Time[i] = f.readline()
    Active[i] = f.readline()
    Countdown[i] = f.readline()
    
    Bondi[i] = f.readline()
    EGain[i] = f.readline()
    Collision[i]= f.readline()
    
    ZLoss[i] = f.readline()
    SN[i] = f.readline()
    BH[i] = f.readline()
    
    SFMGain[i] = f.readline()
    SFE[i] = f.readline()
    
    NumCollision[i] = f.readline()
    NumSupernova[i] = f.readline()
    for a in range(0,int(Active[i])):
        SFM[a] = f.readline()
    f.readline()
    for a in range(0,int(Active[i])):
        x[a] = f.readline()
    f.readline()
    for a in range(0,int(Active[i])):
        y[a] = f.readline()
    f.readline()
    for a in range(0,int(Active[i])):
        z[a] = f.readline()
    f.readline()
    for a in range(0,int(Active[i])):
        m[a] = f.readline()
    ParticleM[i] = m[6] #Specfic particle tracking
    
    if i%2 == 0 and Animation == True:
        for a in range(0,int(Active[i])):
            ax.scatter(x[a],y[a],z[a],s=m[a]**(2./3))
        ax.set_xlim(0,100)
        ax.set_ylim(0,100)
        ax.set_zlim(-175,175)
        fig.savefig("Step%d" %i)
        ax.cla()
    TotalM[i] = np.average(m) #Total GL growth over time
SITime = Time*st/10**6 #Conversion to Myrs

#Second file reading
f2.readline()
for i in range(0,int(NumSupernova[-1])):
    snn[i] = f2.readline()
    sni[i] = f2.readline()
    snc[i] = f2.readline()
    snm[i] = f2.readline()
    sne[i] = f2.readline()
    snsm[i] = SNF(snm[i],sne[i])
    
#2D plots
if Animation == False:
    Accretion = Bondi + EGain
    BondiRate = (Bondi[1:]-Bondi[:-1])/((Time[1:]-Time[:-1])*st/10**6)
    EGainRate = (EGain[1:]-EGain[:-1])/((Time[1:]-Time[:-1])*st/10**6)
    ZLossRate = (ZLoss[1:]-ZLoss[:-1])/((Time[1:]-Time[:-1])*st/10**6)

    plt.semilogy(Time*st/10**6,Active)
    GraphFmt("","Number of particles in system","Time (Myr)")
    plt.savefig("PBActive")
    plt.clf()

    plt.plot(Time*st/10**6,Countdown)
    GraphFmt("","Number of particles with high mass star","Time (Myr)")
    plt.savefig("PBCountdown")
    plt.clf()

    plt.semilogy(Time*st/10**6,Accretion,label="Accretion")
    plt.semilogy(Time*st/10**6,Collision,label="Collision")
    GraphFmt("","Total mass gain (M*)","Time (Myr)",Legend=True)
    #plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.savefig("PBGrowth")
    plt.clf()
    
    plt.semilogy(Time*st/10**6,Bondi,label="Bondi-Hoyle")
    plt.semilogy(Time*st/10**6,EGain,label="Cloud Radius")
    GraphFmt("","Total Mass gain from accretion (M*)","Time (Myr)",Legend=True)
   # plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.savefig("PBAccretion")
    plt.clf()
    
    plt.semilogy(Time[1:]*st/10**6,BondiRate,label="Bondi-Hoyle",)
    plt.semilogy(Time[1:]*st/10**6,EGainRate,label="Cloud Radius")
    GraphFmt("","Total accretion rate (M*/Myr)","Time (Myr)",Legend=True)
    plt.savefig("PBAccretionRate")
    plt.clf()
    
    
    plt.plot(Time*st/10**6,ZLoss,label="Z-axis Escape")
    plt.plot(Time*st/10**6,SN,label="Supernova")
    GraphFmt("","Total mass loss (M*)","Time (Myr)",Legend=True)
    #plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.savefig("PBLoss")
    plt.clf()
    
    plt.semilogy(Time*st/10**6,SFMGain,label="Total Star mass")
    GraphFmt("","Total Star mass (M*)","Time (Myr)")
    #plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.savefig("PBSFM")
    plt.clf()
    
    plt.semilogy(Time*st/10**6,SFE) #Probably change to individual SFEs.
    GraphFmt("","Average SFE per particle","Time (Myr)")
    plt.savefig("PBSFE")
    plt.clf()
    
    plt.plot(Time*st/10**6,NumCollision) #Probably change to individual SFEs.
    GraphFmt("","Number of collisions","Time (Myr)")
    plt.savefig("PBCollisions")
    plt.clf()
    
    #More interesting plots...
    
    #For stablity testing
    plt.plot(Time*st/10**6,ZLoss+SN,label="Loss")
    plt.plot(Time*st/10**6,Accretion,label="Accretion")  
    plt.plot(Time*st/10**6,Accretion-(ZLoss+SN),'k',label="Net")
    print("Intergal of net GL: %.d" %(np.sum(Accretion-(ZLoss+SN))))
    GraphFmt("","Total mass gain/loss (M*)","Time (Myr)",Legend=True)
    plt.savefig("PBNetGL")
    plt.clf()    
    
    plt.plot(Time[5:]*st/10**6,TotalM[5:],'k',label="Average particle mass")
    GraphFmt("","Average particle mass (M*)","Time (Myr)")
    plt.savefig("PBNetMass")
    plt.clf() 
    
    #Particle tracking
    plt.plot(Time[1:]*st/10**6,ParticleM[1:])
    GraphFmt("","Particle mass (M*)","Time (Myr)")
    plt.savefig("PBSpecficMass")
    plt.clf() 
    
    #Curvefits
    AverageParticleNo = np.average(Active[800:])
    AverageSNMass = (SN[-1]/0.65)/NumSupernova[-1]
    apopt,apcov =  curve_fit(linear,SITime,Accretion/AverageParticleNo)
    aerror = np.sqrt(np.diag(apcov))
    cpopt,cpcov =  curve_fit(linear,SITime,Collision/np.average(Countdown))
    cerror = np.sqrt(np.diag(cpcov))
    AverageLT = (AverageSNMass)/(apopt[0]+cpopt[0])
    print(AverageLT)
    
f.close()
f2.close()
            