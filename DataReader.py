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

st = 14909319.84

n = 16174 #Total number of iterations
f = open("ParticleBoxDump.txt","r")
Time = np.zeros(n)
Active = np.zeros(n)
Countdown = np.zeros(n)
Accretion = np.zeros(n)
Collision = np.zeros(n)
ZLoss = np.zeros(n)
SN = np.zeros(n)
BH = np.zeros(n)

x = np.zeros(n)
y = np.zeros(n)
z = np.zeros(n)
m = np.zeros(n)

Active[0] = 863 #Starting active particle number.

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
    
Animation = True

if Animation == True: #3D plotting
    fig = plt.figure()
    fig.clf()
    ax = fig.add_subplot(111,projection='3d')


for i in range(1,n):
    f.readline()
    Time[i] = f.readline()
    Active[i] = f.readline()
    Countdown[i] = f.readline()
    Accretion[i] = f.readline()
    Collision[i]= f.readline()
    ZLoss[i] = f.readline()
    SN[i] = f.readline()
    BH[i] = f.readline()
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
    if i%21 == 0 and Animation == True:
        for a in range(0,int(Active[i])):
            ax.scatter(x[a],y[a],z[a],s=m[a]**(2./3))
        ax.set_xlim(0,100)
        ax.set_ylim(0,100)
        ax.set_zlim(-175,175)
        fig.savefig("Step%d" %i)
        ax.cla()
        
if Animation == False:
    plt.plot(Time*st/10**6,Active)
    GraphFmt("","Number of particles in system","Time (Myr)")
    plt.savefig("PBActive")
    plt.clf()

    plt.plot(Time*st/10**6,Countdown)
    GraphFmt("","Number of particles with high mass star","Time (Myr)")
    plt.savefig("PBCountdown")
    plt.clf()

    plt.plot(Time*st/10**6,Accretion,label="Accretion")
    plt.plot(Time*st/10**6,Collision,label="Collision")
    GraphFmt("","Total mass growth (M*)","Time (Myr)",Legend=True)
    plt.savefig("PBGrowth")
    plt.clf()

    plt.plot(Time*st/10**6,ZLoss,label="Z-axis Escape")
    plt.plot(Time*st/10**6,SN,label="Supernova")
    GraphFmt("","Total mass loss (M*)","Time (Myr)",Legend=True)
    plt.savefig("PBLoss")
    plt.clf()

f.close()
            