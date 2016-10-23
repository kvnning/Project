# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 11:15:54 2016

@author: Kevin
"""
import numpy as np
import matplotlib.pyplot as plt
from pylab import e
from pylab import pi
#Setting variables and arrays...
Tmin = 0
Tmax = 1 #365.256*24*3600 #Duration of one year.
N = 365*2
G = 4*pi #AU^3 yr^-2 M0^-1
Times = np.linspace(Tmin,Tmax,N+1)
dT = (Times[2]-Times[0])*0.5
p = 2

""" May need to mess with array structure for efficency."""
particle =  np.zeros([10,p]) #Xn,Yn,Zn,Vx,Vy,Vz,fx,fy,fz,Mass
xold = np.zeros([3]) #Storage array for last position values. (Add to particle array?)

#Inital conditions (t = 0)
particle[9,0] = 1 #Sun mass

particle[0,1] = 1 #AU/s
particle[4,1] = particle[0,1]*2*pi #Speed in AU/yr
particle[9,1] = 0.000003003
plt.plot(particle[0,1],particle[1,1],'k.')

def CalTotalAcc(*particles):
    particle[6:9,:] = 0 #Cleaning accelerations.
    for i in range(0,p):
        for j in range(i+1,p):
            dX = particle[0,i]-particle[0,j]
            dY = particle[1,i]-particle[1,j]
            dZ = particle[2,i]-particle[2,j]
            R2 = dX**2+dY**2+dZ**2
            K= G/R2 #G/R^2
            #Adding changes in acceleration due to particles...
            particle[6,i] = particle[6,i]-K*particle[9,j]*dX #-G/R^2 * M * r_x
            particle[7,i] = particle[7,i]-K*particle[9,j]*dY
            particle[8,i] = particle[8,i]-K*particle[9,j]*dZ
            
            particle[6,j] = particle[6,j]+K*particle[9,i]*dX #Opposite dX vector,results in positive total.
            particle[7,j] = particle[7,j]+K*particle[9,i]*dY
            particle[8,j] = particle[8,j]+K*particle[9,i]*dZ
    return particle

#Calculating second position for Verlet Integration (t = dT)
particle = CalTotalAcc(particle)
for i in range(0,p):
    #Temporary storage array (Holding Rn-1)
    xold[0] = particle[0,i]
    xold[1] = particle[1,i]
    xold[2] = particle[2,i]
    #Calculating Rn
    particle[0,i] = particle[0,i]+particle[3,i]*dT+particle[6,i]*dT**2
    particle[1,i] = particle[1,i]+particle[4,i]*dT+particle[7,i]*dT**2
    particle[2,i] = particle[2,i]+particle[5,i]*dT+particle[8,i]*dT**2
    #Updating Rn-1 (Reusing Velocity memory location as no longer needed after : Vx,Vy,Vz -> Xn-1,Yn-1,Zn-1)
    particle[3,i] = xold[0]
    particle[4,i] = xold[1]
    particle[5,i] = xold[2]

for t in Times[1:]:
    for i in range(0,p):
        #Recalculating all accelrations
        particle = CalTotalAcc(particle)
        #Temporary storage array
        xold[0] = particle[0,i]
        xold[1] = particle[1,i]
        xold[2] = particle[2,i]
        #Verlet step : Rn+1 = 2Rn + Rn-1 + fr dT^2
        particle[0,i] = 2*particle[0,i]-particle[3,i]+particle[6,i]*dT**2
        particle[1,i] = 2*particle[1,i]-particle[4,i]+particle[7,i]*dT**2
        particle[2,i] = 2*particle[2,i]-particle[5,i]+particle[8,i]*dT**2
        #Updating Rn-1 values
        particle[3,i] = xold[0]
        particle[4,i] = xold[1]
        particle[5,i] = xold[2]
        plt.plot(particle[0,1],particle[1,1],'k.')
    print(t)
    print(particle[0,:])
    print(particle[1,:])
    
    