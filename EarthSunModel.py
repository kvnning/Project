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
Tmax = 1/14909340.84 #365.256*24*3600 #Duration of one year.
N = 365*8
G = 1 #AU^3 yr^-2 M0^-1
Times = np.linspace(Tmin,Tmax,N+1) #N+1 to change interval to steps.
dT = (Times[2]-Times[0])*0.5
p = 2 #No. of particles.
E = 1e-21 #Softening Coefficent.

Xmin = 0
Ymin = 0
Xmax = 2e-5
Ymax = 2e-5

""" May need to mess with array structure for efficency."""
particle =  np.zeros([10,p]) #Xn,Yn,Zn,  Vx,Vy,Vz,  fx,fy,fz,  Mass
xold = np.zeros([3,p]) #Storage array for last position values. (Add to particle array?)
PE = np.zeros([p])
KE = np.zeros([p])

"""Inital conditions (t = 0)"""
#Sun
particle[9,0] = 1 #Sun mass
#particle[3,0] = 1 #Extra random vector
#Earth
particle[0,1] = 4.84*(10**-6) #AU/s
particle[4,1] = particle[0,1]*2*pi/Tmax #Speed in AU/yr
#particle[3,1] = 1 #Extra random vector
particle[9,1] = 0.000003003

#Mars
"""
particle[0,2] = -1.524
particle[4,2] = (particle[0,2]*2*pi)/1.8809
particle[9,2] = 3.213e-7
"""
#Moon
"""
particle[0,2] = 1.00257
particle[4,2] = particle[4,1] + (0.00257*2*pi)/(27.32/365.25) #Earth speed + (Lunar radius/orbit time)
particle[9,2] = 3.69396868e-8
"""
    
def CalTotalAcc(*particles):
    """Particle acceleration (due to gravity) calculation. Applies for all particles given"""
    particle[6:9,:] = 0 #Cleaning accelerations.
    for i in range(0,p-1):
        for j in range(i+1,p):
            #Calculating distances.
            dX = particle[0,i]-particle[0,j]
            dXb = BoundaryDistance(dX,Xmax,Xmin)
            dY = particle[1,i]-particle[1,j]
            dYb = BoundaryDistance(dY,Ymax,Ymin)
            dZ = particle[2,i]-particle[2,j]
            
            #Choosing the shorter distance...
            if abs(dX) > abs(dXb):
                dX = dXb
            if abs(dY) > abs(dYb):
                dY = dYb
            
            #Acceleration Calaculation
            R2 = (dX**2+dY**2+dZ**2)
            R = np.sqrt(R2)
            K= G/(R+E)**3 #G/R^3 (Avoids using force to speed up code - no extra divisions)
            #Adding changes in acceleration due to particles...
            particle[6,i] = particle[6,i]-K*particle[9,j]*dX #-G/R^3 * M * r_x
            particle[7,i] = particle[7,i]-K*particle[9,j]*dY #-G/R^3 is 'K'
            particle[8,i] = particle[8,i]-K*particle[9,j]*dZ
            
            particle[6,j] = particle[6,j]+K*particle[9,i]*dX #Opposite dX vector,results in positive total.
            particle[7,j] = particle[7,j]+K*particle[9,i]*dY
            particle[8,j] = particle[8,j]+K*particle[9,i]*dZ
            
            """
            PE[0] = -G*particle[9,i]/R2
            KE[0] = 0.5*particle[9,i]*(particle[3,i]**2+particle[4,i]**2+particle[5,i]**2)
            PE[j] = -G*particle[9,j]/R2
            KE[j] = 0.5*particle[9,j]*(particle[3,j]**2+particle[4,j]**2+particle[5,j]**2)
            plt.plot(t,KE[i]+PE[i],'k.')
            plt.plot(t,KE[j]+PE[j],'b.')
            """
    return particle
    
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

def PosBoundary(p,Max,Min):
    """Small function to teleport particles that exceed the boundaries to the other side of the box."""
    particle = p
    if particle > Max:
        return particle-(Max-Min)
    if particle < Min:
        return particle+(Max-Min)
    return particle
    
def BoundaryDistance(d,Max,Min):
    """Acquires the distance from two points going through the boundaries"""
    distance = d
    if distance < 0:
        return distance + (Max-Min)
    else:
        return distance -(Max-Min)
        
#Velocity Verlet function
particle = CalTotalAcc(particle)

for t in Times[1:]:
    for i in range(0,p):
        #Recalculating all accelrations
        #Verlet step (Position) : Rn+1 = Rn + Vn*dT + 0.5*fn*dT^2
        particle[0,i] = particle[0,i] + particle[3,i]*dT + 0.5*particle[6,i]*dT**2
        particle[1,i] = particle[1,i] + particle[4,i]*dT + 0.5*particle[7,i]*dT**2
        particle[2,i] = particle[2,i] + particle[5,i]*dT + 0.5*particle[8,i]*dT**2
        #Position boundaries
        particle[0,i] = PosBoundary(particle[0,i],Xmax,Xmin)
        particle[1,i] = PosBoundary(particle[1,i],Ymax,Ymin)
        #Temporary storage array (Holding fn)
        xold[0,i] = particle[6,i]
        xold[1,i] = particle[7,i]
        xold[2,i] = particle[8,i]
        
    #Updating Acceleration (fn+1)
    particle = CalTotalAcc(particle)
    
    for i in range(0,p):
        #Calculating resulting velocity : Vn+1 = Vn + 0.5*(fn + fn+1)*dT
        particle[3,i] = particle[3,i] + 0.5*(xold[0,i]+particle[6,i])*dT
        particle[4,i] = particle[4,i] + 0.5*(xold[1,i]+particle[7,i])*dT
        particle[5,i] = particle[5,i] + 0.5*(xold[2,i]+particle[8,i])*dT
        
        """Things to plot"""
        plt.plot(particle[0,1],particle[1,1],'b.')
        plt.plot(particle[0,0],particle[1,0],'y.')
        #plt.plot(particle[0,2],particle[1,2],'rx')
        #plt.plot(t,(particle[0,1]**2+particle[1,1]**2)-(particle[0,2]**2+particle[1,2]**2),'k.')
        #plt.plot(particle[0,3],particle[1,3],'r.')
    #print(t)
    #print(particle[0,:])
    #print(particle[1,:])
plt.show()

"""#Basic Verlet Function (Position ONLY)

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

#Main Verlet Integration Loop (Starts second timestep due to above)
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
        plt.plot(particle[0,0],particle[1,0],'b.')
    #print(t)
    #print(particle[0,:])
    #print(particle[1,:])
plt.show()
"""
    