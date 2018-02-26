# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 11:15:54 2016

@author: Kevin
"""
import time as time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pylab import e
from pylab import pi
"""Setting variables and arrays...

Using units of Parsec, Solar masses and strange units of Time (~14.9yrs). Allows gravitational constant to be equal to one.
Conversions: 
Time:       1st = 14909319.84 yrs
            1s = 2.125352995 e-15 t
Distance:   1km = 3.24079289 e-14 parsec
Speed:      1km/s = 15.24819325 parsec/st
Density     1kg/m-3 = 1.477456948 e19 M*/parsec-3
"""
#time.clock()
StartTime = time.clock()
global AccelerationTime
AccelerationTime = 0.
VerletTime = 0.

#Constants
kms = 15.24819325
kgm3 = 1.477456948e19

G = 1 #Parsec^3 yr^-2 M0^-1
p = 10 #No. of particles.
tolerance = 0.01 #Timestep control tolerance
InitialSpeed = 20 #km/s
Cs = 0.35*kms #Sound speed km/s
a = 0.5 #Accretion column parameter (0.5~1)


clouddensity = 100 *(3*1.6726219e-27*10**6)*kgm3 #Density: cloud ptle no., He/H, H+ mass, m^3, Conversion
ISMdensity = 5 *(1.4*1.6726219e-27*10**6)*kgm3
E = (3/(4*pi) * (100/clouddensity) )**(1./3) #Softening Coefficent.
GMcentral = G*86321.5 #Gravitational central constant
Time = 0.
Tmax = 6.71
dTmax = 0.01 #3pc at 

#M=0
#reqspeed = np.sqrt(  (2*G*M/( (3*M/(4*pi*ISMdensity))**(1./3))  )-Cs**2)

#Box limits
Xmin = 0
Ymin = 0
Xmax = 100
Ymax = 100
Zscale = 50

#3D plotting
#fig = plt.figure()
#fig.clf()
#ax = fig.add_subplot(111,projection='3d')

#Arrays
particle =  np.zeros([10,p]) #Xn,Yn,Zn,  Vx,Vy,Vz,  fx,fy,fz,  Mass
xold = np.zeros([3,p]) #Storage array for last position values. (Add to particle array?)
KE = np.zeros([p])
TE = np.zeros([p])
E = np.zeros([p])

#PE defined in Acceleration function.
#speed = np.zeros([p,672])
Speed = np.zeros(p)
Acceleration =  np.zeros(p)
TotalList = np.arange(0,p,1)
ActiveList = TotalList[TotalList>=0]
MaxActive = len(ActiveList)

#################################################################################################################

"""Inital conditions (t = 0)"""
#Position
particle[0,:] = np.random.rand(p)*Xmax
particle[1,:] = np.random.rand(p)*Ymax
particle[2,:] = ((np.random.rand(p)*2)-1)*Zscale

#Velocity
Velocity = np.random.normal(0,20/np.sqrt(3),(3,p))
xdrift = np.average(Velocity[0,:])
ydrift = np.average(Velocity[1,:])
zdrift = np.average(Velocity[2,:])

Velocity[0,:] -= xdrift
Velocity[1,:] -= ydrift
Velocity[2,:] -= zdrift

particle[3,:] = Velocity[0,:]*kms
particle[4,:] = Velocity[1,:]*kms
particle[5,:] = Velocity[2,:]*kms


print(np.std(particle[3,:]))
print(np.std(particle[4,:]))
print(np.std(particle[5,:]))
#for i in range(p):
#    speed[i] = (np.sqrt(np.sum(particle[3:6,i]**2)))
#plt.hist(speed,50)
#plt.show()
print(np.average(speed))

#Mass
particle[9,:] = 100
E[:] = (3/(4*pi) * (particle[9,:]/clouddensity) )**(1./3)


InitialConditionTime = time.clock() - StartTime
#################################################################################################################  
    
def CalTotalAcc(_):   
    """Particle acceleration (due to gravity) calculation. Applies for all particles given"""    
    AccelerationStartTime = time.clock()
    #Collision check -------
    particle[6:9,:] = 0 #Cleaning accelerations.
    PE = np.zeros(MaxActive)    #Cleaning Potentials
    #dT = np.zeros([MaxActive-1,2])
    R = np.zeros([p,p])
    for A_i in range(0,MaxActive-1):
        for A_j in range(A_i+1,MaxActive):
            i = ActiveList[A_i]
            j = ActiveList[A_j]
            #Calculating distances.
            dX = particle[0,i]-particle[0,j]
            dXb = BoundaryDistance(dX,Xmax,Xmin)
            dY = particle[1,i]-particle[1,j]
            dYb = BoundaryDistance(dY,Ymax,Ymin)
            dZ = particle[2,i]-particle[2,j]
            
            #Choosing the shorter distance...
            if abs(dX) >= abs(dXb):
                dX = dXb
            if abs(dY) >= abs(dYb):
                dY = dYb
            
            #Acceleration Calaculation
            R2 = (dX**2+dY**2+dZ**2)
            R[i,j] = np.sqrt(R2) 
            #R[j,i] = np.sqrt(R2) #Not actually needed.
            E[i] = (3/(4*pi) * (particle[9,i]/clouddensity) )**(1./3)
            E[j] = (3/(4*pi) * (particle[9,j]/clouddensity) )**(1./3)
            
            K= G/(R[i,j]+(E[i]+E[j])/2)**3 #G/R^3 (Avoids using force to speed up code - no extra divisions)
            #if (E[i] + E[j]) > R:
            #    print("Collision between %d and %d" %(i,j))
            #    print(np.sqrt(2*E[i]*tolerance/(K*R*particle[9,j])))
                
            #Adding changes in acceleration due to particles...
            particle[6,i] = particle[6,i]-K*particle[9,j]*dX #-G/R^3 * M * r_x
            particle[7,i] = particle[7,i]-K*particle[9,j]*dY 
            particle[8,i] = particle[8,i]-K*particle[9,j]*dZ #-GMcentral*particle[2,i]/(particle[2,i]+Zscale)**3 #Central potential
            
            particle[6,j] = particle[6,j]+K*particle[9,i]*dX #Opposite dX vector,results in positive total.
            particle[7,j] = particle[7,j]+K*particle[9,i]*dY
            particle[8,j] = particle[8,j]+K*particle[9,i]*dZ #-GMcentral*particle[2,i]/(particle[2,i]+Zscale)**3
            
            #if (E[i] + E[j]) > R:
                #print("Collision between %d and %d" %(i,j))
                #MergeMass = particle[9,i] + particle[9,j]
                #MergeSpeed = (particle[9,i]*particle[3:6,i] + particle[9,j]*particle[3:6,j])/MergeMass
                #MergeAcceleration = 
                
            #dT[i,0] = min(dTmax,np.sqrt(2*E[i]*tolerance/(K*R*particle[9,j])))
            #dT[i,1] = min(dTmax,np.sqrt(2*E[j]*tolerance/(K*R*particle[9,i])))

            #Energy calculations
            Potential = -G*particle[9,i]*particle[9,j]/R
            PE[i] = PE[i] + Potential
            PE[j] = PE[j] + Potential
    
    #KE & total calculation
    KE = 0.5*particle[9,:]*(particle[3,:]**2+particle[4,:]**2+particle[5,:]**2)
    TE = KE+PE
    #print(np.sum(TE))
    #plt.plot(np.sum(TE))
    
    global AccelerationTime
    AccelerationTime += (time.clock() - AccelerationStartTime)
    return particle,KE,PE
    
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

def ZEscape(p):
    """Particle escape function"""
    particle = p
    if abs(particle)>200:
        return 0
    else:
        return particle

    
def BoundaryDistance(d,Max,Min):
    """Acquires the distance from two points going through the boundaries"""
    distance = d
    if distance < 0:
        return distance + (Max-Min)
    else:
        return distance -(Max-Min)
        
def Collide(i,j):
    """On successful collision. Merges mass and calculates new particle speed from particle momentum.(Planned at end of loop)"""
    MergeMass = particle[9,i] + particle[9,j]
    MergeSpeed = (particle[9,i]*particle[3:6,i] + particle[9,j]*particle[3:6,j])/MergeMass
    MergeAcceleration = (particle[9,i]*particle[6:9,i] + particle[9,j]*particle[6:9,j])/MergeMass
    #Do something about Xold (old accelerations)
    particle[9,i] = MergeMass
    particle[3:6,i] = MergeSpeed
    particle[6:9,i] = MergeAcceleration
    
    TotalList[j] = -1
    return 0
    
def TimestepControl(dTold):
    """Calculates the timestep for all particles as a function of speed, acceleration and previous timestep.(Planned at start)"""
    dT = np.zeros([MaxActive])
    for A_i in range(0,MaxActive):
        i = ActiveList[A_i]
        #Aqcuiring magnitudes
        Speed[i] = np.sqrt(np.sum(particle[3:6,i]**2))
        Acceleration[i] = np.sqrt(np.sum(particle[6:9,i]**2))
        #Timestep calculation with previous velocity.
        dT[A_i] = (-Speed[i] + np.sqrt(Speed[i]**2+2*Acceleration[i]*tolerance*E[i]) )/Acceleration[i]
    return min(dTmax,np.min(dT))
        
        
#Velocity Verlet function

particle,KE,PE = CalTotalAcc(particle)
dT = dTmax
n = 0 #Iteration counter
while Time < Tmax:
    VerletStartTime = time.clock()
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
        
        #Bondi-Hoyle
        Speed = np.sqrt(np.sum(particle[3:6,i]**2))
        #speed[i,n] = Speed/kms
        R_BH = 2*G*particle[9,i]/(Speed**2+Cs**2)
        #particle[9,i] = particle[9,i] + a*pi*ISMdensity*Speed*R_BH**2*dT
        if R_BH > E[i]:
            particle[9,i] = particle[9,i] + a*pi*ISMdensity*Speed*R_BH**2*dT
        else:
            particle[9,i] = particle[9,i] + a*pi*ISMdensity*Speed*E[i]**2*dT
            if a*pi*ISMdensity*Speed*E[i]**2*dT/particle[9,i] > 0.05:
                print("PANIC-E %.d" %i)
                print(Speed)
                print(particle[0:3,i])
                
        
        
    #Updating Acceleration (fn+1)
    VerletTime += ( time.clock() - VerletStartTime)
    
    particle,KE,PE = CalTotalAcc(particle)

    
    VerletStartTime = time.clock()
    for i in range(0,p):
        #Calculating resulting velocity : Vn+1 = Vn + 0.5*(fn + fn+1)*dT
        particle[3,i] = particle[3,i] + 0.5*(xold[0,i]+particle[6,i])*dT
        particle[4,i] = particle[4,i] + 0.5*(xold[1,i]+particle[7,i])*dT
        particle[5,i] = particle[5,i] + 0.5*(xold[2,i]+particle[8,i])*dT
        
        
        """Things to plot"""
        #if n % 50 == 0:
            #ax.scatter(particle[0,0],particle[1,0],particle[2,0],c='b')
            #ax.scatter(particle[0,1],particle[1,1],particle[2,1],c='r')
            #ax.scatter(particle[0,2],particle[1,2],particle[2,2],c='y')
            #ax.scatter(particle[0,3],particle[1,3],particle[2,3],c='g')
            #ax.scatter(particle[0,4],particle[1,4],particle[2,4],c='k')
        #plt.plot(particle[0,1],particle[1,1],'b.')
        #plt.plot(particle[0,0],particle[1,0],'y.')
        #plt.plot(particle[0,2],particle[1,2],'r.')
        #plt.plot(t,(particle[0,1]**2+particle[1,1]**2)-(particle[0,2]**2+particle[1,2]**2),'k.')
        #plt.plot(particle[0,3],particle[1,3],'r.')
            
    
    #Updating particle list
    ActiveList = TotalList[TotalList>=0] 
    MaxActive = len(ActiveList)
    #Timings & Counters
    n = n + 1
    Time = Time + dT
    VerletTime += ( time.clock() - VerletStartTime)
    

#GraphFmt("","$y$ (AU)","$x$ (AU)",Xmin,Xmax,Ymin,Ymax)
#GraphFmt("","% of Total Energy","Time (yr)",ymin=0,ymax=1)
print(n)

EndTime = time.clock() - StartTime
print(EndTime)
print(AccelerationTime/EndTime)
print(VerletTime/EndTime)
print(InitialConditionTime/EndTime)
print([particle[9,:]])
#plt.savefig("EarthSunMarsEnergy")
#plt.show()

    