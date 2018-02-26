# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 11:15:54 2016

@author: Kevin
"""
from math import tanh
import sys
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
Accel:      1km/s^2 = 7174458613579613 parsec/st^2
Density     1kg/m-3 = 1.477456948 e19 M*/parsec-3
"""

#Program timing
StartTime = time.clock()
global AccelerationTime
AccelerationTime = 0.
VerletTime = 0.

#Conversions
kms = 15.24819325
kms2 = 7174458613579613
kgm3 = 1.477456948e19
st = 14909319.84

#Constants
G = 1 #Parsec^3 yr^-2 M0^-1
p = 1500 #No. of particles.
InitialP = 863 #No. of particles.
tolerance = 0.5 #Timestep control tolerance
MassTolerance = 0.1 #Maximum accretion amount.
InitialSpeed = 20 #km/s
Cs = 0.995*kms #Sound speed km/s
a = 0.5 #Accretion column parameter (0.5~1)
ISMReturn = 0.65 #Fraction of mass from supernova that is returned to the ISM.
Eff = 0.05 #Precentage of cloud converted to stars per freefall time.

clouddensity = 100 *(3*1.6726219e-27*10**6)*kgm3 #Density: cloud ptle no., He/H, H+ mass, m^3, Conversion
ISMdensity = 5 *(1.4*1.6726219e-27*10**6)*kgm3
Tff = np.sqrt(3*pi/(32*clouddensity))

Time = 0.
Tmax = 10.1
dTmax = 0.02

#Box limits
Xmin = 0
Ymin = 0
Xmax = 100
Ymax = 100
Zscale = 25

#Arrays
particle =  np.zeros([10,p]) #Xn,Yn,Zn,  Vx,Vy,Vz,  ax,ay,az,  Mass
xold = np.zeros([3,p]) #Storage array for last acceleretion values. (Add to particle array?)
KE = np.zeros([p])
E = np.zeros([p])
Countdown = np.zeros(p)
Speed = np.zeros(p)
Acceleration =  np.zeros(p)
List = np.arange(0,p,1)
CountdownList = np.zeros(p)
CollisionList = np.zeros(p)
CollidedBefore = np.zeros(p) #Check to test if the particle has collided in it's lifetime.
TotalList = np.arange(0,p,1)
SFM = np.zeros(p) #Amount of star mass within a star.
SFE = np.zeros(p) #Fraction of star mass to cloud mass.

#Global variables (For data dump)
global BondiGain 
BondiGain = 0.
global EGain 
EGain = 0.
global CollisionGain 
CollisionGain = 0.
global SFMGain
SFMGain = 0.
global ZLoss
ZLoss = 0.
global SupernovaLoss
SupernovaLoss = 0.
global BlackholeLoss
BlackholeLoss = 0.

global NumCollision
NumCollision = 0.
global NumSupernova
NumSupernova = 0.

#################################################################################################################

"""Inital conditions (t = 0)"""
#Position
particle[0,:InitialP] = np.random.rand(InitialP)*Xmax
particle[1,:InitialP] = np.random.rand(InitialP)*Ymax
particle[2,:InitialP] = ((np.random.rand(InitialP)*2)-1)*Zscale

#Velocity
Velocity = np.random.normal(0,20/np.sqrt(3),(3,InitialP))
xdrift = np.average(Velocity[0,:])
ydrift = np.average(Velocity[1,:])
zdrift = np.average(Velocity[2,:])

Velocity[0,:] -= xdrift
Velocity[1,:] -= ydrift
Velocity[2,:] -= zdrift

particle[3,:InitialP] = Velocity[0,:]*kms
particle[4,:InitialP] = Velocity[1,:]*kms
particle[5,:InitialP] = Velocity[2,:]*kms

#Mass
particle[9,:InitialP] = 100
E[:InitialP] = (3/(4*pi) * (particle[9,:InitialP]/clouddensity) )**(1./3)

#Removing excess particles
TotalList[InitialP:] = -1
ActiveList = TotalList[TotalList>=0] 
MaxActive = len(ActiveList)

InitialConditionTime = time.clock() - StartTime
#################################################################################################################  
"""FUNCTIONS"""

def StarLT(M):
    """Returns estimated star lifetime. No shorter than 2 Myrs."""
    return max((1e10*M**-2.5)/st,2000000/st)
    
def ListUpdate(TotalList):
    ActiveList = TotalList[TotalList>=0] 
    MaxActive = len(ActiveList)
    return ActiveList,MaxActive
    
def RemoveParticle(i):
    particle[:,i] = 0
    xold[:,i] = 0
    CountdownList[i] = 0
    Countdown[i] = 0
    TotalList[i] = -1
    SFM[i] = 0
    
    
def SNF(CloudMass,SFE):
    """Relationship used to model supernova masses. a*CloudMass^b + c. 'a' is a function of SFE, in the form of a'*SFE^b'.
    SFE in this context refers to the fraction/precentage of cloud mass that is stars.
    a' = 0.24337061
    b' = 0.74382654
    b  = 0.74382654
    c  = -1.00611354
    All values were determined using a curvefit function on the 'required solar mass' integral."""
    return ((0.24337061*SFE**0.7438265)*CloudMass**0.74382654 -1.00611354)
    
def SNCheck(i): 
    """Aqcuires and updates countdown of a particle. DOES NOT DO TIMESTEP. Lowest update possible is 2Myr from StarLT."""
    StarMass = SNF(particle[9,i],SFE[i])
    if StarMass < 8:
        return 0
    StarLifetime = StarLT(StarMass)
    if Countdown[i] <= 0:
        Countdown[i] = StarLifetime
        CountdownList[i] = 1
        print("Countdown started for particle %d (T:%4f, C:%.4f)." %(i,Time,Countdown[i]))
        print(particle[9,i])
        #f3 - Initial high mass star data. loop no, particle no, Collision history, SFE, Mass.
        f3.write("\n%.d \n%.d \n%.d \n%.4f \n%.2f" %(n,i,CollidedBefore[i],SFE[i],particle[9,i]))
    else:
        Countdown[i] = min(StarLifetime,Countdown[i])
        
def Supernova(i):
    """Explodes a particle in a supernova with child particles of mass at least 100.
    Velocity vectors given at random but with a total kinetic energy of 10^44 J"""
    CountdownList[i] = 0
    SFM[i] = 0
    
    """
    #Optional Blackhole code.
    if particle[9,i] > 22000:
        print("Particle %d has turned into a black hole! [%d]" %(i,MaxActive-1) )
        global BlackholeLoss
        BlackholeLoss += particle[9,i]
        RemoveParticle(i)
        return 0
    """
    
    CPN = int(np.floor(particle[9,i]*(1-ISMReturn)/100)) #Child Particle Number
    CPUV = UnitVector(CPN)
    
    print("Particle %d has exploded into %d particles! (T:%.4f) [%d]" %(i,CPN,Time,MaxActive+CPN-1))
    print(particle[9,i])
    #f2 - Supernova data. Loop Num, Supernova particle num, collision history, Mass, SFE
    f2.write("\n%.d \n%.d \n%.d \n%.2f \n%.4f" %(n,i,CollidedBefore[i],particle[9,i],SFE[i]))    
    
    #Child particles given same format as other particles.
    CP = np.zeros([10,CPN]) 
    CP[0:3,:] = CPUV*(1+(0.02*CPN))*E[i] #Arb. factor (prevent sudden collision)
    CP[0,:] += particle[0,i] #Unable to broadcast as a array. Doing individually.
    CP[1,:] += particle[1,i]
    CP[2,:] += particle[2,i]
    
    MaxVelo = np.sqrt(2e44/(particle[9,i]*1.99e30))/1000. #Maximum velocity possible from 10^51 ergs, kms.
    print(MaxVelo)
    CP[3:6,:] = CPUV*MaxVelo*kms
    CP[3,:] += particle[3,i] #Inherits parent speed.
    CP[4,:] += particle[4,i]
    CP[5,:] += particle[5,i]
    #Accelerations set as zero.
    CP[9,:] = particle[9,i]*(1-ISMReturn)/CPN
    
    global SupernovaLoss
    SupernovaLoss += particle[9,i]*(ISMReturn)
    global NumSupernova
    NumSupernova += 1
    CollidedBefore[i] = 0
    
    #Setting Lists - Aqcuire availible slots and reactivate them.
    Availible = List[TotalList<0] 
    Availible = Availible[:CPN-1] #Reusing current index. -> 'CPN-1'
    TotalList[Availible] = List[Availible]
    
    particle[:,i] = CP[:,0] 
    particle[:,Availible] = CP[:,1:]
    xold[0,Availible] = particle[6,i] #Unable to broadcast as a array.
    xold[1,Availible] = particle[7,i]
    xold[2,Availible] = particle[8,i]

    print(List[Availible])
    
def UnitVector(n):
    """Returns 'n' unit vectors in a random directions in 3D"""
    xyz = np.random.rand(3,n)*2-1
    for i in range(0,n):
        normalisation = 1./np.sqrt(np.sum(xyz[:,i]**2))
        xyz[:,i] = xyz[:,i]*normalisation
    return xyz
    
def CalTotalAcc(_):   
    """Particle acceleration (due to gravity) calculation. Applies for all particles given"""    
    AccelerationStartTime = time.clock()
    particle[6:9,:] = 0 #Cleaning accelerations.
    R = np.zeros([p,p])
    for A_i in range(0,MaxActive-1): #Dummy index to allow for two synced arrays
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
            if (E[i] + E[j]) > R[i,j]:
                CollisionList[i] = j
                
            #Adding changes in acceleration due to particles...
            particle[6,i] = particle[6,i]-K*particle[9,j]*dX #-G/R^3 * M * r_x
            particle[7,i] = particle[7,i]-K*particle[9,j]*dY 
            particle[8,i] = particle[8,i]-K*particle[9,j]*dZ +ZGrav(particle[2,i]) #Central potential
            
            particle[6,j] = particle[6,j]+K*particle[9,i]*dX #Opposite dX vector,results in positive total.
            particle[7,j] = particle[7,j]+K*particle[9,i]*dY
            particle[8,j] = particle[8,j]+K*particle[9,i]*dZ +ZGrav(particle[2,j])
    
    global AccelerationTime
    AccelerationTime += (time.clock() - AccelerationStartTime)
    return particle,KE
    
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

def ZEscape(i):
    """Particle escape function. Limit set as 175 parsecs"""
    if abs(particle[2,i])>175:
        global ZLoss
        ZLoss += particle[9,i]
        RemoveParticle(i)
        print("Particle %d has left the system. [%d] (T:%.4f)" %(i,MaxActive-1,Time))
        
def ZGrav(z):
    """External gravitational potential from SILCC."""
    return -4*pi*G*ISMdensity*2*150*tanh(z/(2*150.)) 
    
def BoundaryDistance(d,Max,Min):
    """Acquires the distance from two points going through the boundaries"""
    distance = d
    if distance < 0:
        return distance + (Max-Min)
    else:
        return distance -(Max-Min)
        
def Collide(i,j):
    """On successful collision. Merges mass and calculates new particle speed from particle momentum."""
    MergeMass = particle[9,i] + particle[9,j]
    Merge = (particle[9,i]*particle[0:9,i] + particle[9,j]*particle[0:9,j])/MergeMass
    OldAccMerge = (particle[9,i]*xold[:,i]+particle[9,j]*xold[:,j])/MergeMass
    
    global CollisionGain
    CollisionGain += min(particle[9,i],particle[9,j])
    global NumCollision
    NumCollision += 1
    
    particle[9,i] = MergeMass
    particle[0:9,i] = Merge
    xold[:,i] = OldAccMerge
    if CountdownList[i] == 0:
        CountdownList[i] = CountdownList[j]
        Countdown[i] = Countdown[j]
    if Countdown[j] > 0 :
        Countdown[i] = min(Countdown[i],Countdown[j])
    
    RemoveParticle(j)
    CollidedBefore[i] += CollidedBefore[j] +  1
    
def TimestepControl():
    """Calculates the timestep for all particles as a function of speed and acceleration."""
    dT = np.zeros([MaxActive])
    dTdM = np.zeros([MaxActive])
    for A_i in range(0,MaxActive):
        i = ActiveList[A_i]
        #Aqcuiring magnitudes - WIll NEED FOR LATER
        SpeedSqr = np.sum(particle[3:6,i]**2)
        Speed[i] = np.sqrt(SpeedSqr)
        Acceleration[i] = np.sqrt(np.sum(particle[6:9,i]**2))
        #Timestep calculation with previous velocity.
        dT[A_i] = (-Speed[i] + np.sqrt(Speed[i]**2+2*Acceleration[i]*tolerance*E[i]) )/Acceleration[i]
        dTdM[A_i] = MassTolerance*particle[9,i]/(a*pi*ISMdensity*Speed[i]*(2*G*particle[9,i]/SpeedSqr)**2)
    return min(dTmax,np.min(dT),np.min(dTdM))
        
def Accretion(i):
    """Accretion due to Bondi-Hoyle or from cloud radius (which ever is larger). Accreted mass is then placed at the rear of 
    the particle and centre of mass + mommentum calculations are performed, simultaing a collision between the two.
    """ 
    VCs = Speed[i]**2+Cs**2
    R_BH = 2*G*particle[9,i]/(VCs)
    dM = a*pi*ISMdensity*np.sqrt(VCs)*max(E[i],R_BH)**2*dT #a: Accretion column parameter (0.5-1)"
    
    if E[i]<R_BH:
        global BondiGain
        BondiGain += dM
    else:
        global EGain
        EGain += dM

    #Alert code
    if dM/particle[9,i] > 0.05:
        print("HELP: %d" %i)
        if E[i]>R_BH:
            print("E")
            print(Speed[i])
        else:
            print("R_BH")
            print(Speed[i])
            
        
    NewMass = particle[9,i] + dM
    ParticleRear = (-particle[3:6,i]/Speed[i])*E[i] #Vector pointing to the rear from the centre of the cloud.
    
    particle[0:3,i] = (particle[0:3,i]*particle[9,i] + (particle[0:3,i]+ParticleRear)*dM)/NewMass #CoM
    particle[3:6,i] = particle[9,i]*particle[3:6,i]/NewMass #Momentum conversation
    particle[9,i] = NewMass    
    
#################################################################################################################

"""MAIN VELOCITY VERLET INTEGRATION LOOP"""

particle,KE = CalTotalAcc(particle) #Acceleration calculation required for Verlet

dT = dTmax
n = 0 #Iteration counter

#Data dump files
f= open("ParticleBoxDump.txt","w")
f2 = open("PBSupernovaHist.txt","w")
f3 = open("PBStarHist.txt","w")
while Time < Tmax:
    VerletStartTime = time.clock()
    CollisionList = np.zeros(p)
    dT = TimestepControl()
    #For exterme Bondi-Hoyle accretion
    if dT < 6.707e-07:
        print("TIMESTEP IS CRITICALLY LOW - ABORTING")
        sys.exit()
    
    #Supernova check & countdowns
    for i in List[CountdownList==1]:
        Countdown[i] -= dT #Must occur before SNCheck.
        if Countdown[i] <= 0:
            Supernova(i)
    ActiveList,MaxActive = ListUpdate(TotalList)
    
    #Position integration
    for i in ActiveList:
        #Star formation
        if particle[9,i] > 132.123: #Minimum mass for column density 10^21
            dSFM = (Eff*particle[9,i])*(dT/Tff)
            SFMGain += dSFM
            SFM[i] += dSFM #Star formation Mass. 1% converted per freefall time.
            SFE[i] =  SFM[i]/particle[9,i]
            SNCheck(i)
            
        #Verlet step (Position) : Rn+1 = Rn + Vn*dT + 0.5*an*dT^2
        particle[0,i] = particle[0,i] + particle[3,i]*dT + 0.5*particle[6,i]*dT**2
        particle[1,i] = particle[1,i] + particle[4,i]*dT + 0.5*particle[7,i]*dT**2
        particle[2,i] = particle[2,i] + particle[5,i]*dT + 0.5*particle[8,i]*dT**2
        #Position boundaries
        particle[0,i] = PosBoundary(particle[0,i],Xmax,Xmin)
        particle[1,i] = PosBoundary(particle[1,i],Ymax,Ymin)
        #Temporary storage array (Holding an)
        xold[:,i] = particle[6:9,i]
        ZEscape(i)
    
    #Updating Acceleration (an+1)
    VerletTime += ( time.clock() - VerletStartTime)
    
    ActiveList,MaxActive = ListUpdate(TotalList)  
    particle,KE = CalTotalAcc(particle)
    
    VerletStartTime = time.clock()
    
    #Velocity integration
    for i in ActiveList:
        #Calculating resulting velocity : Vn+1 = Vn + 0.5*(an + an+1)*dT
        particle[3,i] = particle[3,i] + 0.5*(xold[0,i]+particle[6,i])*dT
        particle[4,i] = particle[4,i] + 0.5*(xold[1,i]+particle[7,i])*dT
        particle[5,i] = particle[5,i] + 0.5*(xold[2,i]+particle[8,i])*dT
        
        #Calculations for accretion
        Acceleration[i] = np.sqrt(np.sum(particle[6:9,i]**2))
        Speed[i] = np.sqrt(np.sum(particle[3:6,i]**2))
        Accretion(i)
    
    #Collision loop
    temp = TotalList[CollisionList>0]
    for i in temp[::-1]:
        Collide(i,CollisionList[i])
        print("Collision between %d and %d. [%d]" %(i,CollisionList[i],MaxActive-1)) #Consider removing collision list to remove excess arrays.
    ActiveList,MaxActive = ListUpdate(TotalList)
    
    #Timings & Counters
    n = n + 1
    Time = Time + dT
    VerletTime += ( time.clock() - VerletStartTime)
    
    #Filedumping
    f.write("\n%s \n%s \n%s \n" %(Time,MaxActive,np.sum(CountdownList)))
    f.write("%s \n%s \n%s \n" %(BondiGain,EGain,CollisionGain))
    f.write("%s \n%s \n%s \n" %(ZLoss,SupernovaLoss,BlackholeLoss))
    f.write("%s \n%s \n" %(SFMGain,np.average(SFE)))
    f.write("%s \n%s \n" %(NumCollision,NumSupernova))
    np.savetxt(f,SFM[ActiveList])
    f.write("\n")
    np.savetxt(f,particle[0,ActiveList])
    f.write("\n")
    np.savetxt(f,particle[1,ActiveList])
    f.write("\n")
    np.savetxt(f,particle[2,ActiveList])
    f.write("\n")
    np.savetxt(f,particle[9,ActiveList])

#################################################################################################################
"""Extra prints and data dumping"""

f.write("\n")
np.savetxt(f,CollidedBefore)
f.write("\n\n%.d" %(n))
print(n)

EndTime = time.clock() - StartTime
print(EndTime)
print(AccelerationTime/EndTime)
print(VerletTime/EndTime)
print(InitialConditionTime/EndTime)
print([particle[9,:InitialP]])
f.close()
f2.close()
f3.close()
