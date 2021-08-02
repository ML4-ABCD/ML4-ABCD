from Particle import Particle
import math
import random
import numpy as np
import matplotlib.pyplot as plt

def simulateDecay(m1, m2, M):
    # Determine the phi and theta randomly of one of the daughters
    phi = random.uniform(-math.pi, math.pi)
    theta = random.uniform(0.02, math.pi-0.02)
    
    # Set the total momentum, phi, and theta for daughter one based on momentum conservation
    pmag1 = math.sqrt(((M**2 + m1**2 - m2**2)**2 - 4*M**2*m1**2))/(2*M)
    phi1 = phi
    theta1 = theta
    
    # Set the total momentum, phi, and theta for daughter two based on momentum conservation
    pmag2 = math.sqrt(((M**2 + m2**2 - m1**2)**2 - 4*M**2*m2**2))/(2*M)  
    phi2 = phi + math.pi
    if phi2 > 2*math.pi:
        phi2 -= 2*math.pi
    elif phi2 < -2*math.pi:
        phi2 += 2*math.pi
    theta2 = math.pi - theta

    # Make 4-vector objects for each particle
    particle1 = Particle()
    particle2 = Particle()
    particle1.setMPmagPhiTheta(m1,pmag1,phi1,theta1)
    particle2.setMPmagPhiTheta(m2,pmag2,phi2,theta2)
    return particle1, particle2

def findRandomBoost(mMax=None, eMax = 5000):
    b2 = 999.9
    bx, by, bz = None, None, None
    #b2Max = 1-(mMax**2/eMax**2) if mMax else 1.0 #1 - m**2/E**2    
    b2Max = 0.9
    while abs(b2) >= b2Max:
        bx = random.uniform(-b2Max, b2Max)
        by = random.uniform(-b2Max, b2Max)
        bz = random.uniform(-b2Max, b2Max)     
        b2 = bx**2 + by**2 + bz**2
    return bx, by, bz

def createEvent(m1, m2, M, eType, mMax=None):
    # Simulate 2-Body decay in the CM frame
    particle1, particle2 = simulateDecay(m1, m2, M)
    parent=Particle()
    parent.setMPmagPhiTheta(M, 0.0, 0.0, math.pi/2)

    # Boost randomly
    bx, by, bz = findRandomBoost(mMax)
    particle1.boost(bx, by, bz)
    particle2.boost(bx, by, bz)
    parent.boost(bx, by, bz)

    # Get an array of the 4-vectors
    particle14v=particle1.getFourVector()
    particle24v=particle2.getFourVector()
    parent4v=parent.getFourVector()

    # Save all the 4-vectors used in the event
    event=[]
    event.extend(np.array([parent.M()]));    event.extend(parent4v)
    event.extend(np.array([particle1.M()])); event.extend(particle14v)
    event.extend(np.array([particle2.M()])); event.extend(particle24v)
    event.extend(eType)
    return event

def main():
    ###########################
    #Define simulation constants
    ###########################
    signalPoleMass = 91.1876
    m1 = 0.1056583745
    m2 = m1
    mMax = 500
    bMassLambda = 0.01
    nEvents = 500000
    xsecLumiBg = 1e5
    xsecLumiSg = 5e2
    
    ###########################
    # Simulate events, fill background and signal events. Need to optimize
    ###########################
    background = []
    signal = []
    for i in range(0,nEvents):
        if i % 10000 == 0: print("Simulated {} events".format(i))
        bMass = random.expovariate(bMassLambda) # in GeV
        if bMass < m1 + m2 or bMass > mMax: continue
        background.append(createEvent(m1, m2, bMass, "0", mMax))
    
        sMass = random.gauss(signalPoleMass,1) # in GeV  
        signal.append(createEvent(m1, m2, sMass, "1"))
    background = np.array(background, dtype=float)
    signal = np.array(signal, dtype=float)
    
    ###########################
    # Make event weight arrays
    ###########################
    nEventsGen = len(background)
    bWeight = np.empty(nEventsGen)
    sWeight = np.empty(nEventsGen)
    bWeight.fill(xsecLumiBg/nEventsGen)
    sWeight.fill(xsecLumiSg/nEventsGen)
    
    ###########################
    # Save output to npz file
    ###########################
    newoutput='absolutepath'
    np.save(newoutput, [signal, background])
    
    ###########################
    # Plot training variables
    ###########################
    import matplotlib.pyplot as plt 

    for i in range(signal.shape[1]):
        fig = plt.figure()
        sg = signal[:,i]
        bg = background[:,i]
        tot = np.concatenate((sg, bg))
        tWeight = np.concatenate((sWeight, bWeight))
    
        #plt.hist(sg,  bins=75, range=(0,150), alpha=0.9, histtype='step', lw=2, label=None, log=False, weights=sWeight)
        #plt.hist(bg,  bins=75, range=(0,150), alpha=0.9, histtype='step', lw=2, label=None, log=False, weights=bWeight)
        #plt.hist(tot, bins=75, range=(0,150), alpha=0.9, histtype='step', lw=2, label=None, log=False, weights=tWeight)
    
        plt.hist(sg,  bins=100, alpha=0.9, histtype='step', lw=2, label=None, log=True, weights=sWeight)
        plt.hist(bg,  bins=100, alpha=0.9, histtype='step', lw=2, label=None, log=True, weights=bWeight)
        plt.hist(tot, bins=100, alpha=0.9, histtype='step', lw=2, label=None, log=True, weights=tWeight)
        fig.savefig("trainVar{}.png".format(i), dpi=fig.dpi) 

if __name__ == '__main__':
    main()
