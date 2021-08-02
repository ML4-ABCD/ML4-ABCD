from Particle import Particle
import math
import random
import numpy as np
import matplotlib.pyplot as plt

def simulateDecay(m1, m2, M):
    phi = random.uniform(-math.pi, math.pi)
    theta = random.uniform(0.02, math.pi-0.02)
    
    pmag1 = math.sqrt(((M**2 + m1**2 - m2**2)**2 - 4*M**2*m1**2))/(2*M)
    phi1 = phi
    theta1 = theta
    
    pmag2 = math.sqrt(((M**2 + m2**2 - m1**2)**2 - 4*M**2*m2**2))/(2*M)  
    phi2 = phi + math.pi
    if phi2 > 2*math.pi:
        phi2 -= 2*math.pi
    elif phi2 < -2*math.pi:
        phi2 += 2*math.pi
    theta2 = math.pi - theta

    particle1 = Particle()
    particle2 = Particle()
    particle1.setMPmagPhiTheta(m1,pmag1,phi1,theta1)
    particle2.setMPmagPhiTheta(m2,pmag2,phi2,theta2)
    return particle1, particle2

def createEvent(m1, m2, M, eType):
    particle1, particle2 = simulateDecay(m1, m2, M)
    parent=Particle()
    parent.setMPmagPhiTheta(M, 0.0, 0.0, math.pi/2)

    bx, by, bz = 0.0, 0.0, 0.0
    particle1.boost(bx, by, bz)
    particle2.boost(bx, by, bz)
    parent.boost(bx, by, bz)

    particle14v=particle1.getFourVector()
    particle24v=particle2.getFourVector()
    parent4v=parent.getFourVector()

    event=[]
    event.extend(parent4v)
    event.extend(particle14v)
    event.extend(particle24v)
    event.extend(eType)
    return event

def main():
    ###########################
    #Define simulation constants
    ###########################
    signalPoleMass = 91.1876
    m1 = 0.1056583745
    m2 = m1
    nEvents = 100000
    xsecLumiBg = 1e5
    xsecLumiSg = 5e2
    
    ###########################
    # Simulate events, fill background and signal events. Need to optimize
    ###########################
    background = []
    signal = []
    for i in range(0,nEvents):
        if i % 10000 == 0: print("Simulated {} events".format(i))
        bMass = random.expovariate(0.01) # in GeV
        if bMass < m1 + m2: continue
        background.append(createEvent(m1, m2, bMass, "0"))
    
        sMass = random.gauss(signalPoleMass,2) # in GeV  
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
    
        plt.hist(sg,  alpha=0.9, histtype='step', lw=2, label=None, log=True, weights=sWeight)
        plt.hist(bg,  alpha=0.9, histtype='step', lw=2, label=None, log=True, weights=bWeight)
        plt.hist(tot, alpha=0.9, histtype='step', lw=2, label=None, log=True, weights=tWeight)
        fig.savefig("trainVar{}.png".format(i), dpi=fig.dpi) 

if __name__ == '__main__':
    main()
