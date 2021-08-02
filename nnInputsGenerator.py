import math
import random
import numpy as np
import matplotlib.pyplot as plt

class Particle:
    def __init__(self, E=None, px=None, py=None, pz=None):
        self.E_ = E
        self.px_ = px
        self.py_ = py
        self.pz_ = pz

    def E (self): return self.E_
    def Px(self): return self.px_
    def Py(self): return self.py_
    def Pz(self): return self.pz_
    def P (self): return math.sqrt(self.Px()**2 + self.Py()**2 + self.Pz()**2)
    def Pt(self): return math.sqrt(self.Px()**2 + self.Py()**2)
    def M2(self): return self.E()**2 - self.P()**2
    def M (self): return math.sqrt(self.M2()) if self.M2() > 0.0 else -math.sqrt(-self.M2())
    def atan2(self,y,x): return 0.0 if y==0.0 and x==0.0 else math.atan2(y, x)
    def Phi     (self): return self.atan2(self.Py(), self.Px())
    def Theta   (self): return self.atan2(self.Pt(), self.Pz())
    def CosTheta(self): return 1.0 if self.P() == 0.0 else self.Pz()/self.P()
    def Eta     (self): return -0.5* math.log( (1.0-self.CosTheta())/(1.0+self.CosTheta()) ) if self.CosTheta()**2 < 1.0 else 0.0

    def setMPmagPhiTheta(self, m, pmag, phi, theta):
        dp = 10
        self.E_ = round(math.sqrt(pmag**2 + m**2), dp)
        self.px_ = round(pmag*math.sin(theta)*math.cos(phi), dp)
        self.py_ = round(pmag*math.sin(theta)*math.sin(phi), dp)
        self.pz_ = round(pmag*math.cos(theta), dp)

    def boost(self, B):
        pframe=[0,0,B] #Beam incidence is z dir
        gamma=1/math.sqrt(1-B**2)
        E1 = self.E()
        px1 = self.Px()
        pz1 = self.Pz()
        self.E_ = gamma*E1-B*gamma*pz1
        self.pz_ = -B*gamma*E1+gamma*pz1
        
    def getFourVector(self):
        #return np.array([self.E(),self.Px(),self.Py(),self.Pz()])
        return np.array([self.E(),self.Pt(),self.Eta(),self.Phi()])
    
    def add(self, a, b):
        return 0.0 if abs(a+b) < 1e-10 else a+b

    def __add__(self, other):
        return Particle(self.add(self.E(),other.E()), self.add(self.Px(),other.Px()), self.add(self.Py(),other.Py()), self.add(self.Pz(),other.Pz()))

def simulateDecay(m1, m2, M):
    phi = random.uniform(-math.pi, math.pi)
    theta = random.uniform(0.02, math.pi-0.02)
    
    pmag1 = math.sqrt(((M**2+m1**2-m2**2)**2-4*M**2*m1**2))/(2*M)
    phi1 = phi
    theta1 = theta
    
    pmag2 = math.sqrt(((M**2+m2**2-m1**2)**2-4*M**2*m2**2))/(2*M)  
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
    event=[]
    particle1, particle2 = simulateDecay(m1, m2, M)
    parent=Particle()
    parent.setMPmagPhiTheta(M, 0.0, 0.0, math.pi/2)
    p = particle1 + particle2
    particle14v=particle1.getFourVector()
    particle24v=particle2.getFourVector()
    parent4v=parent.getFourVector()
    p4v = p.getFourVector()

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
        if i % 10000 == 0: print "Simulated {} events".format(i) 
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

