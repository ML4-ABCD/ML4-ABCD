import numpy as np
import math

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

    def boost(self, bx=0.0, by=0.0, bz=0.0):
        b2 = bx*bx + by*by + bz*bz
        gamma = 1.0 / math.sqrt(1.0 - b2)
        bp = bx*self.Px() + by*self.Py() + bz*self.Pz()
        gamma2 = (gamma - 1.0)/b2 if b2 > 0 else 0.0

        self.px_ = self.Px() + gamma2*bp*bx + gamma*bx*self.E()
        self.py_ = self.Py() + gamma2*bp*by + gamma*by*self.E()
        self.pz_ = self.Pz() + gamma2*bp*bz + gamma*bz*self.E()
        self.E_ = gamma*(self.E() + bp) 
        
    def getFourVector(self):
        #return np.array([self.E(),self.Px(),self.Py(),self.Pz()])
        return np.array([self.E(),self.Pt(),self.Eta(),self.Phi()])
    
    def add(self, a, b):
        return 0.0 if abs(a+b) < 1e-10 else a+b

    def __add__(self, other):
        return Particle(self.add(self.E(),other.E()), self.add(self.Px(),other.Px()), self.add(self.Py(),other.Py()), self.add(self.Pz(),other.Pz()))

    def print(self):
        print(self.getFourVector())

def main():
    #import ROOT
    #p = Particle(100,0,0,0)
    #print(p.M())
    #p.print()
    #p.boost(0.1,0.2,0.3)
    #print(p.M())
    #p.print()

    v = ROOT.TLorentzVector()
    v.SetPxPyPzE(0,0,0,100)
    print(v.M()) 
    v.Print()
    v.Boost(0.1,0.2,0.3)
    print(v.M()) 
    v.Print()

if __name__ == '__main__':
    main()
