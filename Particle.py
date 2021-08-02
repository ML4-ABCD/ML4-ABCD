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