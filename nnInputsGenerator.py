import math
import random
import numpy as np
import matplotlib.pyplot as plt
import array
import string

class Particle(object):   
    def __init__(self, E, px, py, pz, pt, eta, phi):
        self.E = E
        self.px = px
        self.py = py
        self.pz = pz
        self.pt = pt
        self.eta = eta
        self.phi = phi

    
    def boost(self, B):

      pframe=[0,0,B] #Beam incidence is z dir

      gamma=1/math.sqrt(1-B**2)
    

      E1=self.E
      px1=self.px
      pz1=self.pz

      self.E = gamma*E1-B*gamma*pz1
      self.pz = -B*gamma*E1+gamma*pz1
      self.eta = -math.log(abs(math.tan(1/2*np.arctan(px1/self.pz))))
  
      return self
    
    def gitfourvector(self):
      return np.array([self.E,self.pt,self.eta,self.phi])

def createEventforRealthisTime(Mass):
  pairs=[]
  event=[]
  
  M=random.gauss(Mass,1) # in GeV
  Msq=M**2
  m1=20 #same for every pair- but does ::" depending on M?
  m2=5

  pmag1=math.sqrt(((Msq+m1**2-m2**2)**2-4*M**2*m1**2))/(2*M)
  pmag2=math.sqrt(((Msq+m2**2-m1**2)**2-4*M**2*m2**2))/(2*M)
  
  phi=random.uniform(0,2*math.pi)
  theta=random.uniform(0,math.pi)

  p1x=pmag1*math.sin(phi)*math.cos(theta)
  p1y=pmag1*math.sin(phi)*math.sin(theta)
  p1z=pmag1*math.cos(phi)
  
  E1=math.sqrt(pmag1**2+m1**2)
  
  phi1=phi 
  pt1= math.sqrt(p1x**2+p1y**2)   #assuming the beam line is in the z direction, px and py come solely from the collision
                                  #pt is invariant under boost
  theta1=theta
  tn=math.tan(phi1/2)
  eta1=-math.log(math.tan(theta1/2)) # way of reconceptualizing angle of incidence to beam- theta- that creates a value deltaeta that is invariant- useful bc no way to know momentum of quarks in lab frame being hit-
  

  p2x=pmag2*math.sin(phi+math.pi)*math.cos(theta+math.pi)
  p2y=pmag2*math.sin(phi+math.pi)*math.sin(theta+math.pi)
  p2z=pmag2*math.cos(phi+math.pi)
  E2=math.sqrt(pmag2**2+m2**2) 


  phi2=phi+math.pi
  pt2= math.sqrt(p2x**2+p2y**2)
  theta2=theta
  eta2=-math.log(math.tan(theta2/2))
  


  particle1=Particle(E1, p1x,p1y,p1z,pt1, eta1, phi1)
  particle14v=particle1.gitfourvector()

  particle2=Particle(E2,p2x,p2y,p2z,pt2,eta2,phi2)
  particle24v=particle2.gitfourvector()

  #pairs=[particle1.gitfourvector(), particle2.gitfourvector()]

  parent=Particle(particle1.E+particle2.E,0,0,0,0,"inf",0)
  parent4v=parent.gitfourvector()

  #event=np.concatenate(particle14v, particle24v,parent4v)
  event.extend(particle14v)
  event.extend(particle24v)
  event.extend(parent4v)
  event.extend('1')
  return event



def createEventforRealthisTimebackground(Mass):
  pairs=[]
  event=[]
  
  M=random.expovariate(Mass) # in GeV
  Msq=M**2
  m1=20 #same for every pair- but does ::" depending on M?
  m2=5

  pmag1=math.sqrt(((Msq+m1**2-m2**2)**2-4*M**2*m1**2))/(2*M)
  pmag2=math.sqrt(((Msq+m2**2-m1**2)**2-4*M**2*m2**2))/(2*M)
  
  phi=random.uniform(0,2*math.pi)
  theta=random.uniform(0,math.pi)

  p1x=pmag1*math.sin(phi)*math.cos(theta)
  p1y=pmag1*math.sin(phi)*math.sin(theta)
  p1z=pmag1*math.cos(phi)
  
  E1=math.sqrt(pmag1**2+m1**2)
  
  phi1=phi 
  pt1= math.sqrt(p1x**2+p1y**2)   #assuming the beam line is in the z direction, px and py come solely from the collision
                                  #pt is invariant under boost
  theta1=theta
  tn=math.tan(phi1/2)
  eta1=-math.log(math.tan(theta1/2)) # way of reconceptualizing angle of incidence to beam- theta- that creates a value deltaeta that is invariant- useful bc no way to know momentum of quarks in lab frame being hit-
  

  p2x=pmag2*math.sin(phi+math.pi)*math.cos(theta+math.pi)
  p2y=pmag2*math.sin(phi+math.pi)*math.sin(theta+math.pi)
  p2z=pmag2*math.cos(phi+math.pi)
  E2=math.sqrt(pmag2**2+m2**2) 


  phi2=phi+math.pi
  pt2= math.sqrt(p2x**2+p2y**2)
  theta2=theta
  eta2=-math.log(math.tan(theta2/2))
  


  particle1=Particle(E1, p1x,p1y,p1z,pt1, eta1, phi1)
  particle14v=particle1.gitfourvector()

  particle2=Particle(E2,p2x,p2y,p2z,pt2,eta2,phi2)
  particle24v=particle2.gitfourvector()


  #parent=Particle(particle1.E+particle2.E,0,0,0,0,"inf",0) No parent bc this is background
  #parent4v=parent.gitfourvector()

  event.extend(particle14v)
  event.extend(particle24v)
  #event.extend(parent4v)
  event.extend("0")

  return event



listofevents=[[createEventforRealthisTime(80) for _ in range(0,1000)], [createEventforRealthisTimebackground(80) for _ in range(0,1000)]]
newoutput='absolutepath'
np.save(newoutput,listofevents)
