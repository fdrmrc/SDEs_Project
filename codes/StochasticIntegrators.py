import numpy as np
import matplotlib.pyplot as plt
import sys

class Integrators_LSO:
    def __init__(self, met_name,dt,y0,T,**kwargs):
        self.met_name = met_name #just a string 
        self.dt = dt
        self.y0 = y0
        self.alpha = kwargs.get('alpha') #parameter in the stoch. oscillator equation
        self.eta = kwargs.get('eta') #eta parameter poisson process 
        self.omega = kwargs.get('omega')
        self.theta = kwargs.get('theta')
        self.T = T #final time
        self.lamb = kwargs.get('lamb')
        
    
    
    def dW(self):   
        """Sample a random number at each call."""
        return np.random.normal(loc=0.0, scale=np.sqrt(self.dt))
    
        
    def dN(self):   
        #generate Poissonian noise
        Lambda = self.lamb
        if Lambda <=0: sys.exit('Not valid intensity')
        
        return np.random.poisson(lam=Lambda*self.dt)
    
    def poissproc(self):
        dt = self.dt
        n = self.T/dt
        N = np.zeros(int(n)+1)
        N[0] = 0 #a.s.
        for k in range(0,int(n)):
            xk = np.random.poisson(lam=self.lamb*dt)
            N[k+1] = N[k] + xk
        
        return N
            
    
    def cpoissproc(self):
        dt = self.dt
        n = self.T/dt
        
        Nhat = np.zeros(int(n)+1)
        Nhat[0] = 0 #a.s.
        for k in range(0,int(n)):
            
            mk = np.random.poisson(self.lamb*dt)
            mk = int(mk)
            X = []
            for j in range(mk):
                X.append( np.random.normal(loc=0.5, scale=1) )
            
            Nhat[k+1] = Nhat[k] + np.sum(X)
        
        return Nhat
    
    def EM_stepper(self,yn):
        #EM stepper
        dt = self.dt
        alpha = self.alpha
        A = np.array([[1,dt],[-dt,1]])
        b = np.array([0,1])
        return A@yn + alpha*b*self.dW()
    
    
    def BEM_stepper(self,yn):
        dt = self.dt
        alpha = self.alpha
        c = 1/(1+dt**2)
        A = c * np.array([[1,dt],[-dt,1]])
        b = c * np.array([dt,1])
        return A@yn + alpha*b*self.dW()
    
    
    def Theta_stepper(self,yn):
        dt = self.dt
        alpha = self.alpha
        theta= self.theta
        c = 1/(1+(dt*theta)**2)
        A = c * np.array([[1-(1-theta)*theta*dt**2, dt],[-dt, 1-(1-theta)*theta*dt**2]])
        b = c * np.array([dt*theta, 1])
        return A@yn + alpha*b*self.dW()
    
    
    
    def EX_stepper(self,yn):
        dt = self.dt
        alpha = self.alpha
        A = np.array([[np.cos(dt), np.sin(dt)],[-np.sin(dt), np.cos(dt)]])
        b = np.array([0, 1])
        return A@yn + alpha*b*self.dW()
            
    
    
    def SYM_stepper(self, yn):
        dt = self.dt
        alpha = self.alpha
        A = np.array([[1-dt**2, dt],[-dt, 1]])
        b = np.array([dt, 1])
        return A@yn + alpha*b*self.dW()
    
    
    def INT_stepper(self,yn):
        dt = self.dt
        alpha = self.alpha
        A = np.array([[np.cos(dt), np.sin(dt)],[-np.sin(dt), np.cos(dt)]])
        b = np.array([np.sin(dt), np.cos(dt)])
        return A@yn + alpha*b*self.dN()

    def PC_stepper(self,yn):
        dt = self.dt
        alpha = self.alpha
        A = np.array([[1 - dt**2, dt],[-dt, 1 - dt**2]])
        b = np.array([dt,1])
        return A@yn + alpha*b*self.dW()
    
    
    def TRIG_stepper(self,yn):
        dt = self.dt
        alpha = self.alpha
        eta = self.eta
        w = self.omega
        A = np.array([[np.cos(w*dt), (1/w)*np.sin(w*dt)],[-w*np.sin(w*dt), np.cos(w*dt)]])
        b = np.array([(1/w)*np.sin(w*dt),np.cos(w*dt)])
        c = np.array([(1/w)*np.sin(w*dt),np.cos(w*dt)])
        return A@yn + alpha*b*self.dW() + eta*c*self.dN()
    
    def evolve(self):
        #evolve up to final time
        ts = int(self.T/self.dt) #trunc
        y = np.zeros([2,ts+1])
        y[:,0] = self.y0
        if (self.met_name == 'EM'):
            for n in range(0,ts):
                y[:,n+1] = self.EM_stepper(y[:,n])
        elif (self.met_name == 'BEM'):
            for n in range(0,ts):
                y[:,n+1] = self.BEM_stepper(y[:,n])
        elif (self.met_name == 'Theta'):
            if (self.theta <=1 and self.theta>=0):
                for n in range(0,ts):
                    y[:,n+1] = self.Theta_stepper(y[:,n])
            else:
                sys.exit('theta = ' + str(self.theta)+' is not valid') #system exit 
        elif (self.met_name == 'EX'):
            for n in range(0,ts):
                y[:,n+1] = self.EX_stepper(y[:,n])
                
        elif (self.met_name == 'SYM'):
            for n in range(0,ts):
                y[:,n+1] = self.SYM_stepper(y[:,n])
                
        elif (self.met_name == 'INT'):
            for n in range(0,ts):
                y[:,n+1] = self.INT_stepper(y[:,n])
                
        elif (self.met_name == 'PC'):
            for n in range(0,ts):
                y[:,n+1] = self.PC_stepper(y[:,n])
        elif (self.met_name == 'TRIG'):
            for n in range(0,ts):
                y[:,n+1] = self.TRIG_stepper(y[:,n])
        
        return y
        
    
    
    
    #TODO def visualize(self,):
        #show solutions plot