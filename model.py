#%%
import numpy as np
import matplotlib.pyplot as pp
import random as rnd
import scipy.optimize as op


class simulation1():
    
    def __init__(self,ticks,start,f,prob) -> None:
        self.ticks = ticks
        self.prob = prob
        self.start = start
        self.f = f
        self.res = np.empty((ticks+1))
        self.res[0] = start

    def setProb(self,prob):
        self.prob = prob

    
        
    def run(self):
        index = 1
        for i in range(self.ticks):
            val = self.prob.gen()
            self.res[index] = self.res[index - 1] + (self.f*self.res[index - 1])*val
            index += 1
        return self.res


class probD():

    def __init__(self,vec:np.array) -> None:
        self.vec = vec

    def setVec(self,vec:np.array):
        self.vec = vec
    
    def gen(self):
        num = rnd.uniform(0,1)
        sum = 0
        for i,p in enumerate(self.vec[0,:]):
            if sum<=num and num<= sum+p:
                return self.vec[1,i]
            sum += p
        return self.vec[1,-1]


    def find_best_f(self):
        func = lambda f: sum([v[0]*v[1]/(1+f*v[1]) for v in self.vec.transpose()])
        func_prime = lambda f: sum([-v[0]*v[1]**2/((1+f*v[1])**2) for v in self.vec.transpose()])
        return op.brentq(func,0.000001,0.99999)

    
class simulationMulti():
    def __init__(self,ticks,start,f,b,prob) -> None:
       self.ticks = ticks
       self.prob = prob
       self.start = start
       self.f = f
       self.b = b
       self.res = np.empty((ticks+1))
       self.res[0] = start
    
    def _genAll(self):
        vals = np.empty((len(f)))
        for i,prob in enumerate(self.prob):
            vals[i] = prob.gen()
        return vals

    def run(self):
        index = 1
        for i in range(self.ticks):
            vals = self._genAll()
            self.res[index] = self.res[index - 1]*(self.b + sum(vals*self.f))
            # tmp  =(self.b + vals*self.f)
            # for j in range(len(self.f)):
                # self.res[index]*=tmp[j]
            index += 1
        return self.res

def main1():
    vec = np.array([[0.154,0.50,0.32,0.026],[1.5,0.651,-1,-1.93]])
    d = probD(vec)
    s = simulation1(100,1,0.078,d)
    s.run()
    res = s.res

    s2 = simulation1(100,1,d.find_best_f(),d)
    s2.run()
    res2 = s2.res

    pp.plot(res,'-*')
    pp.plot(res2,'-*')
    pp.title("one option with multiple return options")
    pp.xlabel("tick")
    pp.ylabel("value")
    pp.legend(["random f","optimized f"])
    pp.grid(True)
    pp.show()

f = np.array([[0.7,0.3],[1.5,0.4]])
probs = [probD(f) for i in range(3)]
f = np.array([0.2,0.2,0.2])
s = simulationMulti(100,1,f,0.4,probs)
s.run()
res = s.res
x = np.linspace(0,100,101)
pp.plot(x,res,'-*')
pp.title("multi independent options, with multiple returns")
pp.xlabel("tick")
pp.ylabel("value")
pp.legend(["random f"])
pp.grid(True)
pp.show()