#%%
from math import log
import numpy as np
import matplotlib.pyplot as pp
import random as rnd
import scipy.optimize as op
from itertools import product 
from scipy.optimize import minimize,LinearConstraint
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
            self.res[index] = (1-self.f)*self.res[index - 1] + (self.f*self.res[index - 1])*val
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
        func = lambda f: sum([v[0]*(v[1] -1 )/(1+f*(v[1] -1 )) for v in self.vec.transpose()])
        func_prime = lambda f: sum([-v[0]*v[1]**2/((1+f*v[1])**2) for v in self.vec.transpose()])
        return op.brentq(func,0.000001,1)

    
class simMulti1W():
    def __init__(self,ticks,start,f,b,prob) -> None:
       self.ticks = ticks
       self.prob = prob
       self.start = start
       self.f = f
       self.b = b
       self.res = np.empty((ticks+1))
       self.res[0] = start
    
    def _genAll(self):
        vals = np.empty((len(self.prob)))
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


def combinations(probabilities):
    probs = list()
    for p in probabilities:
        probs.append(range(0,len(p.vec[0])))
    products = product(*probs)
    return products

def target_factory(probabilities):
    iter_comb = combinations(probabilities)
    # look at probablity calc optimization
    def target(f:np.array):
        sum = 0
        for comb in iter_comb:
            s = 1
            x = np.empty((len(probabilities)))
            index = 0
            for p,i in zip(probabilities,comb):
                v = p.vec
                s *= v[0,i]
                x[index] = v[1,i]
                index += 1
            s *= np.log(f[0] + x@f[1:])
            sum += s
        return -sum
    return target

def optimize_f(probabilities):
    target = target_factory(probabilities)
    const = {'type': 'eq', 'fun': lambda x:  sum(x) - 1}
    bounds= ((0,None) for x in range(len(probabilities)))
    bounds = list(bounds)
    bounds.insert(0,(0.00001,None))
    bounds = tuple(bounds)
    return minimize(target,np.zeros(len(probabilities)+1),method='SLSQP',constraints=const,bounds=bounds)



def main1():
    vec = np.array([[0.15,0.50,0.32,0.03],[1.5,1.15,0.87,0.3]])
    d = probD(vec)
    s = simulation1(100,1,1,d)
    s.run()
    res = s.res

    s2 = simulation1(100,1,d.find_best_f(),d)
    s2.run()
    res2 = s2.res

    pp.plot(res,'-*')
    pp.plot(res2,'-*')
    pp.title("one option with multiple outcomes")
    pp.xlabel("tick")
    pp.ylabel("value")
    pp.legend(["f = 1","optimized f = {0:.2f}".format(d.find_best_f())])
    pp.grid(True)
    pp.show()
# main1()
def main2():
    f = np.array([[0.7,0.3],[1.5,0.4]])
    probs = [probD(f) for i in range(3)]
    
    f_max = optimize_f(probs)
    f_max = f_max['x']
    f = np.array([0.2,0.2,0.2])
    s = simMulti1W(100,1,f,0.4,probs)
    s.run()
    res = s.res
    s_max = simMulti1W(100,1,f_max[1:],f_max[0],probs)
    s_max.run()
    res_max = s_max.res
    x = np.linspace(0,100,101)
    pp.plot(x,res,'-*')
    pp.plot(x,res_max,'-*')
    pp.title("multi independent options, with multiple returns")
    pp.xlabel("tick")
    pp.ylabel("value")
    pp.legend(["random f"])
    pp.grid(True)
    pp.show()
# main2()
