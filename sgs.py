import numpy as np

from kriging import kriging

import matplotlib.pyplot as plt

"""
This is a PNGE 436 - Reservoir Characterization Class Module including

    - sequential gaussian simulation
    
"""

class sgs():

    def __init__(self,obs,est):
        
        self.obs = obs
        self.est = est

        self.simulate()

    def update(self,found):

        self.obs.X = np.append(self.obs.X,found.X)
        self.obs.Y = np.append(self.obs.Y,found.Y)
        self.obs.Z = np.append(self.obs.Z,found.Z)

        f_m = found.F
        f_v = found.F_variance

        f_r = np.random.normal(f_m,np.sqrt(f_v))

        self.obs.F = np.append(self.obs.F,f_r)

    def simulate(self):

        class found: pass

        N = self.est.X.size

        while N>0:

            randint = np.random.randint(0,N)

            found.X = self.est.X[randint]
            found.Y = self.est.Y[randint]
            found.Z = self.est.Z[randint]

            krig = kriging(self.obs)
            krig.ordinary(found)
            
            self.update(found)
            
            self.est.X = np.delete(self.est.X,randint)
            self.est.Y = np.delete(self.est.Y,randint)
            self.est.Z = np.delete(self.est.Z,randint)
            
            N = self.est.X.size

if __name__ == "__main__":

    class observation: pass
    class estimation: pass

    observation.X = np.array([2,4,6])
    observation.Y = np.array([1,1,1])
    observation.Z = np.array([1,1,1])
    
    observation.F = np.array([30,50,20])

    observation.type = 'exponential'
    observation.nugget = 0
    observation.sill = 100
    observation.range = 10

    estimation.X = np.array([1,3,5,7,8])
    estimation.Y = np.array([1,1,1,1,1])
    estimation.Z = np.array([1,1,1,1,1])

    class obs1(observation): pass
    class obs2(observation): pass

    class est1(estimation): pass
    class est2(estimation): pass
    
    sgs(obs1,est1)

    krig = kriging(obs2)
    krig.ordinary(est2)

    obs2.X = np.append(obs2.X,est2.X)
    obs2.F = np.append(obs2.F,est2.F)

    idx1 = np.argsort(obs1.X)
    idx2 = np.argsort(obs2.X)
    
    plt.plot(obs1.X[idx1],obs1.F[idx1])
    plt.plot(obs2.X[idx2],obs2.F[idx2])
    plt.scatter(observation.X,observation.F,c='k')

    plt.xlim([0,9])
    plt.ylim([0,60])

    plt.legend(('simulation','kriging','given data'))

    plt.show()
    
