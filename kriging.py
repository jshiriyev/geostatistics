import numpy as np

from geostatistics.variogram import variogram

"""
This is a PNGE 436 - Reservoir Characterization Class Module including

    - simple kriging
    - ordinary kriging
    
"""

class kriging():

    def __init__(self,obs):

        self.x = obs.X
        self.y = obs.Y
        self.z = obs.Z
        
        self.f = obs.F

        self.type = obs.type
        self.nugget = obs.nugget
        self.sill = obs.sill
        self.range = obs.range
        
        variogram.set_distance(self)
        variogram.set_theoretical(self)

        self.covariance = self.sill-self.theoretical

    def simple(self,est):

        x_T = np.reshape(self.x,(-1,1))
        y_T = np.reshape(self.y,(-1,1))
        z_T = np.reshape(self.z,(-1,1))

        f_T = np.reshape(self.f,(-1,1))

        est.distance = np.sqrt((x_T-est.X)**2+(y_T-est.Y)**2+(z_T-est.Z)**2)

        est.type = self.type
        est.nugget = self.nugget
        est.sill = self.sill
        est.range = self.range

        variogram.set_theoretical(est)
        
        est.covariance = est.sill-est.theoretical

        est.lambdas = np.linalg.solve(self.covariance,est.covariance)

        est.F = est.mean+(est.lambdas*(f_T-est.mean)).sum(axis=0)
        est.F_variance = self.sill-(est.lambdas*est.covariance).sum(axis=0)
        
        return est

    def ordinary(self,est):

        x_T = np.reshape(self.x,(-1,1))
        y_T = np.reshape(self.y,(-1,1))
        z_T = np.reshape(self.z,(-1,1))

        f_T = np.reshape(self.f,(-1,1))

        est.distance = np.sqrt((x_T-est.X)**2+(y_T-est.Y)**2+(z_T-est.Z)**2)

        est.type = self.type
        est.nugget = self.nugget
        est.sill = self.sill
        est.range = self.range

        variogram.set_theoretical(est)

        est.covariance = est.sill-est.theoretical
        
        Am = self.covariance
        Ar = np.ones(Am.shape[0]).reshape((-1,1))
        Ab = np.ones(Am.shape[0]).reshape((1,-1))
        Ab = np.append(Ab,np.array([[0]]),axis=1)
        Am = np.append(Am,Ar,axis=1)
        Am = np.append(Am,Ab,axis=0)

        bm = est.covariance
        bb = np.ones(bm.shape[1]).reshape((1,-1))
        bm = np.append(bm,bb,axis=0)

        xm = np.linalg.solve(Am,bm)
        
        est.lambdas = xm[:-1,:]
        est.beta = xm[-1,:]

        est.F = (est.lambdas*f_T).sum(axis=0)
        est.F_variance = self.sill-est.beta-(est.lambdas*est.covariance).sum(axis=0)
        
        return est

if __name__ == "__main__":

##    class observation: pass
##    class estimation1: pass
##    class estimation2: pass
##
##    observation.X = np.array([600,400,800])
##    observation.Y = np.array([800,700,100])
##    observation.Z = np.array([1,1,1])
##    
##    observation.F = np.array([0.25,0.43,0.56])
##
##    observation.type = 'spherical'
##    observation.nugget = 0.0025*0
##    observation.sill = 0.0025
##    observation.range = 700
##
##    estimation1.X = np.array([500,100])
##    estimation1.Y = np.array([500,200])
##    estimation1.Z = np.array([1,1])
##
##    estimation1.mean = 0.38
##
##    estimation2.X = np.array([500,100])
##    estimation2.Y = np.array([500,200])
##    estimation2.Z = np.array([1,1])
##    
##    krig = kriging(observation)
##    
##    krig.simple(estimation1)
##    krig.ordinary(estimation2)

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

    estimation.X = np.array([8])
    estimation.Y = np.array([1])
    estimation.Z = np.array([1])
    
    krig = kriging(observation)
    krig.ordinary(estimation)
