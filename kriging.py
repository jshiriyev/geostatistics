import numpy as np

from setpy.setpy import setup

from variogram import variogram

"""
This is a PNGE 436 - Reservoir Characterization Class Module including

    - ordinary kriging
    - simple kriging
    
"""

class kriging():

    def __init__(self,data):

        self.x = data.X
        self.y = data.Y
        self.z = data.Z
        
        self.f = data.F
        
        variogram.set_distance(self)

    def set_variogram(self,var_model):

        self.var_model = var_model

        self.var_mat = variogram.set_theoretical(self.dist_mat,self.var_model)

        self.cov_mat = self.var_model.sill-self.var_mat

    def estimate(self,coord,mean):

        x = coord[0]
        y = coord[1]
        z = coord[2]

        d = np.sqrt((x-self.x)**2+(y-self.y)**2+(z-self.z)**2)

        variogram.set_distance

        v = variogram.models(d,self.var_model)
        c = self.var_model.sill-v
        
        self.lambdas = np.linalg.solve(self.cov_mat,c)

        fvar = (self.f-mean)
        
        self.estimate = mean+(self.lambdas*fvar).sum()
        self.variance = self.var_model.sill-(self.lambdas*c).sum()

if __name__ == "__main__":

    class data: pass
    class model: pass

    data.X = np.array([600,400,800])
    data.Y = np.array([800,700,100])
    data.Z = np.array([1,1,1])
    
    data.F = np.array([0.25,0.43,0.56])

    var_model.type = 'spherical'
    var_model.nugget = 0.0025*0.5
    var_model.sill = 0.0025
    var_model.range = 700
    
    krig = kriging(raman)
    krig.set_variogram(var_model)
    krig.estimate(np.array([100,200,1]),0.38)
