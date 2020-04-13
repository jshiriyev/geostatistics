import numpy as np

from setpy.setpy import setup

from variogram import variogram

"""
This is a PNGE 436 - Reservoir Characterization Class Module including

    - ordinary kriging
    - simple kriging
    
"""

class kriging():

    def __init__(self,raman):

        self.x = raman.X
        self.y = raman.Y
        self.z = raman.Z
        self.f = raman.porosity
        
        variogram.set_distance(self)

    def set_variogram(self,var_model):

##        self.type = model.type
##        self.nugget = model.nugget
##        self.sill = model.sill
##        self.range = model.range

        self.var_model = var_model

        self.var_mat = variogram.models(self.dist_mat,self.var_model)

        self.set_covariance()

    def set_covariance(self):

        self.cov_mat = self.var_model.sill-self.var_mat

    def estimate(self,coord,mean):

        x = coord[0]
        y = coord[1]
        z = coord[2]

        d = np.sqrt((x-self.x)**2+(y-self.y)**2+(z-self.z)**2)

        v = variogram.models(d,self.var_model)
        c = self.var_model.sill-v
        
        self.lambdas = np.linalg.solve(self.cov_mat,c)

        fvar = (self.f-mean)
        
        self.estimate = mean+(self.lambdas*fvar).sum()
        self.variance = self.var_model.sill-(self.lambdas*c).sum()

if __name__ == "__main__":

##    sheets = {
##        "names": ["porosity"],
##        "num_cols": [4],
##        "dataTypes": ["col"]
##        }

    class raman: pass
    class model: pass
    
##    setup('kriging.xlsx',sheets,raman).

    raman.X = np.array([600,400,800])
    raman.Y = np.array([800,700,100])
    raman.Z = np.array([1,1,1])
    raman.porosity = np.array([0.25,0.43,0.56])

    model.type = 'spherical'
    model.nugget = 0.0025*0.5
    model.sill = 0.0025
    model.range = 700
    
    krig = kriging(raman)
    krig.set_variogram(model)
    krig.estimate(np.array([100,200,1]),0.38)
