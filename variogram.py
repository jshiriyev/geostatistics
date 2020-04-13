import numpy as np

"""
This is a PNGE 436 - Reservoir Characterization Class Module including

    - variogram class:
    - calculates experimental variograms
    - calculates theoretical variograms
    
"""

class variogram():

    def __init__(self,data):

        self.x = data.X
        self.y = data.Y
##        self.z = data.Z

        self.f = data.F

        self.set_distance()

    def set_distance(self):

        x_T = np.reshape(self.x,(-1,1))
        y_T = np.reshape(self.y,(-1,1))
##        z_T = np.reshape(self.z,(-1,1))

        self.dx = self.x-x_T
        self.dy = self.y-y_T
##        self.dz = self.z-z_T

##        self.distance = np.sqrt(self.dx**2+self.dy**2+self.dz**2)
        self.distance = np.sqrt(self.dx**2+self.dy**2)

    def set_experimental(self,lagdistance=None,lagdisTol=None,lagdip=None,lagdipTol=None):

        if lagdistance is None:
            self.lagdistance = np.where(self.distance==0.,np.inf,self.distance).min()
        else:
            self.lagdistance = lagdistance

        if lagdisTol is None:
            self.lagdisTol = self.lagdistance/2.
        else:
            self.lagdisTol = lagdisTol

        if lagdip is not None:
            self.lagdip = lagdip

        if lagdipTol is None:
            self.lagdipTol = self.lagdip/2.+1e-5
        else:
            self.lagdipTol = lagdipTol

        self.bins = np.arange(self.lagdistance,self.distance.max(),self.lagdistance)

        """
        if we set x direction as east and y direction as north
        then the following azimuth will be zero toward east and
        will be positive in counterclockwise direction
        """
        
        self.azimuth = np.degrees(np.arctan2(self.dy,self.dx))+180.
        self.azimuth = np.where(self.azimuth==360,0,self.azimuth)

        """
        finding indexes when lag_distance matches data spacing, disMatch
        and when dip angle matches azimuth, azmMatch
        for non uniform spacing most modification will be here probably.
        """
        
        ##### what is the best way of comparing floating numbers with some tolerance???
        
        conDis = np.logical_and(self.distance>self.lagdistance-self.lagdisTol,
                                self.distance<self.lagdistance+self.lagdisTol)
        
        conAzm = np.logical_and(self.azimuth>self.lagdip-self.lagdipTol,
                                self.azimuth<self.lagdip+self.lagdipTol)
        
        disMatch = np.asfortranarray(np.where(conDis)).T
        azmMatch = np.asfortranarray(np.where(conAzm)).T

        """
        comparing disMatch to azmMatch to find indices matching both
        """

        dtype={'names':['f{}'.format(i) for i in range(2)],'formats':2*[disMatch.dtype]}

        match = np.intersect1d(disMatch.view(dtype),azmMatch.view(dtype))
        match = match.view(disMatch.dtype).reshape(-1,2)

        print(match[:,0])
        print(match[:,1])

        self.experimental = ((self.f[match[:,0]]-self.f[match[:,1]])**2).sum()/(2*match.shape[0])

    def set_theoretical(dist_mat,model):

        Dmat = dist_mat

        var_type = model.type
        var_nugget = model.nugget
        var_sill = model.sill
        var_range = model.range
        
        var_mat = np.zeros_like(Dmat)
        
        C0 = var_nugget
        
        C = var_sill-var_nugget
        a = var_range

        hpa = Dmat/a

        if var_type == 'spherical':
            var_mat[Dmat>0] = C0+C*(3/2*hpa[Dmat>0]-1/2*hpa[Dmat>0]**3)
            var_mat[Dmat>a] = C0+C
        elif var_type == 'exponential':
            var_mat = C0+C*(1-np.exp(-3*hpa))
        elif var_type == 'gaussian':
            var_mat = C0+C*(1-np.exp(-3*hpa**2))
        elif var_type == 'hole_effect':
            var_mat = C0+C*(1-np.sin(hpa)/hpa)
            
        return var_mat

if __name__ == "__main__":

    class data: pass

    data.X = np.array([0,0,0,0,10,10,10,10,20,20,20,20,30,30,30,30])
    data.Y = np.array([0,10,20,30,0,10,20,30,0,10,20,30,0,10,20,30])
    data.Z = np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])

    data.F = np.array([10,20,24,32,12,17,20,28,9,10,16,12,8,7,12,18])

    var = variogram(data)
    var.set_experimental(lagdistance=20*np.sqrt(2),lagdisTol=2,lagdip=135,lagdipTol=45)
    print(var.experimental)
