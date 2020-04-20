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
        self.z = data.Z

        self.f = data.F

        self.set_distance()

    def set_distance(self):

        x_T = np.reshape(self.x,(-1,1))
        y_T = np.reshape(self.y,(-1,1))
        z_T = np.reshape(self.z,(-1,1))

        self.dx = self.x-x_T
        self.dy = self.y-y_T
        self.dz = self.z-z_T

        self.distance = np.sqrt(self.dx**2+self.dy**2+self.dz**2)

    def set_bins(self,hmin,hmax):

        """
        bins is the array of lag distances 
        """

        tol = hmin*1e-6

        self.bins = np.arange(hmin,hmax+tol,hmin)

        self.lagdistol = hmin/2.

    def set_experimental(self,lagdip,lagdiptol):

        """
        - only directional experimental variogram calculation exist for now
        - calculations were verified for 2D data only
        """

        while lagdip<0:
            lagdip += 360

        while lagdip>360:
            lagdip -= 360

        self.lagdip = lagdip
        self.lagdiptol = lagdiptol

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

        conAzm = np.logical_and(self.azimuth>self.lagdip-self.lagdiptol,
                                self.azimuth<self.lagdip+self.lagdiptol)
        
        azmMatch = np.asfortranarray(np.where(conAzm)).T

        self.experimental = np.zeros_like(self.bins)

        for i,h in enumerate(self.bins):
            
            conDis = np.logical_and(self.distance>h-self.lagdistol,self.distance<h+self.lagdistol)

            disMatch = np.asfortranarray(np.where(conDis)).T

            """
            comparing disMatch to azmMatch to find indices matching both
            """

            dtype={'names':['f{}'.format(i) for i in range(2)],'formats':2*[disMatch.dtype]}

            match = np.intersect1d(disMatch.view(dtype),azmMatch.view(dtype))
            match = match.view(disMatch.dtype).reshape(-1,2)

            semivariance = ((self.f[match[:,0]]-self.f[match[:,1]])**2).sum()/(2*match.shape[0])

            self.experimental[i] = semivariance

    def set_theoretical(self):

        Dmat = self.distance
        
        var_mat = np.zeros_like(Dmat)
        
        C0 = self.nugget
        
        C = self.sill-self.nugget
        a = self.range

        hpa = Dmat/a

        if self.type == 'spherical':
            var_mat[Dmat>0] = C0+C*(3/2*hpa[Dmat>0]-1/2*hpa[Dmat>0]**3)
            var_mat[Dmat>a] = C0+C
        elif self.type == 'exponential':
            var_mat[Dmat>0] = C0+C*(1-np.exp(-3*hpa[Dmat>0]))
        elif self.type == 'gaussian':
            var_mat[Dmat>0] = C0+C*(1-np.exp(-3*hpa[Dmat>0]**2))
        elif self.type == 'hole_effect':
            var_mat[Dmat>0] = C0+C*(1-np.sin(hpa[Dmat>0])/hpa[Dmat>0])
            
        self.theoretical = var_mat

if __name__ == "__main__":

    class data: pass

    data.X = np.array([0,0,0,0,10,10,10,10,20,20,20,20,30,30,30,30])
    data.Y = np.array([0,10,20,30,0,10,20,30,0,10,20,30,0,10,20,30])
    data.Z = np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])

    data.F = np.array([10,20,24,32,12,17,20,28,9,10,16,12,8,7,12,18])

    var = variogram(data)
    var.set_bins(20*np.sqrt(2),20*np.sqrt(2))
    var.set_experimental(lagdip=135,lagdiptol=2)
    print(var.experimental)
