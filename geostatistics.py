import csv
import numpy as np

from scipy import sparse as sps

from scipy.stats import norm

"""
This is a PNGE 436 - Reservoir Characterization Class Module including

    - csv reader for petrophysical data

    univariate analysis:
    - heterogeneity coefficients calculator
    	Dykstra-Parson and Lorenz coefficients

    bivariate analysis:
    - correlation coefficient calculator
    
"""

def csvreader(obj,filename):

    data = np.genfromtxt(filename,skip_header=1,delimiter=',')

    with open(filename) as csvfile:
        reader = csv.reader(csvfile,delimiter=',')
        headers = next(reader)

    for columnID,header in enumerate(headers):
        setattr(obj,header,data[:,columnID])

    return obj

def findslope(X,Y,*args):
    
    if not args:
        flag = False
    else:
        flag = args[0]

    N = X.size

    O = np.ones((N,1))
    X = X.reshape((-1,1))
    Y = Y.reshape((-1,1))

    G = np.concatenate((O,X),axis=1)

    A = np.dot(G.transpose(),G)
    b = np.dot(G.transpose(),Y)

    m = np.linalg.solve(A,b)
    
    if flag:
        yapp = np.dot(G,m)	# approximated y
        return m, yapp.flatten()
    else:
        return m

def correlation(X,Y):

    N = X.shape[0]

    std_X = np.sqrt(1/(N-1)*np.sum((X-X.mean())**2))
    std_Y = np.sqrt(1/(N-1)*np.sum((Y-Y.mean())**2))
    
    cov_XY = 1/(N-1)*np.sum((X-X.mean())*(Y-Y.mean()))  
    rho_XY = cov_XY/(std_X*std_Y)
    
    return rho_XY

def ranking(X):
    
    rank = np.empty_like(X)
    
    rank[X.argsort()] = np.arange(len(X))

    return rank

def bootstrap(X,Nrealization):

    """
    X should be an array with one dimension,
    The size of X defines number of rows, and
    Nrealization specifies number of columns of an array
    created for bootstrap analyzes
    """
    
    N = X.size
    
    idx = np.random.randint(0,N,(N,Nrealization))
    
    return idx

def interpolation(X,Y,x):

    """
    X are the locations where Y values are given
    x are the locations where we want to calculate y
    based on the interpolation of Y values
    """
    
    xadded = X[x.max()<X]
    
    x = np.append(x,xadded)
    
    N = x.size
    L = X.size

    d1 = np.array(list(range(N)))
    d2 = np.array(list(range(N,N+L)))
    
    row = np.concatenate(((d1[0],d1[-1]),d1[:-1],d1[1:],d1[1:-1]))
    col = np.concatenate(((d1[0],d1[-1]),d1[1:],d1[:-1],d1[1:-1]))

    Mone = np.ones(2)*(-1)
    Pone = np.ones(2*(N-1))
    Mtwo = np.ones((N-2))*(-2)

    data = np.concatenate((Mone,Pone,Mtwo))

    G = sps.csr_matrix((data,(row,col)),shape=(N,N))
    d = sps.csr_matrix((Y,(d2,np.zeros(L))),shape=(N+L,1))

    x = x.reshape((1,-1))
    X = X.reshape((-1,1))
    
    Glow = np.zeros((L,N))

    dmat = np.abs(x-X)              # distance matrix for the given x and X vectors
    
    colID = dmat.argsort()[:,:2]    # column indices of two minimum row values in dmat
    rowID = np.tile(np.arange(L).reshape((-1,1)),2)
                                    # row indices of two minimum row values in dmat
    
    dmin = dmat[rowID,colID]        # two minimum distance values of each row in dmat

    Glow[rowID,colID] = 1-dmin/np.sum(dmin,axis=1,keepdims=True)

    G = sps.vstack([G,sps.csr_matrix(Glow)])

    A = G.transpose()*G
    b = G.transpose()*d

    y = sps.linalg.spsolve(A,b)

    if xadded.size:
        return y[:-xadded.size]
    else:
        return y

class heterogeneity():

    def __init__(self,filename):

        self = csvreader(self,filename)

        if not hasattr(self,'height'):
            self.height = self.depth-np.insert(self.depth[:-1],0,0)
    
    def lorenz(self):

        p = np.flip(self.permeability.argsort())

        sk = self.permeability[p]
        sp = self.porosity[p]
        sh = self.height[p]
        
        flow = np.cumsum(sk*sh)/np.sum(sk*sh)
        storage = np.cumsum(sp*sh)/np.sum(sp*sh)

        area = np.trapz(flow,storage)
        
        coefficient = (area-0.5)/0.5
        
        return coefficient

    def dykstraParson(self):

        p = np.flip(self.permeability.argsort())

        sk = self.permeability[p]
        
        numdata = sk.shape[0]

        probs = 1/(numdata+1)

        xaxis = np.linspace(1,numdata,numdata)
        xaxis = xaxis*probs
        xaxis = norm.ppf(xaxis)

        yaxis = np.log(sk)
        #yaxis2 = np.log10(sortedPerm)

        #plt.plot(xaxis,yaxis,'k.')

        m,c = np.polyfit(xaxis,yaxis,1)

        ybestfit = m*xaxis+c

        #plt.plot(xaxis,ybestfit,'k')

        #plt.show()

        k50p0 = np.exp(m*norm.ppf(0.5)+c)
        k84p1 = np.exp(m*norm.ppf(0.841)+c)

        coefficient = (k50p0-k84p1)/k50p0
        
        return coefficient

class directionalVariogram():

    def __init__(self,
                 filename,
                 lagdistance=None,
                 lagdisTol=None,
                 lagdip=None,
                 lagdipTol=None):

        self = csvreader(self,filename)

        self.X = self.west_east.reshape(-1,1)
        self.Y = self.north_south.reshape(-1,1)
        self.Z = self.toc.reshape(-1,1)

        self.distance = np.sqrt((self.X-self.X.T)**2+(self.Y-self.Y.T)**2)

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
            self.lagdipTol = self.lagdip/2.
        else:
            self.lagdipTol = lagdipTol

        """
        for now I calculate self.bins, however it is not used yet
        variogram is calculated only for lagdistance
        """

        self.bins = np.arange(self.lagdistance,self.distance.max(),self.lagdistance)

        """
        if we set x direction as east and y direction as north
        then the following azimuth will be zero toward east and
        will be positive in counterclockwise direction
        """
        
        self.azimuth = 180.+np.degrees(np.arctan2(self.Y-self.Y.T,self.X-self.X.T))
        self.azimuth = np.where(self.azimuth==360,0,self.azimuth)

        """
        finding indexes when lag_distance matches data spacing, disMatch
        and when dip angle matches azimuth, azmMatch
        for non uniform spacing most modification will be here probably.
        """

        conDis = np.logical_and(self.distance>self.lagdistance-self.lagdisTol,
                                self.distance<self.lagdistance+self.lagdisTol)
        
        disMatch = np.asfortranarray(np.where(conDis)).T
        azmMatch = np.asfortranarray(np.where(self.azimuth==self.lagdip)).T

        """
        comparing disMatch to azmMatch to find indices matching both
        """

        dtype={'names':['f{}'.format(i) for i in range(2)],'formats':2*[disMatch.dtype]}

        match = np.intersect1d(disMatch.view(dtype),azmMatch.view(dtype))
        match = match.view(disMatch.dtype).reshape(-1,2)

        self.experimental = ((self.Z[match[:,0]]-self.Z[match[:,1]])**2).sum()/(2*match.shape[0])

if __name__ == "__main__":

    raman = heterogeneity("petrophysics.csv")

    LC = raman.lorenz()
    DC = raman.dykstraParson()

    print(LC)
    print(DC)
