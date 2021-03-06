import numpy as np

from scipy import sparse as sps

from geostatistics import csvreader

"""
This is a PNGE 436 - Reservoir Characterization Class Module including

    bivariate analysis:
    - findslope: calculates slope and intercept from linear regression
    - correlation: correlation coefficient calculator

    the functions below looks like more of a univariate analysis:
    
    - ranking: ranks the given 1D numpy array (more of a function for now)
    - bootstrap: bootstraps the given 1D numpy array
    - interpolation: 1D dimensional analysis
    
"""

class bivariate():

    def __init__(self,filename):

        self = csvreader(self,filename)

    def findslope(self,X,Y,*args):
        
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

    def correlation(self,x,y):

        X = getattr(self,x)
        Y = getattr(self,y)

        N = X.shape[0]

        std_X = np.sqrt(1/(N-1)*np.sum((X-X.mean())**2))
        std_Y = np.sqrt(1/(N-1)*np.sum((Y-Y.mean())**2))
        
        cov_XY = 1/(N-1)*np.sum((X-X.mean())*(Y-Y.mean()))  
        rho_XY = cov_XY/(std_X*std_Y)
        
        return rho_XY

    def ranking(self,X):
        
        rank = np.empty_like(X)
        
        rank[X.argsort()] = np.arange(len(X))

        return rank

    def bootstrap(self,X,Nrealization):

        """
        X should be an array with one dimension,
        The size of X defines number of rows, and
        Nrealization specifies number of columns of an array
        created for bootstrap analyzes
        """
        
        N = X.size
        
        idx = np.random.randint(0,N,(N,Nrealization))
        
        return idx

    def interpolation(self,X,Y,x):

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

if __name__ == "__main__":

    raman = bivariate("univariate.csv")

    print(raman.correlation('porosity','permeability'))
