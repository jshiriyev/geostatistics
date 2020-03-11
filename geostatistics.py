import csv
import numpy as np

from scipy import sparse as sps

from scipy.stats import norm

"""
This is a PNGE 436 - Reservoir Characterization Class Module including

    - csv reader for petrophysical data
    - correlation coefficient calculator
    - heterogeneity coefficients calculator
    	Dykstra-Parson and Lorenz coefficients
    
"""

def csvreader(obj,filename):

    data = np.genfromtxt(filename,skip_header=1,delimiter=',')

    with open(filename) as csvfile:
        reader = csv.reader(csvfile,delimiter=',')
        headers = next(reader)

    for columnID,header in enumerate(headers):
        setattr(obj,header,data[:,columnID])

    return obj

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

if __name__ == "__main__":

    raman = heterogeneity("petrophysics.csv")

    LC = raman.lorenz()
    DC = raman.dykstraParson()

    print(LC)
    print(DC)
