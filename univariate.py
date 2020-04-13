import numpy as np

from scipy.stats import norm

from geostatistics import csvreader

"""
This is a PNGE 436 - Reservoir Characterization Class Module including

    univariate analysis:
    - dykstraParson: Dykstra-Parson coefficient calculator
    - lorenz: Lorenz coefficient calculator
    
"""

class univariate():

    def __init__(self,filename):

        self = csvreader(self,filename)

        if not hasattr(self,'height'):
            self.height = self.depth-np.insert(self.depth[:-1],0,0)

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

if __name__ == "__main__":

    raman = univariate("univariate.csv")

    LC = raman.lorenz()
    DC = raman.dykstraParson()

    print(LC)
    print(DC)
