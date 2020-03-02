import numpy as np
#import matplotlib.pyplot as plt
from scipy.stats import norm

class heterogenity():

    def __init__(self,data):

        data = data[np.argsort(data[:,2])]
        data = np.flipud(data)
        
        self.layerID = data[:,0]
        self.height = data[:,1]
        self.permeability = data[:,2]
        self.porosity = data[:,3]
        self.wsaturation = data[:,4]
    
    def lorenz(self):
        
        flowcapacity = self.permeability*self.height
        storagecapacity = self.porosity*self.height

        fcapacity = np.cumsum(flowcapacity)/np.sum(flowcapacity)
        scapacity = np.cumsum(storagecapacity)/np.sum(storagecapacity)

        area = np.trapz(fcapacity,scapacity)

        coefficient = (area-0.5)/0.5
        
        return coefficient

    def dykstraParson(self):
        
        numdata = self.permeability.shape[0]

        probs = 1/(numdata+1)

        xaxis = np.linspace(1,numdata,numdata)
        xaxis = xaxis*probs
        xaxis = norm.ppf(xaxis)

        yaxis = np.log(self.permeability)
        #yaxis2 = np.log10(sortedPerm)

        #plt.plot(xaxis,yaxis,'k.')

        m,c = np.polyfit(xaxis,np.log(self.permeability),1)

        ybestfit = m*xaxis+c

        #plt.plot(xaxis,ybestfit,'k')

        #plt.show()

        k50p0 = np.exp(m*norm.ppf(0.5)+c)
        k84p1 = np.exp(m*norm.ppf(0.841)+c)

        coefficient = (k50p0-k84p1)/k50p0
        
        return coefficient

if __name__ == "__main__":

    data = np.loadtxt("petrophysics_data.txt",skiprows=1)

    raman = heterogenity(data)

    LC = raman.lorenz()
    DC = raman.dykstraParson()

    print(LC)
    print(DC)


