import numpy as np

class Network:

    def __init__(self, N, tauW=32.0, tauX=1.0, tauY=1.0, dt=1.0, divThresh = 0.000001, maxSteps = 400):
        self.N = N
        self.X = np.zeros(self.N)
        self.U = np.zeros(self.N)
        self.Y = np.zeros(self.N)
        self.F = np.zeros(self.N)
        self.J = np.zeros(self.N)
        self.W = []
        self.Pre = np.array([],dtype=int)
        self.Post = np.array([],dtype=int)
        self.Fprime = np.zeros(self.N)
        self.Nplus1 = self.N
        self.divThresh = divThresh * self.N
        self.maxSteps = maxSteps
        self.dt = dt
        self.dtOverTauW = self.dt/tauW
        self.dtOverTauX = self.dt/tauX
        self.dtOverTauY = self.dt/tauY

    def setInputID(self,x):
        self.inputID = x

    def setOutputID(self,x):
        self.outputID = x

    def setMapInput(self,x):
        self.input = x

    def setMapOutput(self,x):
        self.output = x

    def addBias(self):
        for i in range(self.N):
            self.W = np.hstack([self.W,0.0])
            self.Pre = np.hstack([self.Pre,self.N])
            self.Post = np.hstack([self.Post,i])
        self.X = np.hstack([self.X,1.0])
        self.Nplus1 = self.N+1
        self.V = np.zeros(self.Nplus1)
        self.Input = np.zeros(self.Nplus1)

    def connect(self, pre, post, weight=0.0):
        self.W = np.hstack([self.W,weight])
        self.Pre = np.hstack([self.Pre,pre])
        self.Post = np.hstack([self.Post,post])

    def randomizeWeights(self,weightMin, weightMax):
        weightRange = weightMax-weightMin
        for i in range(len(self.W)):
            self.W[i] = np.random.rand()*weightRange+weightMin

    def setNet(self):
        self.Nweight = len(self.W)
        self.Wbest = self.W.copy()

    def postConnect(self):
        self.addBias()
        self.setNet()

    def reset(self):
        self.X[:-1] = 0.0
        self.Y[:-1] = 0.0
        self.Input[:] = 0.0

    def forward(self):
        self.U[:] = 0.0
        for k in range(self.Nweight):
            self.U[self.Post[k]] = self.U[self.Post[k]] + self.X[self.Pre[k]] * self.W[k]

        for i in range(self.N):
            self.F[i] = 1./(1.0+np.exp(-self.U[i]))

        for i in range(self.N):
            self.X[i] = self.X[i] + self.dtOverTauX * (-self.X[i] + self.F[i] + self.Input[i])

    def setError(self, oID, targetOutput):
        self.J[:] = 0.0
        for i in range(len(oID)):
            self.J[oID[i]] = targetOutput[i] - self.X[oID[i]]

    def backward(self):
        self.Fprime = self.F*(1.0 - self.F)
        self.V[:-1] = 0.0
        for k in range(self.Nweight):
            self.V[self.Pre[k]] = self.V[self.Pre[k]] + self.Fprime[self.Post[k]] * self.W[k] * self.Y[self.Post[k]]
        for i in range(self.N):
            self.Y[i] = self.Y[i] +self.dtOverTauY * (self.V[i] - self.Y[i] + self.J[i])

    def weightUpdate(self):
        for k in range(self.Nweight):
            delta = self.X[self.Pre[k]] * self.Y[self.Post[k]]* self.Fprime[self.Post[k]]
            if(delta<-1.0):
                self.W[k] = self.W[k] -self.dtOverTauW
            elif (delta>1.0):
                self.W[k] = self.W[k] +self.dtOverTauW
            else:
                self.W[k] = self.W[k] +self.dtOverTauW*delta


    def getError(self):
        return np.sum(self.J**2) * 0.5

    def convergeForward(self):
        Xpre = np.zeros(self.N)
        total = self.N*1.0
        for t in range(self.maxSteps):
            if(total>self.divThresh):
                Xpre=self.X.copy()
                self.forward()
                total = 0.0
                for i in range(self.N):
                    total = total + (self.X[i]-Xpre[i])**2
            else:
                return True
        return False


    def convergeBackward(self):
        Ypre = np.zeros(self.N)
        total = self.N*1.0
        for t in range(self.maxSteps):
            if(total>self.divThresh):
                Ypre=self.Y.copy()
                self.backward()
                total = 0.0
                for i in range(self.N):
                    total = total + (self.Y[i]-Ypre[i])**2
            else:
                return True
        return False


    def setInput(self,inputID,inputVal):
        self.reset()
        for i in range(len(inputID)):
            self.Input[inputID[i]] = inputVal[i]

    def run(self, K, errorSamplePeriod=1000):
        Error = []
        errMin = 1e9
        for k in range(K):
            if(k%errorSamplePeriod):
                i = int(np.floor(np.random.rand()*len(self.output)))
                self.setInput(self.inputID,self.input[:,i])
                self.convergeForward()
                self.setError(self.outputID,[self.output[i]])
                self.convergeBackward()
                self.weightUpdate()
            else:
                err = 0.
                count = 0
                for i in range(len(self.output)):
                    self.setInput(self.inputID,self.input[:,i])
                    self.convergeForward()
                    self.setError(self.outputID,[self.output[i]])
                    err += self.getError()
                    count = count+1
                err = err/(count*1.0)
                if(err<errMin):
                    errMin = err
                if (err>2.0*errMin):
                    self.W = self.Wbest
                else:
                    self.Wbest = self.W.copy()
                Error = np.hstack([Error, err])

                if(k%(K/100)==0):
                    print("progress: "+str(100.0*k/K)+"%")

        self.W = self.Wbest
        return Error
