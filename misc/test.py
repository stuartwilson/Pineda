import numpy as np
import pylab as pl
from network import Network

'''
    DEFINE YOUR NEURAL NETWORK ARCHITECTURE
'''

# Define a network and declare the number of nodes
P = Network(5)

# Identify the input and output nodes
P.setInputID([0,1])
P.setOutputID([3])

# Define the connections (presynaptic node, postsynaptic node)
P.connect(0,2)
P.connect(0,3)
P.connect(1,2)
P.connect(1,3)
P.connect(2,3)
P.connect(3,2)
P.connect(4,2)
P.connect(4,3)
P.connect(4,4)

# Finalize network
P.postConnect()
P.randomizeWeights(-1.0, +1.0)

'''
    DEFINE THE TRAINING PATTERNS (INPUTS AND OUTPUTS)
'''

P.setMapInput(np.array([[0.0,0.0,1.0,1.0],[0.0,1.0,0.0,1.0]]))
P.setMapOutput(np.array([0.0,1.0,1.0,0.0]))


'''
    TRAIN THE NETWORK
'''

# Run the network for a given number of steps
Error = P.run(100000)


'''
    PRINT OUT THE RESPONSE OF THE OUTPUT NODE TO EACH INPUT PATTERN
'''

for i in range(len(P.output)):
    P.setInput(P.inputID,P.input[:,i])
    P.convergeForward()
    print('target='+str(P.output[i])+', output='+str(P.X[P.outputID]))


'''
    PLOT A GRAPH OF THE NETWORK ERROR CHANGING OVER TIME
'''

F = pl.figure()
f = F.add_subplot(111)
f.plot(Error)
f.set_xlabel('iterations (x1000)')
f.set_ylabel('error')

pl.show()

