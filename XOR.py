import h5py

In = [0.0,0.0,0.0,1.0,1.0,0.0,1.0,1.0]
Out = [0.0,1.0,1.0,0.0]

hf = h5py.File('XORmap.h5', 'w')
hf.create_dataset('In', data=In)
hf.create_dataset('Out', data=Out)
hf.close()

pre = [0,0,1,1,2,3,4,4,4]
post = [2,3,2,3,3,2,2,3,4]
inputNodes = [0,1]
outputNodes = [3]

hf = h5py.File('XORnet.h5', 'w')
hf.create_dataset('pre', data=pre)
hf.create_dataset('post', data=post)
hf.create_dataset('inputNodes', data=inputNodes)
hf.create_dataset('outputNodes', data=outputNodes)
hf.close()


