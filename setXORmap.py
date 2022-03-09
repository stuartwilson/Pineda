import h5py

In = [0.0,0.0,0.0,1.0,1.0,0.0,1.0,1.0]
Out = [0.0,1.0,1.0,0.0]

hf = h5py.File('XORmap.h5', 'w')
hf.create_dataset('In', data=In)
hf.create_dataset('Out', data=Out)
hf.close()


