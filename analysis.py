import numpy as np
import pylab as pl
import h5py
import sys

inFname = sys.argv[1]
outFname = sys.argv[2]

F = h5py.File(inFname)
inp = F['In'][:]
tar = F['Out'][:]
F.close()

inp = inp.reshape([4,2])
in1 = inp[:,0]
in2 = inp[:,1]

tar = tar.reshape([4,1])


F = h5py.File(outFname)
E = F['error'][:]
R = F['response'][:]
F.close()

out = R.reshape([4,1])

F = pl.figure()
f = F.add_subplot(121)
f.plot(E)
f.set_xlabel('time')
f.set_ylabel('error')
f.set_aspect(np.diff(f.get_xlim())/np.diff(f.get_ylim()))


f = F.add_subplot(122)
cmap = pl.cm.get_cmap('coolwarm')
P = len(in1);
for p in range(P):
    circ = pl.Circle([in1[p],in2[p]],out[p]*0.5,color=cmap(out[p])[0])
    f.add_patch(circ)
f.axis([-0.5,1.5,-0.5,1.5])
f.set_aspect(np.diff(f.get_xlim())/np.diff(f.get_ylim()))
f.set_xticks([0,1])
f.set_yticks([0,1])
f.set_xlabel('in1')
f.set_ylabel('in2')
f.set_title('response')
pl.show()

