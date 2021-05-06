import numpy as np
from visdom import Visdom
vis=Visdom(env="heat_map")
 
a=np.concatenate((np.arange(0,255),np.arange(255,0,-1)))
b=np.concatenate((np.arange(0,255),np.arange(255,0,-1)))

print(a.shape)
print(b.shape)
X =np.outer(a,b)
print(X.shape)

