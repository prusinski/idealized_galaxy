import pickle
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
# x and y are arrays containing the values of the x and y axes
# image is the array containing the values of the image plane
image, x, y = pickle.load( open( "image.p", "rb" ) )
plt.pcolormesh(x,y,image.T, norm=LogNorm())
plt.xscale('log')
plt.colorbar()
plt.axis([x.min(), x.max(), y.min(), y.max()])
plt.savefig('image.png')

