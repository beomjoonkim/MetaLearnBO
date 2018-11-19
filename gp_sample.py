import numpy as np
from functions import ContinuousObjFcn

# Synthetic objective funtion for testing the continuous case
class SynthObjFcn(ContinuousObjFcn):
  def __init__(self, domain, rnd_seed, n_features, bfunc=np.cos, 
               noise_sigma=0.1, wiggliness=1.):
    # rnd_seed: the seed of randomness. Each rnd_seed corresponds to one deterministic function
    # bfunc: basis function form
    # wiggliness: the function is more wiggly if it's higher
    super(SynthObjFcn,self).__init__(domain)
    self.rnd_stream = np.random.RandomState(rnd_seed)
    self.n_features = n_features
    self.Sigma = self.rnd_stream.rand(n_features,n_features*10)
    self.Sigma = np.dot(self.Sigma, self.Sigma.T)# + np.eye(n_features)*0.1
    self.mu = self.rnd_stream.rand(n_features)
    self.bfunc = bfunc
    self.noise_sigma = noise_sigma
    self.dim = len(domain[0])
    self.wiggliness = wiggliness
    self.sample_func()

  def phi(self, x):
    # x : N * d
    # phi: n_features * N
    phi = np.sqrt(2.0/self.n_features)*\
            self.bfunc(np.dot(self.b,x.T)+np.tile(self.c,( 1, x.shape[0] ) ) )
    return phi

  def sample_func(self):
    # sample basis func
    self.b = self.rnd_stream.randn(self.n_features,self.dim) * self.wiggliness
    self.c = self.rnd_stream.uniform(-np.pi,np.pi, (self.n_features,1))

    # sample coefficients
    self.w = self.rnd_stream.multivariate_normal(self.mu, self.Sigma)

  def __call__(self, x):
    if len(x.shape)==1:
      x = x[None,:]
    y= np.dot(self.phi(x).T,self.w) + self.rnd_stream.normal(0, self.noise_sigma, len(x))
    return np.array(y).squeeze()

def test():
  domain = [[-10.],[10.]]
  rnd_seed = 7
  n_features = 1000
  syn_fcn = SynthObjFcn(domain, rnd_seed, n_features, noise_sigma=0.01, wiggliness=2.)
  xvec = np.array([np.linspace(-10,10,500)]).T
  y = syn_fcn(xvec)
  import matplotlib.pyplot as plt
  plt.plot(xvec.T[0], y)
  plt.show()

def test2d():
  domain = [[-10., -10.],[10., 10.]]
  rnd_seed = 12
  n_features = 1000
  syn_fcn = SynthObjFcn(domain, rnd_seed, n_features, noise_sigma=0.01)
  x, y = np.meshgrid(np.linspace(-10,10,100), np.linspace(-10,10,100))
  xy = np.array([x.ravel(),y.ravel()]).T
  z = syn_fcn(xy)
  import matplotlib.pyplot as plt
  fig = plt.figure()
  from mpl_toolkits.mplot3d import Axes3D
  ax = Axes3D(fig)
  from matplotlib import cm

  surf = ax.plot_surface(x, y, z.reshape(100,100), cmap=cm.coolwarm, linewidth=0)
  plt.show()

