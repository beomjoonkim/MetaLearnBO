import numpy as np

class Function(object):
  def __init__(self):
    raise NotImplemented

  def __call__(self,x): 
    raise NotImplemented

class Domain(object):
  '''
  Domain only supports 
  (0) continuous hyper rectangles in R^d
  (1) discrete and finite set of items
  '''
  CONTINUOUS = 0
  DISCRETE = 1
  TYPES = [CONTINUOUS, DISCRETE]

  def __init__(self, space_type, domain):
    assert(space_type in self.TYPES)
    self.space_type = space_type

    # check validity of domain
    if space_type == self.CONTINUOUS:
      domain = np.array(domain)
      assert(domain.ndim == 2)
      assert((domain[1]-domain[0] > 0).all())
    elif space_type == self.DISCRETE:
      assert(type(domain) is list)

    self.domain = domain

class DiscreteObjFcn( Function ):
  def __init__(self,y_values):
    self.domain = Domain(space_type=1,domain=range(len(y_values)))
    self.y_values = y_values  
    self.fg=None

  def __call__(self,x_idx):
    return self.y_values[x_idx]

class ContinuousObjFcn( Function ):
  def __init__(self,domain):
    self.domain = Domain(space_type=0,domain=domain)
    self.fg=None

  def __call__(self,x):
    raise NotImplemented