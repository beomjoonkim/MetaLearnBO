import numpy as np
import helper

class KtimesRandomSearch(object):
  def __init__(self,obj_fcn,k):
    self.evaled_x   = []
    self.evaled_y   = []
    self.obj_fcn    = obj_fcn
    self.domain     = self.obj_fcn.domain
    self.k          = k
    self.eval_order = np.random.permutation(self.domain.domain)
    self.order      = 0

  def choose_next_point(self):
    x = self.eval_order[self.order]
    y = self.obj_fcn(x)
    self.evaled_x.append(x)
    self.evaled_y.append(y)   

    self.order+=1
    return x, y
    
  def generate_evals(self, T):
    for i in range(T):
      yield self.choose_next_point()
  
class KtimesContRandomSearch(object):
  def __init__(self,obj_fcn,k):
    self.evaled_x   = []
    self.evaled_y   = []
    self.obj_fcn    = obj_fcn
    self.domain     = self.obj_fcn.domain
    self.k          = k
    self.order      = 0

  def choose_next_point(self):
    x = np.random.uniform(low=self.domain.domain[0,:],high=self.domain.domain[1,:])
    y = self.obj_fcn(x)
    return x, y
    
  def generate_evals(self, T):
    for i in range(T):
      yield self.choose_next_point()
  
