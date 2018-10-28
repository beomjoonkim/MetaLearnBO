import time
import numpy as np

import cvxpy as cvx

def complete_matrix(A,mask,mu):
  #X = Variable(*A.shape)
  X = cvx.Variable((A.shape[0],A.shape[1]))
  objective = Minimize(normNuc(X)) 
  problem = Problem(objective, [multiply(mask,X) == multiply(mask,A)])
  print "Completing matrix..."
  stime = time.time()
  problem.solve(solver=SCS)
  print "Done. It took %f seconds"%(time.time()-stime)
  return X.value

def test_complete_matrix():
  A = np.random.random((10,10))
  mask = np.zeros((10,10))
  mask[1,1]=1
  mask[2,1]=1
  mask[3,2]=1
  mask[4,3]=1
  mask[0,4]=1
  X = complete_matrix(A,mask,2)
  import pdb;pdb.set_trace()

if __name__ == '__main__':
  test_complete_matrix()
  

  
