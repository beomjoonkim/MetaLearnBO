import numpy as np
import scipy.io as sio
import os
import sys
import threading

from Queue import Queue
from multiprocessing.pool import ThreadPool # dummy is nothing but multiprocessing but wrapper around threading
from multiprocessing import cpu_count


import pickle
import socket
import argparse
import csv
import time
import itertools
import sys

def worker_p(config):
  domain  = config[0]
  dtype   = config[1]
  pidx    = config[2]
  alg     = config[3]
  if dtype=='partial':
    p = config[4]
    trial   = config[5]
    Nportion= config[6]
    BO      =config[7]
    command = 'python ./run_openrave_domains.py -domain '+domain \
              + ' -data_type ' + dtype + ' -pidx ' \
              + str(pidx) + ' -trial '+str(trial) + ' -Nportion ' + str(Nportion)\
              + ' -bo ' + BO
  elif dtype=='cont':
    trial = config[4]
    Nportion= config[5]
    BO =config[6]
    #command = 'python ./run_experiments.py -d '+domain \
    #          + ' -dtype ' + dtype + ' -pidx ' \
    #          + str(pidx) + ' -trial '+str(trial) + ' -Nportion ' + str(Nportion)\
    #          + ' -a ' + alg + ' -bo '+BO \
    #          + ' -n_output '+ str(500)
    if alg == 'zbk':
      command = 'python ./run_experiments.py -algorithm ' +alg +' -domain '+domain \
               + ' -pidx ' + str(pidx)  + ' -trial '+str(trial) + ' -Nportion '+str(Nportion) + ' -bo '+BO
    else:
      command = 'python ./run_experiments.py -algorithm ' +alg +' -domain '+domain \
               + ' -pidx ' + str(pidx)  + ' -trial '+str(trial) + ' -bo '+ BO
  else:
    Nportion= config[4]
    BO =config[5]
    command = 'python ./run_openrave_domains.py -d '+domain \
                + ' -dtype ' +dtype + ' -pidx '+str(pidx)  \
                + ' -bo ' + BO + ' -Nportion ' + str(Nportion)\
                + ' -a ' + alg \

  print command + "\n"
  os.system(command)

def worker_wrapper_multi_input(multi_args):
  return worker_p(multi_args)

def main():
  dtype       = sys.argv[1]
  domain_name = [sys.argv[2]]
  dtype       = [dtype]
  bo        = ['pi']
  alg_name  = ['zbk']
  pidx      = range(100)
  #alg_name  = ['plain']
  if alg_name[0] == 'zbk':
    #Nportions = [0.1,0.3,0.5,0.7,0.9]
    Nportions = [1]
  else:
    Nportions = [1]

  if dtype[0]=='partial':
    # For learning curve, just do n missing data = 0.6
    if len(Nportions) > 1:
      p=[0.6]
    elif len(Nportions)==1:
      p = [0.3,0.6,0.8]
    trials = range(1,2)
    configs   = list(itertools.product( domain_name,dtype,pidx,alg_name,p,trials,Nportions,bo ))
  elif dtype[0]=='cont':
    trials=[0,1,2,3]
    configs   = list(itertools.product( domain_name,dtype,pidx,alg_name,trials,Nportions,bo ))
  else:
    configs   = list(itertools.product( domain_name,dtype,pidx,alg_name,Nportions,bo ))

  n_workers = int(cpu_count())
  pool = ThreadPool(n_workers)
  results = pool.map(worker_wrapper_multi_input,configs)

if __name__ == '__main__':
  main()


