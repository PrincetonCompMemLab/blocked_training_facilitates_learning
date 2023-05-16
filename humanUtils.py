import numpy as np

## utils

def mov_avg(A,window):
  N = len(A)-window
  M = -np.ones([N])
  for t in range(N):
    M[t] = A[t:t+window].mean()
  return M


## interface with model

node2stateD = {
  "BEGIN":0,
  "LOCNODEB":1,
  "LOCNODEC":2,
  "NODE11":3,
  "NODE12":4,
  "NODE21":5,
  "NODE22":6,
  "NODE31":7,
  "NODE32":8,
  "END":9
}

ALL_CONDITIONS = [

    'blocked',
    'interleaved',
    'blocked_rep',
    'interleaved_rep',
    'explicit_interleaved',
    'inserted_early',
    'inserted_middle',
    'inserted_late',
    'inserted_early_rep',
    'inserted_middle_rep',
    'inserted_late_rep'
]