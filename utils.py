
import pandas as pd
import numpy as np
import seaborn as sns
import pdb
import os
import pickle
from io import BytesIO
from scipy.stats import truncnorm
import matplotlib.pyplot as plt
import requests
r = requests.get("https://raw.githubusercontent.com/PrincetonCompMemLab/blocked_training_facilitates_learning/master/model.py")
with open('model.py', 'w') as f:
    f.write(r.text)
from model import *
## import human data for fitting

FLAG_SMACC = True
SMTEMP = 4  

def param2str(args):
    param_str = "-".join(["%s_%.3f"%(i,j) for i,j in args['sch'].items()])
    param_str += "-"+"-".join(["%s_%.3f"%(i,j) for i,j in args['sem'].items()])
    return param_str

def softmax_custom(x,tau):
    return np.exp(x*tau)/np.sum(np.exp(x*tau))

def get_sm(xth,norm=True):
  """ 
  given x_t_hat from subject
  [trial,layer,node]
  get 2afc normalized softmax for layer 2/3
  return: [layer2/3,trial,node56/78]
  norm=true 
   apply softmax to xth
   when prediction done with multiple schemas
  """
  nodes = {2:(5,6),3:(7,8)} 
  L = [] # layer 2 and 3
  for l,ns in nodes.items():
    # l is the time point
    # ns are the states we want to get the posterior of
    y = xth[:,l,ns]
    # y contains the information of the posterior of the next state for all trials
    # for this particular time point
    # print(y.shape)
    # assert False
    if norm:
      # y = softmax(y,1)

      y = np.array([softmax_custom(yt,SMTEMP) for yt in y])
    L.append(y)
  # so L contains 2 200 by 2 values
  return np.array(L)

def get_acc(data,acc_mode=FLAG_SMACC):
    """ 
    returns 2afc softmax of 
    layer 2/3 transitions
    single seed
    """
    if acc_mode: # compute softmax acc
        # for this seed and condition, xth is trial by time point by probability distribution over states in that time point
        yhat_sm = get_sm(data['xth']) # yhat_sm
        # yhat_sm is 2 by 200 by 2 so for each of the starting nodes
        # the target time point is here for each trial
        # print(data['xth'],yhat_sm)
        L = []
        # loop over layers
        # loop is over the two time points that we are computing accuracy for
        # namely the timepoints 2 -> 3 and 3 -> 4 transitions
        for i in range(2):
            # yhat_sm_l for this particular transition is trial by the softmaxed probabilities of the two targets
            yhat_sm_l = yhat_sm[i,:,:] # softmax of layer (float)
            yt = data['exp'][:,i+3] # target tonode (int)
            # pr_yt is the probability of the target node, so if the target is 5
            # and the current node is i = 0, then yt - (5+2*i) is a 0 or 1 for each trial
            # and yt is 5, then 5 - 5 = 0 which is the correct, but if target was 
            # 6 then we would get index 1 so that is great
            pr_yt = yhat_sm_l[range(len(yhat_sm_l)),yt - (5+2*i)] # 
            L.append(pr_yt)
        return np.array(L)
    else: # compute score
        xth = data['xth']
        resp = xth.argmax(-1)
        exp = data['exp']
        score = resp[:,2:4] == exp[:,3:5]
        # transpose required for backcompatibility
        return score.T


def unpack_acc(cbatch_data,mean_over_tsteps=True):
    """ 
    given cbatch data (data from multiple curr and seeds)
    return acc [curr,seed,trial]
    """
    accL = [] # curr
    for cidx in range(len(cbatch_data)):
        acc = np.array([get_acc(sbatch) for sbatch in cbatch_data[cidx]])
        if mean_over_tsteps:
            # this is the mean over the two time steps for each trial
            accL.append(acc.mean(1)) # mean over layers
        else:
            accL.append(acc)
    return np.array(accL)


def unpack_data(cbatch_data,dtype='priors'):
    """ unpacks batch data from multiple curr and seeds
    dtype: priors,likes,post
    """
    L = []
   
    for cidx in range(len(cbatch_data)):
        L.append([])
        # for each seed in the data for this condition for this particular
        for sidx,sbatch_data in enumerate(cbatch_data[cidx]):
            # print(sidx,sbatch_data[dtype].shape)
            # assert False
            # np.any on axis 0 basically looks at each of the 205 by 5 matrices stacked on each other
            # and in each of those entries, ask if anywhere in the stack is true
            mask = np.any(sbatch_data[dtype]!=-1,0)[0]
            # print(mask)
            # list of lists indexed by first condition and then seeds
            L[cidx].append(sbatch_data[dtype][:,:,mask])
    return L


### RUN EXP
def run_batch_exp(ns,args, concentration_info = None, 
                  stickiness_info = None,
                  sparsity_info = None):
  """ exp over seeds, 
  single task_condition / param config
  return full data
  """
  dataL = []
  c_list = [] # concentration
  s_list = [] # stickiness
  spars_list = [] # sparsity

  if concentration_info != None:
    concentration_lower,concentration_upper = concentration_info["concentration_lb"], concentration_info["concentration_ub"]
    concentration_mu, concentration_sigma = concentration_info["concentration_mean"], concentration_info["concentration_sd"]

  if stickiness_info != None:
    stickiness_lower,stickiness_upper = stickiness_info["stickiness_lb"], stickiness_info["stickiness_ub"]
    stickiness_mu, stickiness_sigma = stickiness_info["stickiness_mean"], stickiness_info["stickiness_sd"]

  if sparsity_info != None:
    sparsity_lower,sparsity_upper = sparsity_info["sparsity_lb"], sparsity_info["sparsity_ub"]
    sparsity_mu, sparsity_sigma = sparsity_info["sparsity_mean"], sparsity_info["sparsity_sd"]


  for i in range(ns):
    # only the concentrationAcross will have this parameter in here
    if concentration_info != None:
        args['sch']["concentration"] = truncnorm.rvs(
                                      a = (concentration_lower - concentration_mu) / concentration_sigma, 
                                      b = (concentration_upper - concentration_mu) / concentration_sigma,
                                      loc = concentration_mu, 
                                      scale = concentration_sigma, 
                                      size = 1)[0]
        c_list.append(args['sch']["concentration"])

    if stickiness_info != None:
      new_sticky = truncnorm.rvs(
                                a = (stickiness_lower - stickiness_mu) / stickiness_sigma, 
                                b = (stickiness_upper - stickiness_mu) / stickiness_sigma,
                                loc = stickiness_mu, 
                                scale = stickiness_sigma, 
                                size = 1)[0]
      args['sch']["stickiness_wi"] = new_sticky
      args['sch']["stickiness_bt"] = new_sticky
      s_list.append(new_sticky)

    if sparsity_info != None:
      args['sch']["sparsity"] = truncnorm.rvs(
                                    a = (sparsity_lower - sparsity_mu) / sparsity_sigma, 
                                    b = (sparsity_upper - sparsity_mu) / sparsity_sigma,
                                    loc = sparsity_mu, 
                                    scale = sparsity_sigma, 
                                    size = 1)[0]
      spars_list.append(args['sch']["sparsity"])

    #print("seed: ",args['sch']["concentration"])
    task = Task()
    # sem contains skipt1,ppd_allsch so unpack it
    sem = SEM(schargs=args['sch'],**args['sem'])
    exp,curr  = task.generate_experiment(**args['exp'])
    # the experiment is all of the paths we want to give at each time point
    # and the condition in args['exp'] helps to convey what 
    data = sem.run_exp(exp)
    data['exp']=exp
    dataL.append(data)
  # if concentration_info != None:
  #   plt.hist(c_list)
  #   plt.show()
  return dataL, c_list, s_list, spars_list

def run_batch_exp_sim4(ns,args, condition = None, concentration_info = None, 
                  stickiness_info = None,
                  sparsity_info = None):
  """ exp over seeds, 
  single task_condition / param config
  return full data
  """
  dataL = []
  c_list = [] # concentration
  s_list = [] # stickiness
  spars_list = [] # sparsity

  if concentration_info != None:
    lower,upper = concentration_info["concentration_lb"], concentration_info["concentration_ub"]
    mu, sigma = concentration_info["concentration_mean"], concentration_info["concentration_sd"]

  if stickiness_info != None:
    lower,upper = stickiness_info["stickiness_lb"], stickiness_info["stickiness_ub"]
    mu, sigma = stickiness_info["stickiness_mean"], stickiness_info["stickiness_sd"]

  if sparsity_info != None:
    lower,upper = sparsity_info["sparsity_lb"], sparsity_info["sparsity_ub"]
    mu, sigma = sparsity_info["sparsity_mean"], sparsity_info["sparsity_sd"]


  for i in range(ns):
    # only the concentrationAcross will have this parameter in here
    if concentration_info != None:
        args['sch']["concentration"] = truncnorm.rvs(
                                      a = (lower - mu) / sigma, 
                                      b = (upper - mu) / sigma,
                                      loc = mu, 
                                      scale = sigma, 
                                      size = 1)[0]
        c_list.append(args['sch']["concentration"])

    if stickiness_info != None:
      new_sticky = truncnorm.rvs(
                                a = (lower - mu) / sigma, 
                                b = (upper - mu) / sigma,
                                loc = mu, 
                                scale = sigma, 
                                size = 1)[0]
      args['sch']["stickiness_wi"] = new_sticky
      args['sch']["stickiness_bt"] = new_sticky
      s_list.append(new_sticky)

    if sparsity_info != None:
      args['sch']["sparsity"] = truncnorm.rvs(
                                    a = (lower - mu) / sigma, 
                                    b = (upper - mu) / sigma,
                                    loc = mu, 
                                    scale = sigma, 
                                    size = 1)[0]
      spars_list.append(args['sch']["sparsity"])

    #print("seed: ",args['sch']["concentration"])
    task = Task()
    # sem contains skipt1,ppd_allsch so unpack it
    sem = SEM(schargs=args['sch'],**args['sem'])
    exp,curr  = task.generate_experiment(**args['exp'])
    # the experiment is all of the paths we want to give at each time point
    # and the condition in args['exp'] helps to convey what 
  
    #print("condition: ", condition)
    #pdb.set_trace()
    if (condition == "blocked") or (condition == "early"):
      data = sem.run_exp_sim4_blocked(exp,40)
    elif (condition == "middle"):
      data = sem.run_exp_sim4_blocked(exp,80)
    elif (condition == "late"):
      data = sem.run_exp_sim4_blocked(exp,120)
    else:
      data = sem.run_exp(exp)
    data['exp']=exp
    dataL.append(data)
  # if concentration_info != None:
  #   plt.hist(c_list)
  #   plt.show()
  return dataL, c_list, s_list, spars_list

def run_exps_feedHumanExp(condition_to_participant_to_exp_path, args):
  # create the participant to concentration variance here!

  condition_to_participant_to_measures = {} # measures are like accuracy and so on
  condition_to_participant_to_exp = pickle.load(BytesIO(requests.get(condition_to_participant_to_exp_path).content))
  for condition in condition_to_participant_to_exp:
    condition_to_participant_to_measures[condition] = {}
    for participant in condition_to_participant_to_exp[condition]:
      exp = condition_to_participant_to_exp[condition][participant]
      sem = SEM(schargs=args['sch'],**args['sem'])
      data = sem.run_exp(exp)
      data['exp']=exp
      condition_to_participant_to_measures[condition][participant] = data
  return condition_to_participant_to_measures

def run_exps_feedHumanExp_v7(participant_to_exp_path, args):
  # create the participant to concentration variance here!
  participant_to_measures = {} # measures are like accuracy and so on
  participant_to_exp = pickle.load(BytesIO(requests.get(participant_to_exp_path).content))
  
  for participant in participant_to_exp:
    exp = participant_to_exp[participant]
 
    sem = SEM(schargs=args['sch'],**args['sem'])
    data = sem.run_exp(exp)
    data['exp']=exp
    participant_to_measures[participant] = data
  return participant_to_measures

def unpack_acc_feedHuman_v7(pid_to_data,
                        mean_over_tsteps=False):
    all_pids = [x for x in range(len(pid_to_data.keys()))]
    acc = np.array([get_acc(pid_to_data[pid]) for pid in all_pids]) # equivalent to over seeds
    if mean_over_tsteps:
        # this is the mean over the two time steps for each trial
        return acc.mean(1)
    else:
        return acc

def unpack_acc_feedHuman(condition_to_participant_to_measures,
                        mean_over_tsteps=False):
    """ 
  
    """
    condition_to_accuracy = {}
    for condition in condition_to_participant_to_measures:
        pid_to_data = condition_to_participant_to_measures[condition]
        all_pids = condition_to_participant_to_measures[condition].keys()
        acc = np.array([get_acc(pid_to_data[pid]) for pid in all_pids]) # equivalent to over seeds
        if mean_over_tsteps:
            # this is the mean over the two time steps for each trial
            condition_to_accuracy[condition] = acc.mean(1)
        else:
            condition_to_accuracy[condition] = acc
    return condition_to_accuracy


def run_batch_exp_curr(ns,args,currL=['blocked','interleaved'], 
                  concentration_info = None, stickiness_info = None,
                  sparsity_info = None):
  """ loop over task conditions, 
  return acc [task_condition,seed,trial]
  """
  accL = []
  dataL = []
  c_list_each_condition = []
  s_list_each_condition = []
  spars_list_each_condition = []
  # dataD = {}
  for curr in currL:
    args['exp']['condition'] = curr
    ## extract other data here
    data_batch, c_list, s_list, spars_list = run_batch_exp(ns,args, 
                          concentration_info = concentration_info,
                          stickiness_info= stickiness_info,
                          sparsity_info = sparsity_info)
    c_list_each_condition.append(c_list)
    s_list_each_condition.append(s_list)
    spars_list_each_condition.append(spars_list)
    dataL.append(data_batch)
    ## unpack seeds and take mean over layers
    # for each seed in this condition, get the accuracy over the 200 t
    acc = np.array([get_acc(data) for data in data_batch]).mean(1) # mean over layer
    accL.append(acc)
  # data L is a list of 5 on the outer layer for the 5 conditions
  # then for one condition, it is a list of num seeds 
  # and one element contains one of the outputs from sem.run_exp()
  return dataL, c_list_each_condition, s_list_each_condition, spars_list_each_condition

def run_batch_exp_curr_sim4(ns,args,currL=['blocked','interleaved'], 
                  concentration_info = None, stickiness_info = None,
                  sparsity_info = None):
  """ loop over task conditions, 
  return acc [task_condition,seed,trial]
  """
  accL = []
  dataL = []
  c_list_each_condition = []
  s_list_each_condition = []
  spars_list_each_condition = []
  # dataD = {}
  for curr in currL:
    args['exp']['condition'] = curr
    ## extract other data here
    data_batch, c_list, s_list, spars_list = run_batch_exp_sim4(ns,args,
                          condition = curr, 
                          concentration_info = concentration_info,
                          stickiness_info= stickiness_info,
                          sparsity_info = sparsity_info)
    c_list_each_condition.append(c_list)
    s_list_each_condition.append(s_list)
    spars_list_each_condition.append(spars_list)
    dataL.append(data_batch)
    ## unpack seeds and take mean over layers
    # for each seed in this condition, get the accuracy over the 200 t
    acc = np.array([get_acc(data) for data in data_batch]).mean(1) # mean over layer
    accL.append(acc)
  # data L is a list of 5 on the outer layer for the 5 conditions
  # then for one condition, it is a list of num seeds 
  # and one element contains one of the outputs from sem.run_exp()
  return dataL, c_list_each_condition, s_list_each_condition, spars_list_each_condition

## plotting

