import pandas as pd
import numpy as np
import json

# filler question exclusion threshold 
THRESHOLD = 0.9

# this function gets the db from psyturk
#
def get_sql_df(exp_version=None,verb=False):
  import sqlite3 as sql
  from glob import glob as glob
  if exp_version == "RT40B1000cl":
      db_fpath = './CSWparticipants.db'
  else:
    db_fpath = './csw_mturk_spring19.db'
  print(db_fpath)
  with sql.connect(db_fpath) as conn:
    # connection objecs represent the database
    # cursor objects point to rows in the database
    c = conn.cursor()

    # list tables in database
    c.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables_in_db = c.fetchall()
    table_name = "CSWfall18" # table name must be constant to prevent repeat subjects
    print(tables_in_db)

    # table header
    c.execute("PRAGMA table_info(%s)"%table_name)
    col_names = [i[1] for i in c.fetchall()]

    # select everything within table
    db_datastring = c.execute("""SELECT * FROM %s"""%table_name).fetchall()
    sql_df = pd.DataFrame(db_datastring,columns=col_names)
    
    # remove unneeded rows 
    sql_df = sql_df[sql_df['mode'] != 'debug'] # debug rows
    sql_df = sql_df[pd.notnull(sql_df['datastring'])] # rows with no datastring 
    # return only select code version
    versions = sql_df.codeversion.unique()
    if verb:
        print('exp versions found in table:')
        for v in versions: print(v)
    # if version not specified, take latest
    if exp_version==None:
      exp_version = versions[-1]
    print('\n-loading code version:',exp_version)
    sql_df = sql_df[sql_df.codeversion == exp_version]
    print('N =',len(sql_df),'subjects')
  return sql_df


def get_dataset_code(CONDITION):
  # if CONDITION == 0 or CONDITION == 'interleaved':
  #   dataset_code = 'i1000cl'
  if CONDITION == 0 or CONDITION == 'interleaved':
    dataset_code = 'RT01B1000cl'
  elif CONDITION == 1 or CONDITION == 'interleaved_rep':
    dataset_code = 'csw1000block01.04.25.19'
  elif CONDITION == 2 or CONDITION == 'blocked':
    dataset_code = 'RT40B1000cl'
  elif CONDITION == 3 or CONDITION == 'blocked_rep':
    dataset_code = 'csw1000block40.04.07.19'
  elif CONDITION == 4 or CONDITION == 'explicit_interleaved':
    dataset_code = 'csw1000boldmdp.05.09.19'
  elif CONDITION == 5 or CONDITION == 'inserted_early':
    dataset_code = 'csw1000insertedblock0.09.22.19'
  elif CONDITION == 6 or CONDITION == 'inserted_early_rep':
    dataset_code = 'csw1000insertedblock0.10.08.19'
  elif CONDITION == 7 or CONDITION == 'inserted_middle':
    dataset_code = 'csw1000insertedblock1.09.30.19'
  elif CONDITION == 8 or CONDITION == 'inserted_middle_rep':
    dataset_code = 'csw1000insertedblock1.10.07.19'
  elif CONDITION == 9 or CONDITION == 'inserted_late':
    dataset_code = 'csw1000insertedblock2.10.09.19'
  elif CONDITION == 10 or CONDITION == 'inserted_late_rep':
    dataset_code = 'csw1000insertedblock2.10.10.19'
  else:
    print("INVALID CONDITION")
    assert False
  return dataset_code


def get_block_indices(subj_df,num_stories=110,num_blocks=11):
  """ given a subj_df
  returns the indices of when blocks begins and ends
  which can be used to index a block of stories in subj_df 
  """
  # get indices for begining of stories
  begin_story_bool = subj_df.state == 'BEGIN'
  begin_story_idx = subj_df[begin_story_bool].index
  # get indices for begin and end of blocks
  block_linsp = np.arange(0,109,10)
  begin_block_idx = begin_story_idx[block_linsp]
  end_block_idx = np.hstack([begin_block_idx[1:],np.array([len(subj_df)])])
  return begin_block_idx,end_block_idx


def include_block_idx(subj_df):
  """ inplace column
  """
  subj_df['block'] = 0
  begin_block_idx,end_block_idx = get_block_indices(subj_df)
  for block_num,(bidx,eidx) in enumerate(zip(begin_block_idx,end_block_idx)):
    subj_df.loc[bidx:eidx,'block'] = int(block_num)
  return subj_df


def include_story_idx(subj_df,num_stories):
  """ inplace include column with stories
  """
  bidx = subj_df.loc[subj_df['state']=='BEGIN'].index
  subj_df.loc[bidx,'story'] = np.arange(num_stories,dtype=np.int)
  subj_df.story = np.floor(subj_df.story.interpolate()).astype('int')
  return subj_df


frnode2depth = lambda x: DEPTH_DICT[x]
DEPTH_DICT = {np.nan:100,'BEGIN':0,
              'NODE11':1,'NODE12':1,
              'NODE21':2,'NODE22':2,
              'NODE31':3,'NODE32':3,
              'END':4}


def make_subj_df(psiturk_dstr,num_instruction_trials=3,stories_per_block=40):
  # initialize
  subj_df = pd.DataFrame([t['trialdata'] for t in json.loads(psiturk_dstr)['data']])
  # remove instruction trials
  subj_df = subj_df.iloc[num_instruction_trials:]
  # subj response
  subj_df['response_right'] = subj_df['key_press'] == 39.0 
  subj_df['true_on_right'] = (subj_df['true_on_right'] == "True")
  subj_df['correct_response'] = (subj_df['true_on_right'] == subj_df['response_right']).astype(int)
  # only keep some columns
  subj_df = subj_df[['type','qtype',
                     'state','fromnode','true_tonode','false_tonode',
                     'rfc','true_rfc','false_rfc',
                     'response_right','true_on_right','correct_response',]]
  ## customizing
  # story columns
  subj_df = include_story_idx(subj_df,num_stories=200)
  # edges
  subj_df['edge'] = subj_df['fromnode'] + subj_df['true_tonode']
  # blocks
  subj_df['block'] = subj_df.story//stories_per_block
  # depth
  subj_df['depth']=subj_df.fromnode.apply(frnode2depth)
  return subj_df


def make_group_df(sql_df):
  """ looping `make_subj_df`
  """
  subj_df_L = []
  # loop making subject dfs
  for s in range(len(sql_df)):
    psiturk_dstr = sql_df['datastring'].iloc[s] 
    subj_df = make_subj_df(psiturk_dstr)
    subj_df['subjnum']=s
    subj_df_L.append(subj_df)
  # make group dfs
  group_df = pd.concat(subj_df_L)
  # reshape
  group_df = group_df.set_index(['subjnum','block','story','depth',group_df.index])
  return group_df


def get_thresholded_tqdf(group_tqdf,group_fqdf,threshold):
  """ takes in dataframe
  returns dataframe
  """
  # thresholding
  # make sure the correct response to filler questions have an accuracy above
  # a threshold
  sub_mask = group_fqdf.correct_response.groupby('subjnum').mean() > threshold
  sub_mask = sub_mask[sub_mask].index.values
  masked_group_df = group_tqdf.loc[sub_mask,:,:]
  return masked_group_df


def select_transition_probes(group_tqdf):
  """ remove transition from BEGIN node"""
  return group_tqdf[group_tqdf['fromnode']!='BEGIN']
  

## database pipeline


def load_dfs(condition):
  ## load psiturk `experiment dataframe`
  sql_df = get_sql_df(get_dataset_code(condition))
  ## from `experiment dataframe`, pull data pertaining to subjects responses
  subj_df = make_subj_df(sql_df['datastring'].iloc[0])
  ## reindex dataframe to make easy to read
  group_df = make_group_df(sql_df) # this just looks at the datastring
  # breakup dataframe by story and question nodes
  # type is type of node, so there are story nodes and question nodes
  # group_sdf is story state so this is what we 
  # want to use to know about 
  _,_,(_,group_qdf),(_,group_sdf) = group_df.groupby('type')
  # breakup question dataframe by filler and transition nodes
  (_,group_fqdf),(_,group_tqdf) = group_qdf.groupby('qtype')
  return group_df,group_tqdf,group_sdf,group_qdf,group_fqdf

def get_states_df(condition):
  group_df,group_tqdf,group_sdf,group_qdf,group_fqdf = load_dfs(condition)
  return group_sdf


def load_final_df(condition,threshold=THRESHOLD):
  group_df,group_tqdf,group_sdf,group_qdf,group_fqdf = load_dfs(condition)

  ## threshold based on filler questions
  # filler question are an attention check
  plt_group_tqdf = get_thresholded_tqdf(group_tqdf,group_fqdf,threshold=threshold)
  ## remove begin node probe (only happens during training)
  plt_group_tqdf = select_transition_probes(plt_group_tqdf)
  return plt_group_tqdf

