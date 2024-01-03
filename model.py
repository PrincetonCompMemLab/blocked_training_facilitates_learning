import numpy as np

NSTATES = 9
MAX_SCH = 205

class SchemaTabularBayes():
    """ CRP prior
    tabluar predictive distirbution
    """
    def __init__(self,concentration,stickiness,sparsity,
        schidx=None):
        self.Tmat = np.zeros([NSTATES,NSTATES])
        self.alfa = concentration
        self.beta = stickiness
        self.lmbda = sparsity
        self.ntimes_sampled = 0
        self.schidx = schidx

    def get_prior(self,ztm1):
        """ 
        ztm1 : z of tstep t minus 1
        """
        if self.ntimes_sampled == 0:
            return self.alfa
        ztm1_flag = ztm1 == self.schidx
        crp = self.ntimes_sampled + self.beta* ztm1_flag
        return crp

    def get_like(self,xtm1,xt):
        PARAM_S = 2
        num = self.lmbda + self.Tmat[xtm1,xt] # sparsity plus number of transitions from the previous state to the current under this schema 
        den = (PARAM_S*self.lmbda) + self.Tmat[xtm1,:].sum() 
        like = num/den
        return like

    def update(self,xtm1,xt):
        # update schema transiton
        self.Tmat[xtm1,xt]+=1
        return None

    def predict(self,xtm1):
        """ returns un-normalized count """
        xthat = np.array([
            self.get_like(xtm1,x) for x in range(NSTATES)
            ])
        return xthat


class SEM():

    def __init__(self,schargs,skipt1,ppd_allsch):
        self.SchClass = SchemaTabularBayes
        self.schargs = schargs
        self.skipt1 = skipt1 
        self.ppd_allsch = ppd_allsch 
        self.init_schlib()

    def init_schlib(self):
        """ 
        initialize with two schemas
        one active one inactive
        """
        sch0 = self.SchClass(**self.schargs,schidx=0)
        sch1 = self.SchClass(**self.schargs,schidx=1)
        self.schlib = [sch0,sch1]
        return None

    def decay_allsch(self):
        for sch in self.schlib:
            sch.decay()
        return None


    def calc_posteriors(self,xtm1,xt,ztm,ztrm,active_only=False):
        """ loop over schema library
        """
        if active_only: # prediction
            priors = [sch.get_prior(ztm) for sch in self.schlib if sch.ntimes_sampled>0]
            likes = [sch.get_like(xtm1,xt) for sch in self.schlib if sch.ntimes_sampled>0]
        else: # sch inference
            priors = [sch.get_prior(ztm) for sch in self.schlib]
            likes = [sch.get_like(xtm1,xt) for sch in self.schlib]
            # record
            self.data['prior'][self.tridx,self.tstep,:len(priors)] = priors
            self.data['like'][self.tridx,self.tstep,:len(likes)] = likes
        posteriors = [p*l for p,l in zip(priors,likes)]
        return posteriors

    def select_sch(self,xtm1,xt,ztm,ztrm):
        """ xt and xtm1 are ints
        """
        posteriors = self.calc_posteriors(xtm1,xt,ztm,ztrm)
        # for each trial and each time step we want to get the posterior probability of each of the schemas 
        self.data['post'][self.tridx,self.tstep,:len(posteriors)] = posteriors
        active_k = np.argmax(posteriors)
        if active_k == len(self.schlib)-1:
            self.schlib.append(self.SchClass(**self.schargs,schidx=len(self.schlib)))
        return active_k

    def predict(self,xtm1,ztm,ztrm):
        """ 
        """
        pr_xt_z = np.array([
            self.calc_posteriors(xtm1,x,ztm,ztrm,active_only=True
                ) for x in range(NSTATES)
            ]) # probability of each next state under each schema
        if np.shape(pr_xt_z)[1] == 0:
            pr_xtp1 = np.array([np.nan for i in range(9)])
        else:
            pr_xtp1 = np.sum(pr_xt_z,axis=1)
        pr_xtp1 = pr_xtp1 / pr_xtp1.sum()
        return pr_xtp1

    def run_exp(self,exp, transition_matrix_analysis = False):
        """ exp is L of trialL
        trialL is L of obs (ints) 
        """
        ## recording
        Mt0ij = np.zeros([len(exp),NSTATES,NSTATES])
        self.data = data = {
            'zt':-np.ones([len(exp),len(exp[0])]),
            'xth':-np.ones([len(exp),len(exp[0]),NSTATES]),
            'prior':-np.ones([len(exp),len(exp[0]),MAX_SCH]),
            'like':-np.ones([len(exp),len(exp[0]),MAX_SCH]),
            'post':-np.ones([len(exp),len(exp[0]),MAX_SCH]),
        }
        scht = schtm = schtrm = self.schlib[0] # sch0 is active to start
        for tridx,trialL in enumerate(exp):
            self.tridx = tridx
            for tstep,(xtm,xt) in enumerate(zip(trialL[:-1],trialL[1:])):
                if (tstep==1) and (self.skipt1): 
                    continue
                if len(self.schlib)>=MAX_SCH: return data
                self.tstep = tstep
                if self.ppd_allsch:
                    xth = self.predict(xtm,schtm.schidx,schtrm.schidx)
                else:
                    xth = scht.predict(xtm)
                zt = self.select_sch(xtm,xt,schtm.schidx,schtrm.schidx)
                scht = self.schlib[zt] # scht is the current schema at time t
                # update transition matrix
                scht.update(xtm,xt) # this updates the transition matrix
                # of the newly selected schema
                scht.ntimes_sampled += 1
                # update schema history
                schtm = scht
                data['xth'][tridx][tstep] = xth
                data['zt'][tridx][tstep] = zt       
            # final schema of trial
            schtrm = scht 
        if transition_matrix_analysis:
            transition_matrices = [sch.Tmat for sch in self.schlib if sch.ntimes_sampled>0]
            nschemas = len(transition_matrices)
            return data, transition_matrices, nschemas
        return data


    def run_exp_sim4_blocked(self,exp,skipt1trial=40):
        """ exp is L of trialL
        trialL is L of obs (ints) 
        """
        ## recording
        Mt0ij = np.zeros([len(exp),NSTATES,NSTATES])
        self.data = data = {
            'zt':-np.ones([len(exp),len(exp[0])]),
            'xth':-np.ones([len(exp),len(exp[0]),NSTATES]),
            'prior':-np.ones([len(exp),len(exp[0]),MAX_SCH]),
            'like':-np.ones([len(exp),len(exp[0]),MAX_SCH]),
            'post':-np.ones([len(exp),len(exp[0]),MAX_SCH]),
        }
        scht = schtm = schtrm = self.schlib[0] # sch0 is active to start
        for tridx,trialL in enumerate(exp):
            self.tridx = tridx
            if tridx >= skipt1trial:
                self.skipt1 = True
            for tstep,(xtm,xt) in enumerate(zip(trialL[:-1],trialL[1:])):
                if (tstep==1) and (self.skipt1):    
                    continue
                if len(self.schlib)>=MAX_SCH: return data
                self.tstep = tstep
                if self.ppd_allsch:
                    # this first predict is called over
                    xth = self.predict(xtm,schtm.schidx,schtrm.schidx)
                else:
                    xth = scht.predict(xtm)
                zt = self.select_sch(xtm,xt,schtm.schidx,schtrm.schidx)
                scht = self.schlib[zt] # scht is the current schema at time t
                # update transition matrix
                scht.update(xtm,xt) # this updates the transition matrix
                # of the newly selected schema
                scht.ntimes_sampled += 1
                # update schema history
                schtm = scht
                data['xth'][tridx][tstep] = xth
                data['zt'][tridx][tstep] = zt       
            # final schema of trial
            schtrm = scht 
        return data



class Task():
    """ 
    """

    def __init__(self):
        A1,A2,B1,B2 = self._init_paths_csw()
        self.paths = [[A1,A2],[B1,B2]]
        self.tsteps = len(self.paths[0][0])
        self.exp_int = None
        return None


    def _init_paths_csw(self):
        """ 
        begin -> locA -> node11, node 21, node 31, end
        begin -> locA -> node12, node 22, node 32, end
        begin -> locB -> node11, node 22, node 31, end
        begin -> locB -> node12, node 21, node 32, end
        """
        begin = 0 # E0
        locA,locB = 1,2 # E1
        node11,node12 = 3,4 # E2 
        node21,node22 = 5,6 # E3
        node31,node32 = 7,8 # E4
        end = 9
        A1 = np.array([begin,locA,
            node11,node21,node31
            ])
        A2 = np.array([begin,locA,
            node12,node22,node32
            ])
        B1 = np.array([begin,locB,
            node11,node22,node31
            ])
        B2 = np.array([begin,locB,
            node12,node21,node32
            ])
        return A1,A2,B1,B2

    def _init_paths_toy(self):
        """ 
        begin -> locA -> node11, node 21, node 31, end
        begin -> locA -> node12, node 22, node 32, end
        begin -> locB -> node11, node 22, node 31, end
        begin -> locB -> node12, node 21, node 32, end
        """
        locA,locB = 0,1
        node11,node12 = 2,3
        node21,node22 = 4,5
        A1 = np.array([locA,
            node11,node21
            ])
        A2 = np.array([locA,
            node12,node22
            ])
        B1 = np.array([locB,
            node11,node22
            ])
        B2 = np.array([locB,
            node12,node21
            ])
        return A1,A2,B1,B2


    def get_curriculum(self,condition,n_train,n_test):
        """ 
        order of events
        NB blocked: ntrain needs to be divisible by 4
        """
        # ross: 0/1 here is a way that schema A and B is coded
        curriculum = []   
        
        if condition == 'blocked':
            assert n_train%4==0
            # if blocked, then get A then B then A then B
            curriculum =  \
                [0] * (n_train // 4) + \
                [1] * (n_train // 4) + \
                [0] * (n_train // 4) + \
                [1] * (n_train // 4 )
        elif condition == 'early':
            # first do blocked then interleave
            curriculum =  \
                [0] * (n_train // 4) + \
                [1] * (n_train // 4) + \
                [0, 1] * (n_train // 4)
        elif condition == 'middle':
            # interleave, blocked, interleaved
            curriculum =  \
                [0, 1] * (n_train // 8) + \
                [0] * (n_train // 4) + \
                [1] * (n_train // 4) + \
                [0, 1] * (n_train // 8)
        elif condition == 'late':
            # interleaved, then blocked
            curriculum =  \
                [0, 1] * (n_train // 4) + \
                [0] * (n_train // 4) + \
                [1] * (n_train // 4)

        elif condition == 'interleaved':
            # all interleaved
            curriculum = [0, 1] * (n_train // 2)
        elif condition == 'single': ## DEBUG
            curriculum =  \
                [0] * (n_train) 
        else:
            print('condition not properly specified')
            assert False
        # then get the testing which is consistent across
        curriculum += [int(np.random.rand() < 0.5) for _ in range(n_test)]
        return np.array(curriculum)


    def generate_experiment(self,condition,n_train,n_test):
        """ 
        exp: arr [ntrials,tsteps]
        curr: arr [ntrials]
        """
        # get curriculum
        n_trials = n_train+n_test
        curr = self.get_curriculum(condition,n_train,n_test)
        # generate trials
        exp = -np.ones([n_trials,self.tsteps],dtype=int)
        for trial_idx in range(n_train+n_test):
            # select A1,A2,B1,B2
            # for each trial, you select whether it is from A or B
            event_type = curr[trial_idx]
            # then take 50/50 choice between the two paths
            path_type = np.random.randint(2)
            # then you get the path
            path_int = self.paths[event_type][path_type]
            # embed
            # and then you turn that row into the path
            exp[trial_idx] = path_int
        # exp has the path in the columns and each row is a specific trial
        return exp,curr