
import numpy as np

class mockSQLienv(object):
    def __init__(self,htmlpages,actions=10,S0S1=None,S1S2=None,verbose=False):
        
        self.nactions = actions
        self.htmlpages = htmlpages
        self.verbose = verbose
        
        if S0S1==None and S1S2==None:
            self.S0S1, self.S1S2 = np.random.choice(np.arange(actions),2,replace=False)
        else:
            self.S0S1 = S0S1
            self.S1S2 = S1S2
        
        self.state = 0
        self.termination = False
        
    def reset(self):
        self.state = 0
        self.termination = False
        
        return self.htmlpages[2],0,self.termination,'env reset'
    
    def step(self,action):
        
        if(action==self.S0S1):
            self.state = 1
            return self.htmlpages[0],-1,self.termination,'correct escape'
        
        elif(self.state==1 and action==self.S1S2):
            self.state = 2
            self.termination = True
            return self.htmlpages[1],100,self.termination,'flag'
        
        else:
            return self.htmlpages[3],-1,self.termination,'wrong request'
        
    def _get_solution(self):
        return self.S0S1, self.S1S2