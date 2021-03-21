import numpy as np

from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer,text_to_word_sequence


class Agent():
    
    def __init__(self,vae,tokenizer,n_actions,verbose=True):       
        self.n_actions = n_actions
        self.vae = vae
        self.tokenizer = tokenizer
        
        self.state = np.array([0.,0.])
        self.Q = {self.state.tobytes(): np.ones(self.n_actions)} 
        
        self.total_trials = 0
        self.total_successes = 0
        
        self.verbose = verbose
             
    def set_learning_options(self,exploration=0.2,learningrate=0.1,discount=0.9,max_step=100):
        self.expl = exploration
        self.lr = learningrate
        self.discount = discount 
        self.max_step = max_step
        
    def set_padding_options(self, padding='post', maxlen=200, truncating='post'):
        self.padding = padding
        self.maxlen = maxlen
        self.truncating = truncating
        
        
    def _select_action(self):
        if (np.random.random() < self.expl):
            return np.random.randint(0,self.n_actions)        
        else:
            return np.argmax(self.Q[self.state.tobytes()])
        
    def step(self):
        self.steps = self.steps+1
        action = self._select_action()
        
        response,reward,termination,s = self.env.step(action)
        self.rewards = self.rewards + reward
        
        self._analyze_response(action,response,reward)
        self.terminated = termination
        if(self.verbose): print(s)
        
        return    
        
    
    def reset(self,env):
        self.env = env  
        self.terminated = False                     
        
        self.steps = 0
        self.rewards = 0
        
                
    def run_episode(self):
        _,_,self.terminated,s = self.env.reset()
        if(self.verbose): print(s)
        
        while not(self.terminated) and self.steps < self.max_step:
            self.step()
            #if(self.steps>500): self.terminated=True
            
        self.total_trials += 1
        if(self.terminated):
            self.total_successes += 1      
    
            
    def _analyze_response(self,action,response,reward):
        tokens = self.tokenizer.texts_to_sequences([response])
        tokens = keras.preprocessing.sequence.pad_sequences(tokens, padding=self.padding, maxlen=self.maxlen, truncating=self.truncating)
        
        newstate = self.vae.encoder(tokens)[0].numpy()[0]
        print(newstate)
        self._update_Q(self.state,newstate,action,reward)
        self.state = newstate
            
    def _update_Q(self,oldstate,newstate,action,reward):
        
        if(newstate.tobytes() in self.Q.keys()):
            best_action_newstate = np.argmax(self.Q[newstate.tobytes()])
        else:
            self.Q[newstate.tobytes()] = np.ones(self.n_actions)
            best_action_newstate = np.argmax(self.Q[newstate.tobytes()])
        
        self.Q[oldstate.tobytes()][action] = self.Q[oldstate.tobytes()][action] + self.lr * (reward + self.discount*self.Q[newstate.tobytes()][best_action_newstate] - self.Q[oldstate.tobytes()][action])
        
        
        
        
        
