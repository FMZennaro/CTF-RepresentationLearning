import numpy as np
import sys

from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer,text_to_word_sequence

"""
agent.py is based on FMZennaro's agent on https://github.com/FMZennaro/CTF-RL/blob/master/Simulation1/agent.py
"""

class Agent():
    def __init__(self,tokenizer,actions, verbose=True):
        self.actions = actions
        self.num_actions = len(actions)
        self.tokenizer = tokenizer

        self.Q = {(): np.ones(self.num_actions)}

        self.verbose = verbose
        self.set_learning_options()
        self.set_padding_options()
        
        self.used_actions = []
        self.steps = 0
        self.rewards = 0
        self.total_trials = 0
        self.total_successes = 0

    def set_learning_options(self,exploration=0.2,learningrate=0.1,discount=0.9, max_step = 100):
        self.expl = exploration
        self.lr = learningrate
        self.discount = discount
        self.max_step = max_step
        
    def set_padding_options(self, padding='post', maxlen=200, truncating='post'):
        self.padding = padding
        self.maxlen = maxlen
        self.truncating = truncating

    def _select_action(self, learning = True):
        if (np.random.random() < self.expl and learning):
            return np.random.randint(0,self.num_actions)
        else:
            return np.argmax(self.Q[self.state])


    def step(self, deterministic = False):
        self.steps = self.steps + 1

        action = self._select_action(learning = not deterministic)
        self.used_actions.append(action)

        state_resp, reward, termination, debug_msg = self.env.step(action)

        self.rewards = self.rewards + reward

        self._analyze_response(action, state_resp, reward, learning = not deterministic)
        self.terminated = termination
        if(self.verbose): print(debug_msg)

        return


    def _update_state(self, action_nr, encoded_page):
        """
        action_nr is concatenated to the page representation and hashed into a integer
        """
        
        action_response = np.concatenate((np.array([action_nr]),np.expand_dims(encoded_page,axis=0)))
        x = hash(action_response.tobytes())
        self.Q[x] = self.Q.get(x, np.ones(self.num_actions))

        self.oldstate = self.state
        self.state = x
        

    def _analyze_response(self, action, response, reward, learning = True):
        tokens = self.tokenizer.texts_to_sequences([response])
        tokens = keras.preprocessing.sequence.pad_sequences(tokens, padding=self.padding, maxlen=self.maxlen, truncating=self.truncating)
        
        encoded_page = np.sum(np.log(tokens,where=(tokens!=0)))
        
        self._update_state(action,encoded_page)
        if(learning): self._update_Q(action,reward)


    def _update_Q(self, action, reward):
        best_action_newstate = np.argmax(self.Q[self.state])
        self.Q[self.oldstate][action] = self.Q[self.oldstate][action] + self.lr * (reward + self.discount*self.Q[self.state][best_action_newstate] - self.Q[self.oldstate][action])

    def reset(self,env):
        self.env = env
        self.terminated = False
        self.state = () #empty tuple
        self.oldstate = None
        self.used_actions = []

        self.steps = 0
        self.rewards = 0


    def run_episode(self, deterministic = False):
        _,_,self.terminated,s = self.env.reset()
        if(self.verbose): print(s)

        #Limiting the maximimun number of steps we allow the attacker to make to avoid overly long runtimes and extreme action spaces
        while (not(self.terminated)) and self.steps < self.max_step:
            self.step(deterministic = deterministic)

        self.total_trials += 1
        if(self.terminated):
            self.total_successes += 1
        return self.terminated

