import numpy as np

from tensorflow.keras.preprocessing.text import Tokenizer,text_to_word_sequence

class SQLiTokenizer():
    def __init__(self,
                 lower=False,
                 filters='"#$%&()*+,.;?@[\\]^_`{|}~\t\n',
                 tokens_per_file=200, 
                 padding='NOP'):
        
        self.lower = lower
        self.filters = filters
        self.tokens_per_file = tokens_per_file
        self.padding = padding
        
        self.tknizer = Tokenizer(lower=lower, filters=filters)
        
    def fit(self,inputs):
        self.tknizer.fit_on_texts(inputs)
        
        self.padding_integer = len(self.tknizer.word_index)+1
        self.tknizer.word_index[self.padding]= self.padding_integer
        
    def transform(self,inputs):
        tokens_list = self.tknizer.texts_to_sequences(inputs)
        
        for i in range(len(tokens_list)):
            l = len(tokens_list[i])
            if l < self.tokens_per_file:
                tokens_list[i] = tokens_list[i] + [self.padding_integer]*(self.tokens_per_file-l)
            elif l > self.tokens_per_file:
                tokens_list[i] = tokens_list[i][0:self.tokens_per_file]
                
        flat_tokens_list = [item for sublist in tokens_list for item in sublist]
        return np.array(flat_tokens_list).reshape(len(tokens_list),self.tokens_per_file)