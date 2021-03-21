
# CTF-RepresentationLearning
Learning Representation for CTF


## Simulation 1
This simulation runs a basic tabular Q-learning agent with a VAE module to process HTML responses. The server is given by a simple finite state automaton with two states.

### Setup

- **SQLiServer_1.py**: basic mock server defined as a finite state automaton with two states
- **SQLiServer.ipynb**: validates the server (input: None; output: None)

- **Tokenizer_1.ipynb**: trains and save a Tokenizer (input: None; output: Tokenizer)

- **VAEAgent_1.py**: basic agent integrating a VAE
- **VAETraining_1.ipynb:** trains and save a VAE (input: Tokenizer; output: VAE)
- **VAEAgent_1.ipynb:** runs an agent on a mock simulation (input: Tokenizer, VAE; output: Agent)
- **VAEStatTraining_1.ipynb:** runs a set of agents (input: Tokenizer, VAE; output: Stats) \[TO BE DONE\]
- **VAEStatAnalysis_1.ipynb:** analyzes the performance of a set of agent (input: Stats; output: graphs) \[TO BE DONE\]


## Simulation 2
This simulation runs the tabular Q-learning agent used for SQL injection in [1], with an additional VAE module to process HTML responses. The server is the same server used in [1].

### Setup

- **SQLiServer_2.py**: SQL server
- **SQLiServer.ipynb**: validates the server (input: None; output: None)

- **Tokenizer_1.ipynb**: trains and save a Tokenizer (input: None; output: Tokenizer)

- **VAEAgent_2.py**: agent integrating a VAE
- **VAETraining_1.ipynb:** trains and save a VAE (input: Tokenizer; output: VAE)
- **VAEAgent_2.ipynb:** runs an agent on a mock simulation (input: Tokenizer, VAE; output: Agent)
- **VAEStatTraining_2.ipynb:** runs a set of agents (input: Tokenizer, VAE; output: Stats) \[TO BE DONE\]
- **VAEStatAnalysis_2.ipynb:** analyzes the performance of a set of agent (input: Stats; output: graphs) \[TO BE DONE\]


## Simulation 3
This simulation runs the tabular Q-learning agent used for SQL injection in [1], with a simple static/random projection to encode HTML responses. The server is the same server used in [1].

### Setup

- **SQLiServer_2.py**: SQL server
- **SQLiServer.ipynb**: validates the server (input: None; output: None)

- **Tokenizer_1.ipynb**: trains and save a Tokenizer (input: None; output: Tokenizer)

- **ProjectionAgent_1.py**: agent with simple random projection
- **ProjectionAgent_1.ipynb:** runs an agent on a mock simulation (input: Tokenizer, VAE; output: Agent)
- **ProjectionStatTraining_1.ipynb:** runs a set of agents (input: Tokenizer, VAE; output: Stats) \[TO BE DONE\]
- **ProjectionStatAnalysis_1.ipynb:** analyzes the performance of a set of agent (input: Stats; output: graphs) \[TO BE DONE\]


## Simulation 4
This simulation runs a basic tabular Q-learning agent with a Seq2Seq module to process HTML responses. The server is given by a simple finite state automaton with two states.

### Setup

- **SQLiServer_1.py**: basic mock server defined as a finite state automaton with two states
- **SQLiServer.ipynb**: validates the server (input: None; output: None)

- **Tokenizer_1.ipynb**: trains and save a Tokenizer (input: None; output: Tokenizer)

- **Seq2SeqAgent_1.py**: basic agent integrating a Seq2Seq \[TO BE DONE\]
- **Seq2SeqTraining_1.ipynb:** trains and save a Seq2Seq (input: Tokenizer; output: Seq2Seq)
- **Seq2SeqStatTraining_1.ipynb:** runs a set of agents (input: Tokenizer, Seq2Seq; output: Stats) \[TO BE DONE\]
- **Seq2SeqStatAnalysis_1.ipynb:** analyzes the performance of a set of agent (input: Stats; output: graphs) \[TO BE DONE\]






### References

\[1\] Erdodi, L., Sommervoll, A.A. and Zennaro, F.M., 2020. Simulating SQL Injection Vulnerability Exploitation Using Q-Learning Reinforcement Learning Agents. arXiv preprint.

