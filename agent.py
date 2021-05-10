import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, nA=6, alpha = 0.25, gamma = 0.77, epsilon=0.0001):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        

    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        policy_s = self.get_probs_epsilon_greedy(self.Q[state])
        action = np.random.choice(np.arange(self.nA), p=policy_s)
        
        return action

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """        
        # Q LEARNING ALGORITHM (SARSAMAX)
        # Update Q, TD estimate of Q
        self.Q[state][action] = self.update_Q_learning(self.Q[state][action], self.Q[next_state], reward, self.alpha, self.gamma)
        
        if done:
            self.Q[state][action] = self.update_Q_learning(self.Q[state][action], 0, reward, self.alpha, self.gamma)
       
    def get_probs_epsilon_greedy(self, Q_s):
        """ obtains the action probabilities corresponding to epsilon-greedy policy """

        policy_s = np.ones(self.nA) * self.epsilon / self.nA
        # default value of epsilon-greedy for actions that dont maximize Q
        best_a = np.argmax(Q_s) # Action that maximizes Q
        policy_s[best_a] = 1 - self.epsilon + (self.epsilon / self.nA)
        return policy_s

    def update_Q_learning(self, Q, Q_s, reward, alpha, gamma):
        """ updates the action-value function estimate using the most recent time step """
        # estimate action 'a' that maximizes Q[next_state], get value Q[next_state][best_action]
        Q_a = np.max(Q_s)
        return Q + (alpha * (reward + (gamma * Q_a) - Q))
