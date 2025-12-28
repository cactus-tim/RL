import numpy as np


class QLearningAgent:
    def __init__(self, initial_state, observation_space, action_space, 
                 learning_rate=0.1, discount_factor=0.95, 
                 exploration_rate=0.5, exploration_decay=0.9995, 
                 exploration_min=0.01):
        self.current_state = initial_state
        self.action_space = action_space
        self.num_actions = action_space.n
        
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.epsilon_decay = exploration_decay
        self.epsilon_min = exploration_min
        
        self.q_values = {}
        
        self.last_action = None
        
    def _initialize_state(self, state):
        if state not in self.q_values:
            self.q_values[state] = np.zeros(self.num_actions)
    
    def select_action(self):
        self._initialize_state(self.current_state)
        
        if np.random.random() < self.epsilon:
            chosen_action = self.action_space.sample()
        else:
            chosen_action = np.argmax(self.q_values[self.current_state])
        
        self.last_action = chosen_action
        
        self._decay_epsilon()
        
        return chosen_action
    
    def _decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def update(self, next_state, reward, is_terminal):
        self._initialize_state(next_state)
        
        if is_terminal:
            target_q = reward
        else:
            max_future_q = np.max(self.q_values[next_state])
            target_q = reward + self.gamma * max_future_q
        
        current_q = self.q_values[self.current_state][self.last_action]
        
        td_error = target_q - current_q
        self.q_values[self.current_state][self.last_action] += self.lr * td_error
        
        self.current_state = next_state
    
    def get_q_table_size(self):
        return len(self.q_values)
    
    def get_exploration_rate(self):
        return self.epsilon
