import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random


class QNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers=(128, 128)):
        super(QNetwork, self).__init__()
        
        # Создаём слои сети
        layer_sizes = [input_size] + list(hidden_layers) + [output_size]
        self.layers = nn.ModuleList()
        
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
    
    def forward(self, state):
        x = state
        for i, layer in enumerate(self.layers[:-1]):
            x = torch.relu(layer(x))
        x = self.layers[-1](x)
        return x


class ExperienceReplayBuffer:

    def __init__(self, max_size, use_prioritized=False, alpha=0.6):
        self.max_size = max_size
        self.use_prioritized = use_prioritized
        self.alpha = alpha
        
        if use_prioritized:
            self.memory = []
            self.priority_values = []
            self.position = 0
        else:
            self.memory = deque(maxlen=max_size)
    
    def add_experience(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        
        if self.use_prioritized:
            max_priority = max(self.priority_values) if self.priority_values else 1.0
            
            if len(self.memory) < self.max_size:
                self.memory.append(experience)
                self.priority_values.append(max_priority)
            else:
                self.memory[self.position] = experience
                self.priority_values[self.position] = max_priority
                self.position = (self.position + 1) % self.max_size
        else:
            self.memory.append(experience)
    
    def sample_batch(self, batch_size, beta=0.4):
        if self.use_prioritized:
            priorities = np.array(self.priority_values)
            probabilities = priorities ** self.alpha
            probabilities /= probabilities.sum()
            
            sampled_indices = np.random.choice(
                len(self.memory), 
                batch_size, 
                p=probabilities, 
                replace=False
            )
            
            experiences = [self.memory[idx] for idx in sampled_indices]
            
            sampling_weights = (len(self.memory) * probabilities[sampled_indices]) ** (-beta)
            sampling_weights /= sampling_weights.max()
            
            return experiences, sampled_indices, sampling_weights
        else:
            experiences = random.sample(self.memory, batch_size)
            return experiences, None, None
    
    def update_priorities(self, indices, new_priorities):
        if self.use_prioritized and indices is not None:
            for idx, priority in zip(indices, new_priorities):
                self.priority_values[idx] = priority
    
    def size(self):
        return len(self.memory)


class DeepQAgent:
    def __init__(self, state_size, action_size,
                 hidden_layers=(64, 64),
                 learning_rate=0.001,
                 discount_factor=0.95,
                 exploration_rate=1.0,
                 exploration_decay=0.995,
                 exploration_min=0.05,
                 memory_size=5000,
                 training_batch_size=32,
                 target_network_update_frequency=100,
                 enable_double_dqn=True,
                 enable_prioritized_replay=False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.epsilon_decay = exploration_decay
        self.epsilon_min = exploration_min
        self.batch_size = training_batch_size
        self.update_target_frequency = target_network_update_frequency
        self.use_double_dqn = enable_double_dqn
        
        self.policy_network = QNetwork(state_size, action_size, hidden_layers).to(self.device)
        self.target_network = QNetwork(state_size, action_size, hidden_layers).to(self.device)
        
        self.target_network.load_state_dict(self.policy_network.state_dict())
        self.target_network.eval()
        
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)
        
        self.replay_buffer = ExperienceReplayBuffer(
            memory_size, 
            use_prioritized=enable_prioritized_replay
        )
        
        self.training_steps = 0
    
    def select_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_size)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            q_values = self.policy_network(state_tensor)
        
        return q_values.argmax().item()
    
    def remember(self, state, action, reward, next_state, done):
        self.replay_buffer.add_experience(state, action, reward, next_state, done)
    
    def learn_from_experience(self):
        if self.replay_buffer.size() < self.batch_size:
            return
        
        if self.replay_buffer.use_prioritized:
            experiences, indices, weights = self.replay_buffer.sample_batch(self.batch_size)
            importance_weights = torch.FloatTensor(weights).to(self.device)
        else:
            experiences, _, _ = self.replay_buffer.sample_batch(self.batch_size)
            importance_weights = torch.ones(self.batch_size).to(self.device)
        
        states = torch.FloatTensor([exp[0] for exp in experiences]).to(self.device)
        actions = torch.LongTensor([exp[1] for exp in experiences]).to(self.device)
        rewards = torch.FloatTensor([exp[2] for exp in experiences]).to(self.device)
        next_states = torch.FloatTensor([exp[3] for exp in experiences]).to(self.device)
        dones = torch.FloatTensor([exp[4] for exp in experiences]).to(self.device)
        
        current_q_values = self.policy_network(states).gather(1, actions.unsqueeze(1)).squeeze()
        
        with torch.no_grad():
            if self.use_double_dqn:
                best_actions = self.policy_network(next_states).argmax(1)
                next_q_values = self.target_network(next_states).gather(
                    1, best_actions.unsqueeze(1)
                ).squeeze()
            else:
                next_q_values = self.target_network(next_states).max(1)[0]
            
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)
        
        td_errors = (current_q_values - target_q_values).abs().detach().cpu().numpy()
        
        loss = (importance_weights * (current_q_values - target_q_values) ** 2).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        if self.replay_buffer.use_prioritized and indices is not None:
            self.replay_buffer.update_priorities(indices, td_errors + 1e-6)
        
        self._decay_exploration()
        
        self.training_steps += 1
        if self.training_steps % self.update_target_frequency == 0:
            self._update_target_network()
    
    def _decay_exploration(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def _update_target_network(self):
        self.target_network.load_state_dict(self.policy_network.state_dict())
    
    def save_model(self, filepath):
        torch.save({
            'policy_network': self.policy_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_steps': self.training_steps
        }, filepath)
    
    def load_model(self, filepath):
        checkpoint = torch.load(filepath)
        self.policy_network.load_state_dict(checkpoint['policy_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.training_steps = checkpoint['training_steps']
