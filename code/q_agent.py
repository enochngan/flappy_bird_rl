"""
Q-Learning Agent for Flappy Bird
Learns the optimal policy through value iteration
"""
import numpy as np
import pickle
import os
from pathlib import Path


class QLearningAgent:
    """Q-Learning agent with discretized state space."""
    
    def __init__(self, 
                 num_bins=12,
                 learning_rate=0.12,
                 discount_factor=0.96,
                 epsilon=1.0,
                 epsilon_decay=0.993,
                 epsilon_min=0.02):
        """
        Initialize Q-Learning agent.
        
        Args:
            num_bins: Number of bins for discretizing each state dimension
            learning_rate: Alpha - how much to update Q-values
            discount_factor: Gamma - importance of future rewards
            epsilon: Initial exploration rate
            epsilon_decay: Decay rate per episode
            epsilon_min: Minimum exploration rate
        """
        self.num_bins = num_bins
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Q-table: dict mapping (discretized_state, action) -> Q-value
        self.q_table = {}
        
        # Track statistics
        self.episode_scores = []
        self.episode_rewards = []
    
    def _discretize_state(self, state):
        """
        Convert continuous state [0, 1] to discrete bins.
        
        Args:
            state: np.array of shape (4,) with values in [0, 1]
        
        Returns:
            Tuple of 4 integers representing discretized state
        """
        # Convert each dimension to a bin index [0, num_bins-1]
        discretized = tuple(
            min(int(s * self.num_bins), self.num_bins - 1)
            for s in state
        )
        return discretized
    
    def _get_q_value(self, state, action):
        """Get Q-value for state-action pair (default to 0)."""
        key = (state, action)
        return self.q_table.get(key, 0.0)
    
    def _set_q_value(self, state, action, value):
        """Set Q-value for state-action pair."""
        key = (state, action)
        self.q_table[key] = value
    
    def choose_action(self, state):
        """
        Choose action using epsilon-greedy policy.
        
        Args:
            state: np.array of continuous state values [0, 1]
        
        Returns:
            Action: 0 (idle) or 1 (flap)
        """
        # Discretize state
        discrete_state = self._discretize_state(state)
        
        # Epsilon-greedy
        if np.random.random() < self.epsilon:
            # Explore: random action
            return np.random.randint(0, 2)
        else:
            # Exploit: best known action
            q0 = self._get_q_value(discrete_state, 0)
            q1 = self._get_q_value(discrete_state, 1)
            return 0 if q0 >= q1 else 1
    
    def update(self, state, action, reward, next_state, done):
        """
        Update Q-value using Q-learning update rule.
        
        Q(s,a) = Q(s,a) + α * (r + γ * max(Q(s',a)) - Q(s,a))
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
        """
        discrete_state = self._discretize_state(state)
        discrete_next_state = self._discretize_state(next_state)
        
        # Current Q-value
        current_q = self._get_q_value(discrete_state, action)
        
        # Max Q-value for next state
        if done:
            max_next_q = 0.0
        else:
            max_next_q = max(
                self._get_q_value(discrete_next_state, 0),
                self._get_q_value(discrete_next_state, 1)
            )
        
        # Q-learning update
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )
        
        self._set_q_value(discrete_state, action, new_q)
    
    def decay_epsilon(self):
        """Reduce exploration rate at end of episode."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def record_episode(self, score, total_reward):
        """Record episode statistics."""
        self.episode_scores.append(score)
        self.episode_rewards.append(total_reward)
    
    def save(self, filepath):
        """Save Q-table to disk."""
        data = {
            'q_table': self.q_table,
            'num_bins': self.num_bins,
            'episode_scores': self.episode_scores,
            'episode_rewards': self.episode_rewards,
            'epsilon': self.epsilon,
            'epsilon_decay': self.epsilon_decay,
            'epsilon_min': self.epsilon_min,
            'learning_rate': self.learning_rate,
            'discount_factor': self.discount_factor,
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"✓ Agent saved to {filepath}")
    
    def load(self, filepath):
        """Load Q-table from disk."""
        if not os.path.exists(filepath):
            print(f"⚠ File {filepath} not found - starting fresh")
            return
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.q_table = data['q_table']
        self.num_bins = data['num_bins']
        self.episode_scores = data.get('episode_scores', [])
        self.episode_rewards = data.get('episode_rewards', [])
        self.epsilon = data.get('epsilon', self.epsilon)
        self.epsilon_decay = data.get('epsilon_decay', self.epsilon_decay)
        self.epsilon_min = data.get('epsilon_min', self.epsilon_min)
        self.learning_rate = data.get('learning_rate', self.learning_rate)
        self.discount_factor = data.get('discount_factor', self.discount_factor)
        print(f"✓ Agent loaded from {filepath}")
        print(f"  - Trained on {len(self.episode_scores)} episodes")
        if self.episode_scores:
            avg_score = np.mean(self.episode_scores[-100:])
            print(f"  - Avg score (last 100): {avg_score:.1f}")
    
    def get_stats(self):
        """Get agent training statistics."""
        if not self.episode_scores:
            return {}
        
        return {
            'total_episodes': len(self.episode_scores),
            'avg_score_all': np.mean(self.episode_scores),
            'avg_score_recent': np.mean(self.episode_scores[-100:]),
            'max_score': max(self.episode_scores),
            'q_table_size': len(self.q_table),
        }
    
    def set_exploit_mode(self):
        """Set epsilon to 0 for testing (no exploration)."""
        self.epsilon = 0.0
    
    def set_train_mode(self):
        """Reset epsilon for training."""
        self.epsilon = max(self.epsilon_min, self.epsilon * 10)  # Restore some exploration


if __name__ == "__main__":
    # Quick test
    agent = QLearningAgent(num_bins=10)
    
    # Simulate a few steps
    state = np.array([0.5, 0.0, 1.0, 0.5])
    action = agent.choose_action(state)
    print(f"Action chosen: {action}")
    
    agent.update(state, action, 0.1, state, False)
    print(f"Q-table size: {len(agent.q_table)}")
    
    # Save/load test
    agent.save("test_agent.pkl")
    agent2 = QLearningAgent()
    agent2.load("test_agent.pkl")
    print(f"Loaded Q-table size: {len(agent2.q_table)}")
    
    import os
    os.remove("test_agent.pkl")
    print("✓ Q-Learning agent working!")
