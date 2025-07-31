# reinforce_agent.py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Dict, List, Tuple
import time

class PolicyNetwork(nn.Module):
    """Enhanced Policy Network for REINFORCE"""

    def __init__(self, state_size, action_size, hidden_size=128):
        super(PolicyNetwork, self).__init__()

        # Network layers with proper initialization
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, action_size)
        self.dropout = nn.Dropout(0.2)

        # Initialize weights using Xavier/Glorot initialization
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return F.softmax(x, dim=-1)

class REINFORCEAgent:
    """Enhanced REINFORCE Agent for Hospital Navigation"""

    def __init__(self, state_size, action_size, lr=1e-3, device='cpu', hyperparams=None):
        self.state_size = state_size
        self.action_size = action_size
        self.device = device

        # Default hyperparameters (well-tuned for hospital navigation)
        self.default_hyperparams = {
            'learning_rate': 1e-3,      # Learning rate for policy updates
            'gamma': 0.99,              # Discount factor
            'hidden_size': 128,         # Hidden layer size
            'max_grad_norm': 1.0,       # Gradient clipping
            'baseline_type': 'mean',    # Baseline for variance reduction ('mean', 'none')
            'entropy_coef': 0.01,       # Entropy regularization coefficient
        }

        # Override with custom hyperparams if provided
        if hyperparams:
            self.hyperparams = {**self.default_hyperparams, **hyperparams}
        else:
            self.hyperparams = self.default_hyperparams

        # Extract hyperparameters
        self.learning_rate = self.hyperparams['learning_rate']
        self.gamma = self.hyperparams['gamma']
        self.max_grad_norm = self.hyperparams['max_grad_norm']
        self.baseline_type = self.hyperparams['baseline_type']
        self.entropy_coef = self.hyperparams['entropy_coef']

        # Policy network
        self.policy_net = PolicyNetwork(
            state_size, action_size,
            hidden_size=self.hyperparams['hidden_size']
        ).to(device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)

        # Episode storage
        self.reset_episode()

        # Training stats
        self.episode_rewards = []
        self.episode_lengths = []
        self.policy_losses = []
        self.entropy_losses = []
        self.eval_rewards = []
        self.eval_episodes = []
        self.training_time = 0

        # Running baseline for variance reduction
        self.reward_baseline = None

    def reset_episode(self):
        """Reset episode storage"""
        self.log_probs = []
        self.rewards = []
        self.entropies = []

    def act(self, state, training=True):
        """Choose action using current policy"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action_probs = self.policy_net(state_tensor)

        if training:
            # Sample action from distribution
            dist = Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()

            # Store for training
            self.log_probs.append(log_prob)
            self.entropies.append(entropy)

            return action.item()
        else:
            # Choose best action for evaluation
            return action_probs.argmax().item()

    def store_reward(self, reward):
        """Store reward for current step"""
        self.rewards.append(reward)

    def compute_returns(self):
        """Compute discounted returns with optional baseline"""
        returns = []
        R = 0

        # Compute returns backwards
        for reward in reversed(self.rewards):
            R = reward + self.gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)

        # Apply baseline for variance reduction
        if self.baseline_type == 'mean':
            if self.reward_baseline is None:
                self.reward_baseline = returns.mean().item()
            else:
                # Exponential moving average
                self.reward_baseline = 0.9 * self.reward_baseline + 0.1 * returns.mean().item()

            returns = returns - self.reward_baseline

        # Normalize returns
        if returns.std() > 1e-8:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        return returns

    def update(self):
        """Update policy using REINFORCE algorithm"""
        if len(self.log_probs) == 0:
            return 0.0

        # Compute returns
        returns = self.compute_returns()

        # Compute policy loss (negative because we want to maximize)
        policy_loss = []
        entropy_loss = []

        for log_prob, R, entropy in zip(self.log_probs, returns, self.entropies):
            policy_loss.append(-log_prob * R)
            entropy_loss.append(-entropy)  # Negative for maximization

        policy_loss = torch.stack(policy_loss).sum()
        entropy_loss = torch.stack(entropy_loss).sum()

        # Total loss with entropy regularization
        total_loss = policy_loss + self.entropy_coef * entropy_loss

        # Update policy
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.max_grad_norm)
        self.optimizer.step()

        # Store losses
        self.policy_losses.append(policy_loss.item())
        self.entropy_losses.append(entropy_loss.item())

        # Reset episode
        self.reset_episode()

        return policy_loss.item()

    def evaluate(self, env, n_episodes=10):
        """Evaluate policy performance"""
        total_rewards = []
        total_lengths = []

        for _ in range(n_episodes):
            state, _ = env.reset()
            episode_reward = 0
            episode_length = 0
            done = False

            while not done:
                action = self.act(state, training=False)
                state, reward, done, truncated, _ = env.step(action)
                episode_reward += reward
                episode_length += 1

                if truncated:
                    done = True

            total_rewards.append(episode_reward)
            total_lengths.append(episode_length)

        mean_reward = np.mean(total_rewards)
        std_reward = np.std(total_rewards)
        mean_length = np.mean(total_lengths)

        return mean_reward, std_reward, mean_length

    def save(self, filepath):
        """Save model and training state"""
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'hyperparams': self.hyperparams,
            'reward_baseline': self.reward_baseline,
            'episode_rewards': self.episode_rewards,
            'policy_losses': self.policy_losses,
        }, filepath)

    def load(self, filepath):
        """Load model and training state"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.reward_baseline = checkpoint.get('reward_baseline', None)
        self.episode_rewards = checkpoint.get('episode_rewards', [])
        self.policy_losses = checkpoint.get('policy_losses', [])

    def plot_training_results(self, save_path=None):
        """Plot comprehensive training results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('REINFORCE Training Results', fontsize=16, fontweight='bold')

        # Episode rewards
        if self.episode_rewards:
            axes[0, 0].plot(self.episode_rewards, alpha=0.7, color='blue')
            axes[0, 0].set_title('Episode Rewards During Training')
            axes[0, 0].set_xlabel('Episode')
            axes[0, 0].set_ylabel('Total Reward')
            axes[0, 0].grid(True, alpha=0.3)

            # Add moving average
            if len(self.episode_rewards) > 10:
                window = min(50, len(self.episode_rewards) // 10)
                moving_avg = np.convolve(self.episode_rewards,
                                       np.ones(window)/window, mode='valid')
                axes[0, 0].plot(range(window-1, len(self.episode_rewards)),
                              moving_avg, 'r-', linewidth=2, label=f'Moving Avg ({window})')
                axes[0, 0].legend()

        # Episode lengths
        if self.episode_lengths:
            axes[0, 1].plot(self.episode_lengths, alpha=0.7, color='orange')
            axes[0, 1].set_title('Episode Lengths During Training')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Steps')
            axes[0, 1].grid(True, alpha=0.3)

        # Policy losses
        if self.policy_losses:
            axes[1, 0].plot(self.policy_losses, color='red', alpha=0.8)
            axes[1, 0].set_title('Policy Loss')
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Policy Loss')
            axes[1, 0].grid(True, alpha=0.3)

        # Training statistics summary
        axes[1, 1].axis('off')
        if self.episode_rewards:
            stats_text = f"""
            Training Statistics:

            Total Episodes: {len(self.episode_rewards)}
            Mean Episode Reward: {np.mean(self.episode_rewards):.2f}
            Std Episode Reward: {np.std(self.episode_rewards):.2f}
            Max Episode Reward: {np.max(self.episode_rewards):.2f}
            Min Episode Reward: {np.min(self.episode_rewards):.2f}

            Final 10 Episodes Mean: {np.mean(self.episode_rewards[-10:]):.2f}

            Hyperparameters:
            Learning Rate: {self.hyperparams['learning_rate']}
            Gamma: {self.hyperparams['gamma']}
            Hidden Size: {self.hyperparams['hidden_size']}
            Baseline Type: {self.hyperparams['baseline_type']}
            Entropy Coef: {self.hyperparams['entropy_coef']}
            """
            axes[1, 1].text(0.1, 0.9, stats_text, transform=axes[1, 1].transAxes,
                           verticalalignment='top', fontsize=9, fontfamily='monospace')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training plots saved to {save_path}")

        plt.show()

    def get_training_data(self):
        """Return training data for comparison plots"""
        return {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'policy_losses': self.policy_losses,
            'entropy_losses': self.entropy_losses,
            'eval_rewards': self.eval_rewards,
            'eval_episodes': self.eval_episodes
        }

class REINFORCETrainer:
    """Trainer class for REINFORCE to match SB3 interface"""

    def __init__(self, env, hyperparams=None):
        self.env = env
        self.hyperparams = hyperparams
        self.agent = None

    def train(self, total_episodes=2000, eval_freq=100):
        """
        Train REINFORCE agent

        Args:
            total_episodes: Number of episodes to train
            eval_freq: Frequency of evaluation
        """
        # Get environment dimensions
        state_size = self.env.observation_space.shape[0]
        action_size = self.env.action_space.n

        # Create agent
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.agent = REINFORCEAgent(
            state_size, action_size,
            device=device, hyperparams=self.hyperparams
        )

        print("Starting REINFORCE training...")
        print(f"Hyperparameters: {self.agent.hyperparams}")

        start_time = time.time()

        for episode in range(total_episodes):
            # Reset environment
            state, _ = self.env.reset()
            episode_reward = 0
            episode_length = 0
            done = False

            # Run episode
            while not done:
                action = self.agent.act(state, training=True)
                next_state, reward, done, truncated, _ = self.env.step(action)

                self.agent.store_reward(reward)
                episode_reward += reward
                episode_length += 1
                state = next_state

                if truncated:
                    done = True

            # Update policy
            loss = self.agent.update()

            # Store episode stats
            self.agent.episode_rewards.append(episode_reward)
            self.agent.episode_lengths.append(episode_length)

            # Evaluate periodically
            if episode % eval_freq == 0:
                mean_reward, std_reward, mean_length = self.agent.evaluate(self.env, n_episodes=5)
                self.agent.eval_rewards.append(mean_reward)
                self.agent.eval_episodes.append(episode)

                print(f"Episode {episode}: Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}, "
                      f"Mean Length: {mean_length:.1f}")

            # Progress update
            if episode % 100 == 0:
                recent_rewards = self.agent.episode_rewards[-10:]
                print(f"Episode {episode}: Recent 10 episodes mean: {np.mean(recent_rewards):.2f}")

        self.agent.training_time = time.time() - start_time
        print(f"Training completed in {self.agent.training_time:.2f} seconds!")

        return self.agent

    def evaluate(self, n_episodes=10):
        """Evaluate trained agent"""
        if self.agent is None:
            raise ValueError("Agent not trained yet!")

        mean_reward, std_reward, mean_length = self.agent.evaluate(self.env, n_episodes)
        print(f"Evaluation over {n_episodes} episodes:")
        print(f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
        print(f"Mean length: {mean_length:.1f}")

        return mean_reward, std_reward

    def get_training_data(self):
        """Get training data for comparison"""
        if self.agent is None:
            return None
        return self.agent.get_training_data()

def run_reinforce_experiment(env, total_episodes=2000, hyperparams=None):
    """
    Run complete REINFORCE experiment

    Args:
        env: Gym environment
        total_episodes: Training duration
        hyperparams: Custom hyperparameters

    Returns:
        trained_agent, training_data
    """
    # Create trainer
    trainer = REINFORCETrainer(env, hyperparams)

    # Train agent
    agent = trainer.train(total_episodes=total_episodes, eval_freq=max(100, total_episodes // 20))

    # Evaluate final performance
    mean_reward, std_reward = trainer.evaluate(n_episodes=20)

    # Plot results
    os.makedirs('plots', exist_ok=True)
    agent.plot_training_results(save_path='plots/reinforce_training_results.png')

    # Save model
    os.makedirs('models', exist_ok=True)
    agent.save('models/reinforce_hospital_navigation.pth')

    return agent, trainer.get_training_data()

def justify_reinforce_hyperparameters():
    """
    Justification for chosen REINFORCE hyperparameters:

    1. Learning Rate (1e-3): Higher than value-based methods since REINFORCE
       has high variance and needs larger steps to overcome noise.

    2. Gamma (0.99): High discount factor for long hospital episodes where
       future patient outcomes matter significantly.

    3. Baseline ('mean'): Uses running mean of returns as baseline to reduce
       variance, crucial for REINFORCE which has inherently high variance.

    4. Entropy Coefficient (0.01): Small entropy bonus to encourage exploration
       of different hospital navigation strategies.

    5. Gradient Clipping (1.0): Essential for REINFORCE to prevent exploding
       gradients due to high variance of policy gradient estimates.

    6. Network Architecture [128, 128, 128]: Deep network to capture complex
       relationships in hospital environment, with dropout for regularization.

    7. Normalization: Returns are normalized per episode to stabilize learning
       across different reward scales in hospital scenarios.

    REINFORCE Challenges in Hospital Environment:
    - High variance due to sparse rewards (patient deliveries)
    - Long episodes (~1000 steps) amplify variance
    - Complex state space requires good exploration

    Mitigations Applied:
    - Baseline subtraction for variance reduction
    - Gradient clipping for stability
    - Entropy regularization for exploration
    - Return normalization for consistent learning
    """
    pass

if __name__ == "__main__":
    # This would be run with your hospital environment
    # from hospital_env import HospitalNavigationEnv
    # env = HospitalNavigationEnv(render_mode=None)
    # agent, training_data = run_reinforce_experiment(env, total_episodes=2000)
    pass