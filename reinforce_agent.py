# reinforce_agent.py - Enhanced with Training Stability Analysis and .zip saving

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import matplotlib.pyplot as plt
import os
import zipfile
import pickle
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

        # REINFORCE Hyperparameters - Stronger Exploration
        self.default_hyperparams = {
            'learning_rate': 5e-4,          # Reduced learning rate
            'gamma': 0.995,                 # Higher discount factor
            'hidden_size': 256,             # Larger hidden layer
            'max_grad_norm': 0.5,           # Tighter gradient clipping
            'baseline_type': 'mean',        # Keep mean baseline
            'entropy_coef': 0.1,
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

        # NEW: Training stability tracking
        self.policy_entropy_values = []  # Track entropy for stability analysis
        self.raw_policy_losses = []      # Raw losses before baseline adjustment
        self.update_numbers = []         # Track update step numbers

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

        # Store losses and stability metrics
        self.policy_losses.append(policy_loss.item())
        self.entropy_losses.append(entropy_loss.item())

        # NEW: Store stability tracking
        current_entropy = torch.stack(self.entropies).mean().item()
        self.policy_entropy_values.append(current_entropy)
        self.raw_policy_losses.append(policy_loss.item())
        self.update_numbers.append(len(self.policy_losses))

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
        """Save model and training state as .zip file (compatible with SB3 format)"""
        # Create temporary directory for saving components
        temp_dir = f"temp_reinforce_{int(time.time())}"
        os.makedirs(temp_dir, exist_ok=True)

        try:
            # Save PyTorch model
            model_path = os.path.join(temp_dir, "policy_net.pth")
            torch.save({
                'policy_net_state_dict': self.policy_net.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'state_size': self.state_size,
                'action_size': self.action_size,
            }, model_path)

            # Save hyperparameters and training data
            data_path = os.path.join(temp_dir, "training_data.pkl")
            training_data = {
                'hyperparams': self.hyperparams,
                'reward_baseline': self.reward_baseline,
                'episode_rewards': self.episode_rewards,
                'policy_losses': self.policy_losses,
                'entropy_losses': self.entropy_losses,
                'policy_entropy_values': self.policy_entropy_values,
                'raw_policy_losses': self.raw_policy_losses,
                'update_numbers': self.update_numbers,
                'eval_rewards': self.eval_rewards,
                'eval_episodes': self.eval_episodes,
                'episode_lengths': self.episode_lengths,
                'training_time': self.training_time
            }

            with open(data_path, 'wb') as f:
                pickle.dump(training_data, f)

            # Create zip file
            with zipfile.ZipFile(filepath, 'w', zipfile.ZIP_DEFLATED) as zipf:
                zipf.write(model_path, "policy_net.pth")
                zipf.write(data_path, "training_data.pkl")

            print(f"REINFORCE model saved to {filepath}")

        finally:
            # Clean up temporary files
            if os.path.exists(model_path):
                os.remove(model_path)
            if os.path.exists(data_path):
                os.remove(data_path)
            if os.path.exists(temp_dir):
                os.rmdir(temp_dir)

    def load(self, filepath):
        """Load model and training state from .zip file"""
        temp_dir = f"temp_reinforce_load_{int(time.time())}"
        os.makedirs(temp_dir, exist_ok=True)

        try:
            # Extract zip file
            with zipfile.ZipFile(filepath, 'r') as zipf:
                zipf.extractall(temp_dir)

            # Load PyTorch model
            model_path = os.path.join(temp_dir, "policy_net.pth")
            checkpoint = torch.load(model_path, map_location=self.device)
            self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            # Load training data
            data_path = os.path.join(temp_dir, "training_data.pkl")
            with open(data_path, 'rb') as f:
                training_data = pickle.load(f)

            self.reward_baseline = training_data.get('reward_baseline', None)
            self.episode_rewards = training_data.get('episode_rewards', [])
            self.policy_losses = training_data.get('policy_losses', [])
            self.entropy_losses = training_data.get('entropy_losses', [])
            self.policy_entropy_values = training_data.get('policy_entropy_values', [])
            self.raw_policy_losses = training_data.get('raw_policy_losses', [])
            self.update_numbers = training_data.get('update_numbers', [])
            self.eval_rewards = training_data.get('eval_rewards', [])
            self.eval_episodes = training_data.get('eval_episodes', [])
            self.episode_lengths = training_data.get('episode_lengths', [])
            self.training_time = training_data.get('training_time', 0)

            print(f"REINFORCE model loaded from {filepath}")

        finally:
            # Clean up temporary files
            for file in ["policy_net.pth", "training_data.pkl"]:
                file_path = os.path.join(temp_dir, file)
                if os.path.exists(file_path):
                    os.remove(file_path)
            if os.path.exists(temp_dir):
                os.rmdir(temp_dir)

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

    # NEW METHOD: Training Stability Analysis
    def plot_training_stability(self, save_path=None):
        """
        Plot training stability analysis - objective function and policy entropy curves
        This is specifically for the mentor's requirement on training stability
        """
        if not self.policy_losses:
            print("No training data available. Train the model first.")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('REINFORCE Training Stability Analysis', fontsize=16, fontweight='bold')

        # Policy Loss (Objective Function) - Main stability indicator
        if self.policy_losses and len(self.policy_losses) > 1:
            axes[0, 0].plot(self.policy_losses, color='red', alpha=0.8, linewidth=1.5)
            axes[0, 0].set_title('Policy Loss (Objective Function)')
            axes[0, 0].set_xlabel('Episode')
            axes[0, 0].set_ylabel('Policy Loss')
            axes[0, 0].grid(True, alpha=0.3)

            # Add trend line for stability analysis
            if len(self.policy_losses) > 10:
                z = np.polyfit(range(len(self.policy_losses)), self.policy_losses, 1)
                p = np.poly1d(z)
                axes[0, 0].plot(range(len(self.policy_losses)), p(range(len(self.policy_losses))),
                              "g--", alpha=0.8, linewidth=2, label=f'Trend (slope: {z[0]:.6f})')
                axes[0, 0].legend()

        # Policy Entropy - Exploration indicator
        if self.policy_entropy_values and len(self.policy_entropy_values) > 1:
            axes[0, 1].plot(self.policy_entropy_values, color='blue', alpha=0.8, linewidth=1.5)
            axes[0, 1].set_title('Policy Entropy (Exploration Level)')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Policy Entropy')
            axes[0, 1].grid(True, alpha=0.3)

            # Add trend line
            if len(self.policy_entropy_values) > 10:
                z = np.polyfit(range(len(self.policy_entropy_values)), self.policy_entropy_values, 1)
                p = np.poly1d(z)
                axes[0, 1].plot(range(len(self.policy_entropy_values)),
                              p(range(len(self.policy_entropy_values))),
                              "g--", alpha=0.8, linewidth=2, label=f'Trend (slope: {z[0]:.6f})')
                axes[0, 1].legend()

        # Loss Variance (Rolling Window) - Stability indicator
        if len(self.policy_losses) > 20:
            window_size = min(50, len(self.policy_losses) // 4)
            rolling_std = []
            for i in range(window_size, len(self.policy_losses)):
                window_losses = self.policy_losses[i-window_size:i]
                rolling_std.append(np.std(window_losses))

            axes[1, 0].plot(range(window_size, len(self.policy_losses)), rolling_std,
                          color='purple', alpha=0.8, linewidth=1.5)
            axes[1, 0].set_title(f'Policy Loss Variance (Rolling {window_size}-episode window)')
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Loss Standard Deviation')
            axes[1, 0].grid(True, alpha=0.3)

        # Stability Analysis Summary
        axes[1, 1].axis('off')

        # Calculate stability metrics
        stability_analysis = self._calculate_stability_metrics()

        analysis_text = f"""
        TRAINING STABILITY ANALYSIS

        Policy Loss Stability:
        • Mean: {stability_analysis['policy_loss_mean']:.4f}
        • Std: {stability_analysis['policy_loss_std']:.4f}
        • Trend: {stability_analysis['policy_loss_trend']:.6f}
        • Stability: {stability_analysis['policy_loss_stability']}

        Policy Entropy:
        • Mean: {stability_analysis['entropy_mean']:.4f}
        • Std: {stability_analysis['entropy_std']:.4f}
        • Trend: {stability_analysis['entropy_trend']:.6f}
        • Stability: {stability_analysis['entropy_stability']}

        REINFORCE Characteristics:
        • High variance expected (inherent to algorithm)
        • Baseline helps reduce variance
        • Entropy should gradually decrease

        Overall Assessment:
        {stability_analysis['overall_assessment']}

        Key REINFORCE Indicators:
        • Decreasing loss trend = Learning progress
        • Controlled entropy decay = Good exploration→exploitation
        • Baseline effectiveness = Variance reduction
        """

        axes[1, 1].text(0.05, 0.95, analysis_text, transform=axes[1, 1].transAxes,
                       verticalalignment='top', fontsize=9, fontfamily='monospace',
                       bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training stability plots saved to {save_path}")

        plt.show()

    def _calculate_stability_metrics(self):
        """Calculate quantitative stability metrics for REINFORCE"""
        metrics = {}

        # Policy loss stability
        if self.policy_losses:
            losses = np.array(self.policy_losses)
            metrics['policy_loss_mean'] = np.mean(losses)
            metrics['policy_loss_std'] = np.std(losses)

            if len(losses) > 10:
                # Calculate trend (slope of linear fit)
                z = np.polyfit(range(len(losses)), losses, 1)
                metrics['policy_loss_trend'] = z[0]

                # For REINFORCE, we expect higher variance, so adjust thresholds
                cv = metrics['policy_loss_std'] / abs(metrics['policy_loss_mean']) if metrics['policy_loss_mean'] != 0 else float('inf')
                if cv < 0.3:
                    metrics['policy_loss_stability'] = "Very Stable (Excellent for REINFORCE)"
                elif cv < 0.6:
                    metrics['policy_loss_stability'] = "Stable (Good for REINFORCE)"
                elif cv < 1.0:
                    metrics['policy_loss_stability'] = "Moderately Stable (Normal for REINFORCE)"
                else:
                    metrics['policy_loss_stability'] = "High Variance (Consider baseline tuning)"
            else:
                metrics['policy_loss_trend'] = 0.0
                metrics['policy_loss_stability'] = "Insufficient Data"

        # Policy entropy stability
        if self.policy_entropy_values:
            entropy = np.array(self.policy_entropy_values)
            metrics['entropy_mean'] = np.mean(entropy)
            metrics['entropy_std'] = np.std(entropy)

            if len(entropy) > 10:
                z = np.polyfit(range(len(entropy)), entropy, 1)
                metrics['entropy_trend'] = z[0]

                # For entropy, we want it to decrease gradually (exploration → exploitation)
                if z[0] < -0.001:  # Decreasing
                    metrics['entropy_stability'] = "Good (Decreasing as expected)"
                elif abs(z[0]) < 0.001:  # Stable
                    metrics['entropy_stability'] = "Stable (May need more exploration)"
                else:  # Increasing
                    metrics['entropy_stability'] = "Increasing (Check entropy coefficient)"
            else:
                metrics['entropy_trend'] = 0.0
                metrics['entropy_stability'] = "Insufficient Data"

        # Overall assessment for REINFORCE
        if 'policy_loss_stability' in metrics and 'entropy_stability' in metrics:
            if "Stable" in metrics['policy_loss_stability'] and "Good" in metrics['entropy_stability']:
                metrics['overall_assessment'] = "✓ REINFORCE training appears stable with good exploration decay"
            elif "High Variance" in metrics['policy_loss_stability']:
                metrics['overall_assessment'] = "⚠ High variance detected - consider stronger baseline or lower learning rate"
            elif "Increasing" in metrics['entropy_stability']:
                metrics['overall_assessment'] = "⚠ Entropy increasing - check hyperparameters"
            else:
                metrics['overall_assessment'] = "→ Training shows typical REINFORCE behavior"
        else:
            metrics['overall_assessment'] = "Insufficient data for assessment"

        return metrics

    def get_training_data(self):
        """Return training data for comparison plots"""
        return {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'policy_losses': self.policy_losses,
            'entropy_losses': self.entropy_losses,
            'eval_rewards': self.eval_rewards,
            'eval_episodes': self.eval_episodes,
            'policy_entropy_values': self.policy_entropy_values,  # NEW
            'raw_policy_losses': self.raw_policy_losses,          # NEW
            'update_numbers': self.update_numbers                  # NEW
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

    # NEW: Plot training stability analysis
    agent.plot_training_stability(save_path='plots/reinforce_training_stability.png')

    # Save model as .zip (now compatible with other agents)
    os.makedirs('models', exist_ok=True)
    agent.save('models/reinforce_hospital_navigation.zip')

    return agent, trainer.get_training_data()

def justify_reinforce_hyperparameters():
    """
    Justification for chosen REINFORCE hyperparameters:

    1. Learning Rate (5e-4): Reduced from 1e-3 because REINFORCE has high variance
       and needs more careful steps to avoid instability.

    2. Gamma (0.995): Very high discount factor for long hospital episodes where
       future patient outcomes matter significantly.

    3. Baseline ('mean'): Uses running mean of returns as baseline to reduce
       variance, crucial for REINFORCE which has inherently high variance.

    4. Entropy Coefficient (0.1): Higher entropy bonus to encourage exploration
       of different hospital navigation strategies, especially important early in training.

    5. Gradient Clipping (0.5): Tighter clipping essential for REINFORCE to prevent
       exploding gradients due to high variance of policy gradient estimates.

    6. Network Architecture [256, 256, 256]: Larger network to capture complex
       relationships in hospital environment, with dropout for regularization.

    7. Hidden Size (256): Larger than default to handle complex hospital state representations.

    REINFORCE Challenges in Hospital Environment:
    - High variance due to sparse rewards (patient deliveries)
    - Long episodes (~1000 steps) amplify variance
    - Complex state space requires good exploration
    - No value function to guide learning

    Mitigations Applied:
    - Baseline subtraction for variance reduction
    - Tighter gradient clipping for stability
    - Higher entropy regularization for exploration
    - Return normalization for consistent learning
    - Larger network capacity for complex patterns
    """
    pass

if __name__ == "__main__":
    # This would be run with your hospital environment
    # from hospital_env import HospitalNavigationEnv
    # env = HospitalNavigationEnv(render_mode=None)
    # agent, training_data = run_reinforce_experiment(env, total_episodes=2000)
    pass
