# dqn_agent.py

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
import torch
import os
from typing import Dict, Any

class TrainingCallback(BaseCallback):
    """
    Custom callback for tracking training progress and logging metrics
    """
    def __init__(self, eval_env, eval_freq=1000, verbose=1):
        super(TrainingCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.episode_rewards = []
        self.episode_lengths = []
        self.eval_rewards = []
        self.eval_steps = []
        self.losses = []

    def _on_step(self) -> bool:
        # Log episode rewards
        if len(self.locals.get('infos', [])) > 0:
            for info in self.locals['infos']:
                if 'episode' in info:
                    self.episode_rewards.append(info['episode']['r'])
                    self.episode_lengths.append(info['episode']['l'])

        # Evaluate periodically
        if self.n_calls % self.eval_freq == 0:
            mean_reward, std_reward = evaluate_policy(
                self.model, self.eval_env, n_eval_episodes=5, deterministic=True
            )
            self.eval_rewards.append(mean_reward)
            self.eval_steps.append(self.n_calls)

            if self.verbose > 0:
                print(f"Step {self.n_calls}: Mean eval reward: {mean_reward:.2f} +/- {std_reward:.2f}")

        return True

class DQNTrainer:
    """
    DQN Trainer using Stable Baselines3
    """
    def __init__(self, env, hyperparams=None):
        self.env = env

        # Default hyperparameters (well-tuned for hospital navigation)
        # self.default_hyperparams = {
        #     'learning_rate': 1e-4,          # Lower LR for stable training
        #     'buffer_size': 50000,           # Large buffer for diverse experiences
        #     'learning_starts': 1000,        # Start learning after initial exploration
        #     'batch_size': 32,               # Standard batch size
        #     'tau': 1.0,                     # Hard update (equivalent to target_update_interval)
        #     'gamma': 0.99,                  # Discount factor
        #     'train_freq': 4,                # Train every 4 steps
        #     'gradient_steps': 1,            # Number of gradient steps per update
        #     'target_update_interval': 1000, # Update target network frequency
        #     'exploration_fraction': 0.3,    # Fraction of training for exploration
        #     'exploration_initial_eps': 1.0, # Initial epsilon
        #     'exploration_final_eps': 0.02,  # Final epsilon
        #     'max_grad_norm': 10,           # Gradient clipping
        #     'policy_kwargs': {
        #         'net_arch': [128, 128, 128],  # Network architecture
        #         'activation_fn': torch.nn.ReLU,
        #     },
        #     'verbose': 1,
        #     'device': 'auto'
        # }
        self.default_hyperparams = {
            'learning_rate': 3e-4,              # Increased for faster learning
            'buffer_size': 100000,              # Larger buffer for complex environment
            'learning_starts': 5000,            # More initial exploration before learning
            'batch_size': 64,                   # Larger batches for stable gradients

            # Target network updates
            'tau': 1.0,                         # Keep hard updates
            'target_update_interval': 2000,     # Less frequent target updates for stability

            # Discount and training frequency
            'gamma': 0.995,                     # Higher discount for long-term planning
            'train_freq': 1,                    # Train every step (more frequent updates)
            'gradient_steps': 1,                # Keep at 1

            # IMPROVED EXPLORATION (Key fix for oscillation)
            'exploration_fraction': 0.6,        # 60% of training for exploration (doubled!)
            'exploration_initial_eps': 1.0,     # Start with full exploration
            'exploration_final_eps': 0.1,       # Higher final exploration (5x increase!)

            # Gradient clipping
            'max_grad_norm': 10,                # Keep gradient clipping

            # IMPROVED NETWORK ARCHITECTURE
            'policy_kwargs': {
                'net_arch': [256, 256, 128, 64], # Deeper, wider network
                'activation_fn': torch.nn.ReLU,
                'normalize_images': False,       # Not using images
            },

            'verbose': 1,
            'device': 'auto'
        }

        # Override with custom hyperparams if provided
        if hyperparams:
            self.hyperparams = {**self.default_hyperparams, **hyperparams}
        else:
            self.hyperparams = self.default_hyperparams

        self.model = None
        self.callback = None

    def create_model(self):
        """Create DQN model with specified hyperparameters"""
        self.model = DQN(
            policy="MlpPolicy",
            env=self.env,
            **self.hyperparams
        )
        return self.model

    def train(self, total_timesteps=100000, eval_env=None):
        """
        Train the DQN model

        Args:
            total_timesteps: Total training steps
            eval_env: Environment for evaluation during training
        """
        if self.model is None:
            self.create_model()

        # Create callback for monitoring
        if eval_env is None:
            eval_env = self.env

        self.callback = TrainingCallback(
            eval_env=eval_env,
            eval_freq=max(1000, total_timesteps // 50),  # Evaluate 50 times during training
            verbose=1
        )

        # Train the model
        print("Starting DQN training...")
        print(f"Hyperparameters: {self.hyperparams}")

        self.model.learn(
            total_timesteps=total_timesteps,
            callback=self.callback,
            log_interval=10
        )

        print("Training completed!")
        return self.model

    def evaluate(self, n_episodes=10):
        """Evaluate trained model"""
        if self.model is None:
            raise ValueError("Model not trained yet!")

        mean_reward, std_reward = evaluate_policy(
            self.model, self.env, n_eval_episodes=n_episodes, deterministic=True
        )

        print(f"Evaluation over {n_episodes} episodes:")
        print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

        return mean_reward, std_reward

    def save_model(self, filepath):
        """Save trained model"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        self.model.save(filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        """Load trained model"""
        self.model = DQN.load(filepath, env=self.env)
        print(f"Model loaded from {filepath}")
        return self.model

    def plot_training_results(self, save_path=None):
        """Plot training results"""
        if self.callback is None:
            print("No training data available. Train the model first.")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('DQN Training Results', fontsize=16, fontweight='bold')

        # Episode rewards
        if self.callback.episode_rewards:
            axes[0, 0].plot(self.callback.episode_rewards, alpha=0.7)
            axes[0, 0].set_title('Episode Rewards During Training')
            axes[0, 0].set_xlabel('Episode')
            axes[0, 0].set_ylabel('Total Reward')
            axes[0, 0].grid(True, alpha=0.3)

            # Add moving average
            if len(self.callback.episode_rewards) > 10:
                window = min(50, len(self.callback.episode_rewards) // 10)
                moving_avg = np.convolve(self.callback.episode_rewards,
                                       np.ones(window)/window, mode='valid')
                axes[0, 0].plot(range(window-1, len(self.callback.episode_rewards)),
                              moving_avg, 'r-', linewidth=2, label=f'Moving Avg ({window})')
                axes[0, 0].legend()

        # Episode lengths
        if self.callback.episode_lengths:
            axes[0, 1].plot(self.callback.episode_lengths, alpha=0.7, color='orange')
            axes[0, 1].set_title('Episode Lengths During Training')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Steps')
            axes[0, 1].grid(True, alpha=0.3)

        # Evaluation rewards
        if self.callback.eval_rewards:
            axes[1, 0].plot(self.callback.eval_steps, self.callback.eval_rewards,
                          'go-', linewidth=2, markersize=6)
            axes[1, 0].set_title('Evaluation Rewards')
            axes[1, 0].set_xlabel('Training Steps')
            axes[1, 0].set_ylabel('Mean Reward')
            axes[1, 0].grid(True, alpha=0.3)

        # Training statistics summary
        axes[1, 1].axis('off')
        if self.callback.episode_rewards:
            stats_text = f"""
            Training Statistics:

            Total Episodes: {len(self.callback.episode_rewards)}
            Mean Episode Reward: {np.mean(self.callback.episode_rewards):.2f}
            Std Episode Reward: {np.std(self.callback.episode_rewards):.2f}
            Max Episode Reward: {np.max(self.callback.episode_rewards):.2f}
            Min Episode Reward: {np.min(self.callback.episode_rewards):.2f}

            Final 10 Episodes Mean: {np.mean(self.callback.episode_rewards[-10:]):.2f}

            Hyperparameters:
            Learning Rate: {self.hyperparams['learning_rate']}
            Buffer Size: {self.hyperparams['buffer_size']}
            Batch Size: {self.hyperparams['batch_size']}
            Gamma: {self.hyperparams['gamma']}
            """
            axes[1, 1].text(0.1, 0.9, stats_text, transform=axes[1, 1].transAxes,
                           verticalalignment='top', fontsize=10, fontfamily='monospace')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training plots saved to {save_path}")

        plt.show()

    def get_training_data(self):
        """Return training data for comparison plots"""
        if self.callback is None:
            return None

        return {
            'episode_rewards': self.callback.episode_rewards,
            'episode_lengths': self.callback.episode_lengths,
            'eval_rewards': self.callback.eval_rewards,
            'eval_steps': self.callback.eval_steps
        }

def run_dqn_experiment(env, total_timesteps=100000, hyperparams=None):
    """
    Run complete DQN experiment

    Args:
        env: Gym environment
        total_timesteps: Training duration
        hyperparams: Custom hyperparameters

    Returns:
        trained_model, training_data
    """
    # Create trainer
    trainer = DQNTrainer(env, hyperparams)

    # Train model
    model = trainer.train(total_timesteps=total_timesteps, eval_env=env)

    # Evaluate final performance
    mean_reward, std_reward = trainer.evaluate(n_episodes=20)

    # Plot results
    os.makedirs('plots', exist_ok=True)
    trainer.plot_training_results(save_path='plots/dqn_training_results.png')

    # Save model
    os.makedirs('models', exist_ok=True)
    trainer.save_model('models/dqn_hospital_navigation.zip')

    return model, trainer.get_training_data()
