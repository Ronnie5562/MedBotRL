# ppo_agent.py

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import torch
import os
from typing import Dict, Any

class PPOTrainingCallback(BaseCallback):
    """
    Custom callback for tracking PPO training progress and logging metrics
    """
    def __init__(self, eval_env, eval_freq=2000, verbose=1):
        super(PPOTrainingCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.episode_rewards = []
        self.episode_lengths = []
        self.eval_rewards = []
        self.eval_steps = []
        self.policy_losses = []
        self.value_losses = []
        self.entropy_losses = []
        self.explained_variances = []

    def _on_step(self) -> bool:
        # Log episode rewards and lengths
        if len(self.locals.get('infos', [])) > 0:
            for info in self.locals['infos']:
                if 'episode' in info:
                    self.episode_rewards.append(info['episode']['r'])
                    self.episode_lengths.append(info['episode']['l'])

        # Log training losses (available after policy updates)
        if hasattr(self.model, 'logger') and self.model.logger.name_to_value:
            logs = self.model.logger.name_to_value
            if 'train/policy_loss' in logs:
                self.policy_losses.append(logs['train/policy_loss'])
            if 'train/value_loss' in logs:
                self.value_losses.append(logs['train/value_loss'])
            if 'train/entropy_loss' in logs:
                self.entropy_losses.append(logs['train/entropy_loss'])
            if 'train/explained_variance' in logs:
                self.explained_variances.append(logs['train/explained_variance'])

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

class PPOTrainer:
    """
    PPO Trainer using Stable Baselines3
    """
    def __init__(self, env, hyperparams=None):
        self.env = env

        # Default hyperparameters (well-tuned for hospital navigation)
        self.default_hyperparams = {
            'learning_rate': 3e-4,          # Standard PPO learning rate
            'n_steps': 2048,                # Steps per rollout (hospital episodes ~1000 steps)
            'batch_size': 64,               # Mini-batch size for SGD
            'n_epochs': 10,                 # Number of epochs per rollout
            'gamma': 0.99,                  # Discount factor
            'gae_lambda': 0.95,             # GAE parameter
            'clip_range': 0.2,              # PPO clipping parameter
            'clip_range_vf': None,          # Value function clipping (None = no clipping)
            'normalize_advantage': True,     # Normalize advantages
            'ent_coef': 0.01,               # Entropy coefficient for exploration
            'vf_coef': 0.5,                 # Value function coefficient
            'max_grad_norm': 0.5,           # Gradient clipping
            'use_sde': False,               # State-dependent exploration
            'sde_sample_freq': -1,          # SDE sampling frequency
            'target_kl': None,              # Target KL divergence
            'policy_kwargs': {
                'net_arch': [dict(pi=[128, 128], vf=[128, 128])],  # Separate actor-critic networks
                'activation_fn': torch.nn.Tanh,  # Tanh activation for stable gradients
                'ortho_init': True,               # Orthogonal initialization
                'log_std_init': 0.0,             # Log std initialization
                'full_std': True,                # Use full std for continuous actions
                'use_expln': False,              # Use exponential likelihood norm
                'squash_output': False,          # Don't squash output
                # 'features_extractor_kwargs': {'flatten': True}
            },
            'verbose': 1,
            'device': 'auto',
            'seed': None
        }

        # Override with custom hyperparams if provided
        if hyperparams:
            self.hyperparams = {**self.default_hyperparams, **hyperparams}
        else:
            self.hyperparams = self.default_hyperparams

        self.model = None
        self.callback = None

    def create_model(self):
        """Create PPO model with specified hyperparameters"""
        self.model = PPO(
            policy="MlpPolicy",
            env=self.env,
            **self.hyperparams
        )
        return self.model

    def train(self, total_timesteps=200000, eval_env=None):
        """
        Train the PPO model

        Args:
            total_timesteps: Total training steps
            eval_env: Environment for evaluation during training
        """
        if self.model is None:
            self.create_model()

        # Create callback for monitoring
        if eval_env is None:
            eval_env = self.env

        self.callback = PPOTrainingCallback(
            eval_env=eval_env,
            eval_freq=max(2000, total_timesteps // 40),  # Evaluate 40 times during training
            verbose=1
        )

        # Train the model
        print("Starting PPO training...")
        print(f"Hyperparameters: {self.hyperparams}")

        self.model.learn(
            total_timesteps=total_timesteps,
            callback=self.callback,
            log_interval=5  # Log every 5 rollouts
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
        self.model = PPO.load(filepath, env=self.env)
        print(f"Model loaded from {filepath}")
        return self.model

    def plot_training_results(self, save_path=None):
        """Plot comprehensive training results"""
        if self.callback is None:
            print("No training data available. Train the model first.")
            return

        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle('PPO Training Results', fontsize=16, fontweight='bold')

        # Episode rewards
        if self.callback.episode_rewards:
            axes[0, 0].plot(self.callback.episode_rewards, alpha=0.7, color='blue')
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

        # Policy losses
        if self.callback.policy_losses:
            axes[1, 0].plot(self.callback.policy_losses, color='red', alpha=0.8)
            axes[1, 0].set_title('Policy Loss')
            axes[1, 0].set_xlabel('Update Step')
            axes[1, 0].set_ylabel('Policy Loss')
            axes[1, 0].grid(True, alpha=0.3)

        # Value losses
        if self.callback.value_losses:
            axes[1, 1].plot(self.callback.value_losses, color='green', alpha=0.8)
            axes[1, 1].set_title('Value Function Loss')
            axes[1, 1].set_xlabel('Update Step')
            axes[1, 1].set_ylabel('Value Loss')
            axes[1, 1].grid(True, alpha=0.3)

        # Evaluation rewards
        if self.callback.eval_rewards:
            axes[2, 0].plot(self.callback.eval_steps, self.callback.eval_rewards,
                          'go-', linewidth=2, markersize=6)
            axes[2, 0].set_title('Evaluation Rewards')
            axes[2, 0].set_xlabel('Training Steps')
            axes[2, 0].set_ylabel('Mean Reward')
            axes[2, 0].grid(True, alpha=0.3)

        # Training statistics summary
        axes[2, 1].axis('off')
        if self.callback.episode_rewards:
            stats_text = f"""
            Training Statistics:

            Total Episodes: {len(self.callback.episode_rewards)}
            Mean Episode Reward: {np.mean(self.callback.episode_rewards):.2f}
            Std Episode Reward: {np.std(self.callback.episode_rewards):.2f}
            Max Episode Reward: {np.max(self.callback.episode_rewards):.2f}
            Min Episode Reward: {np.min(self.callback.episode_rewards):.2f}

            Final 10 Episodes Mean: {np.mean(self.callback.episode_rewards[-10:]):.2f}

            Key Hyperparameters:
            Learning Rate: {self.hyperparams['learning_rate']}
            Rollout Steps: {self.hyperparams['n_steps']}
            Batch Size: {self.hyperparams['batch_size']}
            PPO Epochs: {self.hyperparams['n_epochs']}
            Clip Range: {self.hyperparams['clip_range']}
            """
            axes[2, 1].text(0.1, 0.9, stats_text, transform=axes[2, 1].transAxes,
                           verticalalignment='top', fontsize=9, fontfamily='monospace')

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
            'eval_steps': self.callback.eval_steps,
            'policy_losses': self.callback.policy_losses,
            'value_losses': self.callback.value_losses,
            'entropy_losses': self.callback.entropy_losses
        }

def run_ppo_experiment(env, total_timesteps=200000, hyperparams=None):
    """
    Run complete PPO experiment

    Args:
        env: Gym environment
        total_timesteps: Training duration (PPO typically needs more steps than DQN)
        hyperparams: Custom hyperparameters

    Returns:
        trained_model, training_data
    """
    # Create trainer
    trainer = PPOTrainer(env, hyperparams)

    # Train model
    model = trainer.train(total_timesteps=total_timesteps, eval_env=env)

    # Evaluate final performance
    mean_reward, std_reward = trainer.evaluate(n_episodes=20)

    # Plot results
    os.makedirs('plots', exist_ok=True)
    trainer.plot_training_results(save_path='plots/ppo_training_results.png')

    # Save model
    os.makedirs('models', exist_ok=True)
    trainer.save_model('models/ppo_hospital_navigation.zip')

    return model, trainer.get_training_data()

if __name__ == "__main__":
    # This would be run with your hospital environment
    # from hospital_env import HospitalNavigationEnv
    # env = HospitalNavigationEnv(render_mode=None)
    # model, training_data = run_ppo_experiment(env, total_timesteps=200000)
    pass
