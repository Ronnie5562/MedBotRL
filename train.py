# train.py
import numpy as np
import matplotlib.pyplot as plt
import torch
import argparse
import json
import os
from datetime import datetime

from hospital_env import HospitalNavigationEnv

from dqn_agent import run_dqn_experiment
from ppo_agent import run_ppo_experiment
from reinforce_agent import run_reinforce_experiment
from a2c_agent import run_a2c_experiment



def evaluate_agent_sb3(model, env, episodes=100):
    """Evaluate Stable Baselines3 model"""
    from stable_baselines3.common.evaluation import evaluate_policy

    mean_reward, std_reward = evaluate_policy(
        model, env, n_eval_episodes=episodes, deterministic=True
    )

    print(f"Evaluation Results:")
    print(f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")

    return mean_reward, std_reward


def evaluate_agent_reinforce(agent, env, episodes=100):
    """Evaluate REINFORCE agent"""
    total_rewards = []

    for _ in range(episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False

        while not done:
            action = agent.act(state, training=False)
            state, reward, done, truncated, _ = env.step(action)
            episode_reward += reward

            if truncated:
                done = True

        total_rewards.append(episode_reward)

    mean_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)

    print(f"Evaluation Results:")
    print(f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")

    return mean_reward, std_reward


def compare_algorithms(dqn_timesteps=100000, ppo_timesteps=200000,
                      reinforce_episodes=2000, a2c_timesteps=100000):
    """Compare all four algorithms using their existing experiment functions"""
    print("Comparing DQN, PPO, REINFORCE, and A2C algorithms...")

    # Create environment
    env = HospitalNavigationEnv(render_mode=None)

    # Create directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs("plots", exist_ok=True)

    results = {}

    # Train DQN
    print("\n" + "="*50)
    print("Training DQN...")
    dqn_model, dqn_data = run_dqn_experiment(env, total_timesteps=dqn_timesteps)
    results['DQN'] = {
        'model': dqn_model,
        'data': dqn_data,
        'final_avg_score': np.mean(dqn_data['episode_rewards'][-100:]) if len(dqn_data['episode_rewards']) >= 100 else np.mean(dqn_data['episode_rewards']),
        'type': 'Value-Based (SB3)',
        'is_sb3': True
    }

    # Train PPO
    print("\n" + "="*50)
    print("Training PPO...")
    ppo_model, ppo_data = run_ppo_experiment(env, total_timesteps=ppo_timesteps)
    results['PPO'] = {
        'model': ppo_model,
        'data': ppo_data,
        'final_avg_score': np.mean(ppo_data['episode_rewards'][-100:]) if len(ppo_data['episode_rewards']) >= 100 else np.mean(ppo_data['episode_rewards']),
        'type': 'Policy Gradient (SB3)',
        'is_sb3': True
    }

    # Train REINFORCE
    print("\n" + "="*50)
    print("Training REINFORCE...")
    reinforce_agent, reinforce_data = run_reinforce_experiment(env, total_episodes=reinforce_episodes)
    results['REINFORCE'] = {
        'model': reinforce_agent,
        'data': reinforce_data,
        'final_avg_score': np.mean(reinforce_data['episode_rewards'][-100:]) if len(reinforce_data['episode_rewards']) >= 100 else np.mean(reinforce_data['episode_rewards']),
        'type': 'Policy Gradient (Custom)',
        'is_sb3': False
    }

    # Train A2C
    print("\n" + "="*50)
    print("Training A2C...")
    a2c_model, a2c_data = run_a2c_experiment(env, total_timesteps=a2c_timesteps)
    results['A2C'] = {
        'model': a2c_model,
        'data': a2c_data,
        'final_avg_score': np.mean(a2c_data['episode_rewards'][-100:]) if len(a2c_data['episode_rewards']) >= 100 else np.mean(a2c_data['episode_rewards']),
        'type': 'Actor-Critic (SB3)',
        'is_sb3': True
    }

    # Plot comparisons
    plot_comparison(results)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_results = {}
    for alg, result in results.items():
        save_results[alg] = {
            'episode_rewards': [float(x) for x in result['data']['episode_rewards']],
            'final_avg_score': float(result['final_avg_score']),
            'type': result['type']
        }
        if 'eval_rewards' in result['data']:
            save_results[alg]['eval_rewards'] = [float(x) for x in result['data']['eval_rewards']]
            save_results[alg]['eval_steps'] = [int(x) for x in result['data']['eval_steps']]

    with open(f"results/comparison_{timestamp}.json", 'w') as f:
        json.dump(save_results, f, indent=2)

    # Print final comparison
    print("\n" + "="*60)
    print("FINAL COMPARISON RESULTS")
    print("="*60)

    # Sort by performance
    sorted_results = sorted(results.items(), key=lambda x: x[1]['final_avg_score'], reverse=True)

    for i, (alg, result) in enumerate(sorted_results, 1):
        print(f"{i}. {alg:12} ({result['type']})")
        print(f"   Final Average Score: {result['final_avg_score']:.2f}")
        print(f"   Total Episodes: {len(result['data']['episode_rewards'])}")
        if 'eval_rewards' in result['data']:
            print(f"   Evaluation Points: {len(result['data']['eval_rewards'])}")
        print()

    env.close()
    return results


def plot_comparison(results):
    """Plot comparison of algorithms"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('RL Algorithms Comparison - Hospital Navigation Environment',
                 fontsize=16, fontweight='bold')

    colors = {'DQN': '#1f77b4', 'PPO': '#ff7f0e', 'REINFORCE': '#2ca02c', 'A2C': '#d62728'}

    # Plot 1: Episode Rewards
    for alg_name, result in results.items():
        rewards = result['data']['episode_rewards']
        episodes = range(len(rewards))
        ax1.plot(episodes, rewards, color=colors.get(alg_name, 'black'),
                alpha=0.7, linewidth=1, label=f'{alg_name}')

    ax1.set_title('Episode Rewards')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Moving Average
    window = 50
    for alg_name, result in results.items():
        rewards = result['data']['episode_rewards']
        if len(rewards) >= window:
            moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
            ax2.plot(range(window-1, len(rewards)), moving_avg,
                    color=colors.get(alg_name, 'black'), linewidth=2,
                    label=f'{alg_name} (MA-{window})')

    ax2.set_title(f'Moving Average ({window} episodes)')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Average Reward')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Final Performance
    alg_names = list(results.keys())
    final_scores = [results[alg]['final_avg_score'] for alg in alg_names]
    bar_colors = [colors.get(alg, 'gray') for alg in alg_names]

    bars = ax3.bar(alg_names, final_scores, color=bar_colors, alpha=0.7, edgecolor='black')
    ax3.set_title('Final Performance Comparison')
    ax3.set_ylabel('Average Reward (Final 100 episodes)')
    ax3.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, score in zip(bars, final_scores):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + abs(height)*0.02,
                f'{score:.1f}', ha='center', va='bottom', fontweight='bold')

    # Plot 4: Evaluation Rewards (for algorithms that have them)
    eval_algs = [alg for alg in results if 'eval_rewards' in results[alg]['data']]
    if eval_algs:
        for alg_name in eval_algs:
            eval_data = results[alg_name]['data']
            if eval_data['eval_rewards'] and eval_data['eval_steps']:
                ax4.plot(eval_data['eval_steps'], eval_data['eval_rewards'],
                        color=colors.get(alg_name, 'black'), linewidth=2,
                        marker='o', markersize=4, label=f'{alg_name}')

        ax4.set_title('Evaluation Performance During Training')
        ax4.set_xlabel('Training Steps')
        ax4.set_ylabel('Evaluation Reward')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'No evaluation data available',
                ha='center', va='center', transform=ax4.transAxes, fontsize=12)
        ax4.set_title('Evaluation Performance')

    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f'results/comparison_plot_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.show()


def train_single_algorithm(algorithm, timesteps=None, episodes=None, evaluate=False):
    """Train a single algorithm"""
    env = HospitalNavigationEnv(render_mode=None)

    if algorithm == 'dqn':
        timesteps = timesteps or 100000
        print(f"Training DQN for {timesteps} timesteps...")
        model, data = run_dqn_experiment(env, total_timesteps=timesteps)

        if evaluate:
            evaluate_agent_sb3(model, env)

    elif algorithm == 'ppo':
        timesteps = timesteps or 200000
        print(f"Training PPO for {timesteps} timesteps...")
        model, data = run_ppo_experiment(env, total_timesteps=timesteps)

        if evaluate:
            evaluate_agent_sb3(model, env)

    elif algorithm == 'reinforce':
        episodes = episodes or 2000
        print(f"Training REINFORCE for {episodes} episodes...")
        agent, data = run_reinforce_experiment(env, total_episodes=episodes)

        if evaluate:
            evaluate_agent_reinforce(agent, env)

    elif algorithm == 'a2c':
        timesteps = timesteps or 100000
        print(f"Training A2C for {timesteps} timesteps...")
        model, data = run_a2c_experiment(env, total_timesteps=timesteps)

        if evaluate:
            evaluate_agent_sb3(model, env)

    env.close()
    return model if algorithm != 'reinforce' else agent, data


def main():
    parser = argparse.ArgumentParser(description='Train RL agents for Hospital Navigation')
    parser.add_argument('--algorithm', choices=['dqn', 'ppo', 'reinforce', 'a2c', 'all'],
                       default='all', help='Algorithm to train')
    parser.add_argument('--timesteps', type=int, default=100000,
                       help='Number of timesteps (for DQN, PPO, A2C)')
    parser.add_argument('--episodes', type=int, default=2000,
                       help='Number of episodes (for REINFORCE)')
    parser.add_argument('--evaluate', action='store_true',
                       help='Evaluate trained model')

    args = parser.parse_args()

    if args.algorithm == 'all':
        compare_algorithms(
            dqn_timesteps=args.timesteps,
            ppo_timesteps=args.timesteps * 2,  # PPO typically needs more timesteps
            reinforce_episodes=args.episodes,
            a2c_timesteps=args.timesteps
        )
    else:
        train_single_algorithm(
            args.algorithm,
            timesteps=args.timesteps,
            episodes=args.episodes,
            evaluate=args.evaluate
        )


if __name__ == "__main__":
    main()
