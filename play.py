import pygame
import numpy as np
import time
import argparse
import os
from datetime import datetime
import imageio
import json
import glob
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. Only random agent will work.")

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# Try to import common RL libraries that save as .zip
try:
    from stable_baselines3 import DQN, PPO, A2C
    from stable_baselines3.common.policies import BasePolicy
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False

try:
    import ray
    from ray.rllib.algorithms import Algorithm
    RLLIB_AVAILABLE = True
except ImportError:
    RLLIB_AVAILABLE = False

from hospital_env import HospitalNavigationEnv

class DQNAgent(nn.Module):
    """Deep Q-Network Agent for PyTorch models"""

    def __init__(self, state_size, action_size, hidden_size=256):
        super(DQNAgent, self).__init__()
        self.state_size = state_size
        self.action_size = action_size

        # Neural network layers
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, action_size)

        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

    def act(self, state, epsilon=0.0):
        """Choose action using epsilon-greedy policy"""
        if np.random.random() <= epsilon:
            return np.random.randint(0, self.action_size)

        # Convert state to tensor
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).unsqueeze(0)

        # Get Q-values
        with torch.no_grad():
            q_values = self.forward(state)
            action = q_values.argmax().item()

        return action

class RandomAgent:
    """Random agent for demonstration purposes"""

    def __init__(self, action_size):
        self.action_size = action_size

    def act(self, state, epsilon=0.0):
        """Choose random action"""
        return np.random.randint(0, self.action_size)

class StrategicRandomAgent:
    """Random agent with slight bias towards objectives"""

    def __init__(self, action_size):
        self.action_size = action_size
        self.last_actions = []
        self.max_history = 10

    def act(self, state, epsilon=0.0, env_state=None):
        # 70% random, 30% try to move towards objectives
        if np.random.random() < 0.7 or env_state is None:
            action = np.random.randint(0, self.action_size)
        else:
            # Simple heuristic: if carrying patient, try to move towards rooms
            # if not carrying, try to move towards patients or drugs
            if env_state.carrying_patient is not None:
                # Bias towards moving to room areas
                action = np.random.choice([0, 1, 2, 3], p=[0.3, 0.3, 0.2, 0.2])
            elif not env_state.has_drugs and len(env_state.patients) > 0:
                # Look for patients needing drugs, bias towards pharmacy area
                needs_drugs = any(p['needs_drugs'] for p in env_state.patients)
                if needs_drugs:
                    action = np.random.choice([2, 3, 6, 7], p=[0.3, 0.4, 0.15, 0.15])  # Right/down bias
                else:
                    action = np.random.randint(0, self.action_size)
            else:
                action = np.random.randint(0, self.action_size)

        # Avoid repeating the same action too much
        self.last_actions.append(action)
        if len(self.last_actions) > self.max_history:
            self.last_actions.pop(0)

        if len(self.last_actions) >= 5 and all(a == action for a in self.last_actions[-5:]):
            # If stuck doing same action, choose different one
            action = np.random.randint(0, self.action_size)

        return action

def load_pytorch_model(model_path, state_size, action_size):
    """Load PyTorch DQN model"""
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch not available")

    # Try to load model configuration if available
    config_path = model_path.replace('.pth', '_config.json')
    hidden_size = 256  # default

    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
            hidden_size = config.get('hidden_size', 256)

    # Create and load model
    model = DQNAgent(state_size, action_size, hidden_size)

    try:
        # Load model weights
        checkpoint = torch.load(model_path, map_location='cpu')

        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)
        else:
            model.load_state_dict(checkpoint)

        model.eval()
        print(f"✅ Successfully loaded PyTorch model from {model_path}")

        # Print model info if available in checkpoint
        if isinstance(checkpoint, dict):
            if 'episode' in checkpoint:
                print(f"   Model trained for {checkpoint['episode']} episodes")
            if 'total_reward' in checkpoint:
                print(f"   Best reward: {checkpoint['total_reward']:.2f}")

        return model

    except Exception as e:
        print(f"❌ Error loading PyTorch model: {e}")
        raise

def load_stable_baselines3_model(model_path, env):
    """Load Stable Baselines3 model from .zip file"""
    if not SB3_AVAILABLE:
        raise ImportError("Stable Baselines3 not available. Install with: pip install stable-baselines3")

    try:
        # Try different SB3 algorithms
        algorithms = [DQN, PPO, A2C]

        for Algorithm in algorithms:
            try:
                model = Algorithm.load(model_path, env=env)
                print(f"✅ Successfully loaded {Algorithm.__name__} model from {model_path}")

                class SB3ModelWrapper:
                    def __init__(self, sb3_model):
                        self.model = sb3_model
                        self.action_size = env.action_space.n

                    def act(self, state, epsilon=0.0):
                        if np.random.random() <= epsilon:
                            return np.random.randint(0, self.action_size)

                        action, _ = self.model.predict(state, deterministic=True)
                        return int(action)

                return SB3ModelWrapper(model)

            except Exception as e:
                continue  # Try next algorithm

        raise ValueError("Could not load model with any supported SB3 algorithm")

    except Exception as e:
        print(f"❌ Error loading Stable Baselines3 model: {e}")
        raise

def load_rllib_model(model_path):
    """Load Ray RLlib model from checkpoint"""
    if not RLLIB_AVAILABLE:
        raise ImportError("Ray RLlib not available. Install with: pip install ray[rllib]")

    try:
        # RLlib models are typically saved as checkpoint directories
        algorithm = Algorithm.from_checkpoint(model_path)
        print(f"✅ Successfully loaded RLlib model from {model_path}")

        class RLlibModelWrapper:
            def __init__(self, rllib_model):
                self.model = rllib_model
                self.action_size = None  # Will be set based on environment

            def act(self, state, epsilon=0.0):
                if self.action_size and np.random.random() <= epsilon:
                    return np.random.randint(0, self.action_size)

                action = self.model.compute_single_action(state)
                return int(action)

        return RLlibModelWrapper(algorithm)

    except Exception as e:
        print(f"❌ Error loading RLlib model: {e}")
        raise

def load_tensorflow_model(model_path):
    """Load TensorFlow/Keras model"""
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow not available")

    try:
        model = tf.keras.models.load_model(model_path)
        print(f"✅ Successfully loaded TensorFlow model from {model_path}")

        class TFModelWrapper:
            def __init__(self, tf_model):
                self.model = tf_model
                self.action_size = tf_model.output_shape[-1]

            def act(self, state, epsilon=0.0):
                if np.random.random() <= epsilon:
                    return np.random.randint(0, self.action_size)

                if len(state.shape) == 1:
                    state = state.reshape(1, -1)

                q_values = self.model.predict(state, verbose=0)
                return np.argmax(q_values[0])

        return TFModelWrapper(model)

    except Exception as e:
        print(f"❌ Error loading TensorFlow model: {e}")
        raise

def find_model_files():
    """Find all available model files"""
    model_patterns = [
        "models/*.zip",       # Stable Baselines3 models
        "*.zip",              # Current directory zip models
        "models/*.pth",       # PyTorch models
        "models/*.pt",        # PyTorch models
        "*.pth",              # Current directory PyTorch
        "*.pt",               # Current directory PyTorch
        "models/*.h5",        # TensorFlow models
        "models/*.keras",     # TensorFlow models
        "*.h5",               # Current directory TensorFlow
        "*.keras",            # Current directory TensorFlow
        "saved_models/*/",    # TensorFlow SavedModel format
        "checkpoints/*/",     # RLlib checkpoints
        "ray_results/*/checkpoint_*/"  # RLlib ray results
    ]

    found_models = []
    for pattern in model_patterns:
        found_models.extend(glob.glob(pattern))

    # Remove duplicates and sort
    found_models = sorted(list(set(found_models)))

    return found_models

def select_model_interactive():
    """Interactive model selection"""
    models = find_model_files()

    if not models:
        print("❌ No model files found!")
        print("   Looking for files in: models/*.zip, models/*.pth, models/*.h5, *.zip, etc.")
        print("   Supported formats:")
        print("     - .zip (Stable Baselines3)")
        print("     - .pth/.pt (PyTorch)")
        print("     - .h5/.keras (TensorFlow)")
        print("     - checkpoint directories (RLlib)")
        return None

    print("\n🤖 Available trained models:")
    print("-" * 50)
    for i, model in enumerate(models):
        # Get file size and modification time
        try:
            size = os.path.getsize(model)
            mtime = datetime.fromtimestamp(os.path.getmtime(model))
            size_mb = size / (1024 * 1024)
            print(f"{i+1}. {model}")
            print(f"   Size: {size_mb:.1f} MB, Modified: {mtime.strftime('%Y-%m-%d %H:%M')}")
        except:
            print(f"{i+1}. {model}")

    print(f"{len(models)+1}. Use random agent instead")
    print(f"{len(models)+2}. Use strategic random agent")

    while True:
        try:
            choice = input(f"\nSelect model (1-{len(models)+2}): ").strip()
            choice = int(choice)

            if 1 <= choice <= len(models):
                return models[choice-1]
            elif choice == len(models)+1:
                return "random"
            elif choice == len(models)+2:
                return "strategic"
            else:
                print("Invalid choice. Please try again.")
        except (ValueError, KeyboardInterrupt):
            print("\nExiting...")
            return None

def create_gif(frames, filename, fps=10):
    """Create GIF from frames"""
    if not frames:
        print("No frames to create GIF")
        return

    try:
        imageio.mimsave(filename, frames, fps=fps)
        print(f"🎬 GIF saved as {filename}")
    except Exception as e:
        print(f"❌ Error creating GIF: {e}")

def demonstrate_trained_model(model_path, episodes=5, max_steps_per_episode=500,
                            save_gif=True, epsilon=0.0, render_delay=0.03):
    """Demonstrate trained model performance"""
    print(f"Starting Trained Model Demonstration")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Epsilon (exploration): {epsilon}")
    print(f"Episodes: {episodes}")
    print("=" * 60)

    # Create environment
    env = HospitalNavigationEnv(render_mode='human', max_steps=max_steps_per_episode)

    # Load appropriate agent
    if model_path == "random":
        agent = RandomAgent(env.action_space.n)
        agent_type = "Random"
    elif model_path == "strategic":
        agent = StrategicRandomAgent(env.action_space.n)
        agent_type = "Strategic Random"
    else:
        # Determine model type and load
        if model_path.endswith('.zip'):
            # Try Stable Baselines3 first, then RLlib
            try:
                agent = load_stable_baselines3_model(model_path, env)
                agent_type = "Stable Baselines3"
            except:
                try:
                    agent = load_rllib_model(model_path)
                    agent.action_size = env.action_space.n
                    agent_type = "Ray RLlib"
                except Exception as e:
                    print(f"❌ Failed to load .zip model as SB3 or RLlib: {e}")
                    print("💡 Tip: Make sure you have the correct library installed:")
                    print("   - For SB3: pip install stable-baselines3")
                    print("   - For RLlib: pip install ray[rllib]")
                    raise
        elif model_path.endswith(('.pth', '.pt')):
            state_size = env.observation_space.shape[0]
            action_size = env.action_space.n
            agent = load_pytorch_model(model_path, state_size, action_size)
            agent_type = "PyTorch DQN"
        elif model_path.endswith(('.h5', '.keras')) or os.path.isdir(model_path):
            agent = load_tensorflow_model(model_path)
            agent_type = "TensorFlow DQN"
        elif 'checkpoint' in model_path.lower():
            # Likely RLlib checkpoint directory
            agent = load_rllib_model(model_path)
            agent.action_size = env.action_space.n
            agent_type = "Ray RLlib"
        else:
            raise ValueError(f"Unsupported model format: {model_path}")
            print("💡 Supported formats: .zip (SB3/RLlib), .pth/.pt (PyTorch), .h5/.keras (TensorFlow)")

    # Storage for GIF creation
    frames = []

    # Statistics tracking
    total_rewards = []
    total_patients_saved = []
    total_steps = []

    try:
        for episode in range(episodes):
            print(f"\n🎮 Episode {episode + 1}/{episodes} ({agent_type})")
            print("-" * 40)

            # Reset environment
            obs, _ = env.reset()
            done = False
            step_count = 0
            episode_reward = 0

            # Episode statistics
            patients_at_start = len(env.patients)
            actions_taken = []

            while not done and step_count < max_steps_per_episode:
                # Handle pygame events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        done = True
                        break

                if done:
                    break

                # Get action from agent
                if agent_type == "Strategic Random":
                    action = agent.act(obs, epsilon, env)
                else:
                    action = agent.act(obs, epsilon)

                actions_taken.append(action)

                # Take action
                obs, reward, done, truncated, info = env.step(action)
                episode_reward += reward
                step_count += 1

                # Render environment
                env.render()

                # Capture frame for GIF (every 5th frame to reduce size)
                if save_gif and step_count % 5 == 0:
                    try:
                        screen_array = pygame.surfarray.array3d(env.screen)
                        screen_array = np.transpose(screen_array, (1, 0, 2))
                        frames.append(screen_array)
                    except:
                        pass  # Skip frame if capture fails

                # Print current status
                if step_count % 100 == 0:
                    print(f"  Step {step_count}: Reward = {reward:.2f}, "
                          f"Total = {episode_reward:.2f}, "
                          f"Carrying: {env.carrying_patient is not None}, "
                          f"Has Drugs: {env.has_drugs}, "
                          f"Saved: {env.patients_saved}")

                # Render delay
                time.sleep(render_delay)

            # Episode summary
            patients_saved = env.patients_saved
            total_rewards.append(episode_reward)
            total_patients_saved.append(patients_saved)
            total_steps.append(step_count)

            # Action distribution analysis
            action_counts = np.bincount(actions_taken, minlength=8)
            action_names = ["Up", "Down", "Left", "Right", "Up-Left", "Up-Right", "Down-Left", "Down-Right"]

            print(f"\n  📊 Episode {episode + 1} Summary:")
            print(f"    Total Reward: {episode_reward:.2f}")
            print(f"    Patients Saved: {patients_saved}/{patients_at_start}")
            print(f"    Steps Taken: {step_count}")
            print(f"    Efficiency: {(patients_saved/max(1, step_count))*1000:.2f} patients/1000 steps")
            print(f"    Success Rate: {(patients_saved/max(1, patients_at_start))*100:.1f}%")

            # Show most used actions
            top_actions = np.argsort(action_counts)[-3:][::-1]
            print(f"    Top Actions: " + ", ".join([f"{action_names[i]} ({action_counts[i]})" for i in top_actions]))

            # Wait between episodes
            if episode < episodes - 1:
                print("  ⏳ Waiting 2 seconds before next episode...")
                time.sleep(2)

    except KeyboardInterrupt:
        print("\n⚠️  Demo interrupted by user")

    finally:
        # Create GIF if requested
        if save_gif and frames:
            os.makedirs("demos", exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = Path(model_path).stem if model_path not in ["random", "strategic"] else model_path
            gif_filename = f"demos/hospital_demo_{model_name}_{agent_type.replace(' ', '_')}_{timestamp}.gif"

            print(f"\n🎬 Creating GIF with {len(frames)} frames...")
            create_gif(frames, gif_filename, fps=8)

        # Print overall statistics
        if total_rewards:
            print("\n" + "=" * 60)
            print(f"📈 PERFORMANCE SUMMARY ({agent_type})")
            print("=" * 60)
            print(f"Episodes Completed: {len(total_rewards)}")
            print(f"Average Reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
            print(f"Average Patients Saved: {np.mean(total_patients_saved):.2f} ± {np.std(total_patients_saved):.2f}")
            print(f"Average Steps: {np.mean(total_steps):.1f} ± {np.std(total_steps):.1f}")
            print(f"Best Episode Reward: {max(total_rewards):.2f}")
            print(f"Most Patients Saved: {max(total_patients_saved)}")
            print(f"Best Success Rate: {(max(total_patients_saved)/6)*100:.1f}%")

            # Performance rating
            avg_reward = np.mean(total_rewards)
            avg_patients = np.mean(total_patients_saved)

            if avg_reward > 100 and avg_patients > 3:
                rating = "🌟 Excellent"
            elif avg_reward > 50 and avg_patients > 2:
                rating = "👍 Good"
            elif avg_reward > 0 and avg_patients > 1:
                rating = "📈 Learning"
            else:
                rating = "🔄 Needs Training"

            print(f"Overall Performance: {rating}")

        # Close environment
        env.close()
        print("\n✅ Demo completed!")

def show_environment_layout():
    """Show the environment layout with labels"""
    print("🏥 Displaying Hospital Layout...")

    env = HospitalNavigationEnv(render_mode='human')
    obs, _ = env.reset()

    # Show static layout for 10 seconds
    start_time = time.time()
    while time.time() - start_time < 10:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                break

        env.render()
        time.sleep(0.1)

    env.close()

def main():
    parser = argparse.ArgumentParser(description='Demonstrate Hospital Navigation Environment with Trained Models')
    parser.add_argument('--model', type=str, default=None,
                       help='Path to trained model file (.pth, .h5, .keras, or SavedModel directory)')
    parser.add_argument('--mode', choices=['trained', 'random', 'strategic', 'layout', 'interactive'],
                       default='interactive', help='Demonstration mode')
    parser.add_argument('--episodes', type=int, default=3,
                       help='Number of episodes to run')
    parser.add_argument('--max-steps', type=int, default=500,
                       help='Maximum steps per episode')
    parser.add_argument('--epsilon', type=float, default=0.0,
                       help='Exploration rate (0.0 = no exploration, 0.1 = 10% random actions)')
    parser.add_argument('--no-gif', action='store_true',
                       help='Skip GIF creation')
    parser.add_argument('--render-delay', type=float, default=0.03,
                       help='Delay between renders (seconds)')

    args = parser.parse_args()

    print("🏥 Hospital Navigation Environment - Trained Model Demo")
    print("=" * 55)
    print("\nEnvironment Features:")
    print("- 🏥 Realistic hospital layout with corridors and rooms")
    print("- 🤖 AI agent navigates through corridors only")
    print("- 👥 Patients with different urgency levels (Critical/Moderate/Low)")
    print("- 💊 Drug stations for medication collection")
    print("- 🎯 Multi-objective task: collect, treat, and deliver patients")
    print("- ⚡ Bonus rewards for proper drug administration")
    print("\nControls:")
    print("- Close window or press Ctrl+C to exit")
    print("- Agent uses 8-directional movement")
    print()

    if args.mode == 'layout':
        show_environment_layout()
        return

    # Determine model to use
    model_path = None

    if args.mode == 'interactive':
        model_path = select_model_interactive()
        if model_path is None:
            return
    elif args.mode == 'trained':
        if args.model:
            model_path = args.model
        else:
            models = find_model_files()
            if not models:
                print("❌ No model files found and no --model specified!")
                print("💡 Make sure your trained models are in supported formats:")
                print("   - .zip files (Stable Baselines3, RLlib)")
                print("   - .pth/.pt files (PyTorch)")
                print("   - .h5/.keras files (TensorFlow)")
                return
            model_path = models[0]  # Use first found model
            print(f"🤖 Using model: {model_path}")
    elif args.mode == 'random':
        model_path = 'random'
    elif args.mode == 'strategic':
        model_path = 'strategic'

    if model_path:
        demonstrate_trained_model(
            model_path,
            args.episodes,
            args.max_steps,
            not args.no_gif,
            args.epsilon,
            args.render_delay
        )

if __name__ == "__main__":
    main()
