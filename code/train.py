"""
Training script for Q-Learning Flappy Bird agent
Trains the agent through experience
"""
import sys
import os
from pathlib import Path
import numpy as np

# Change to script directory for proper relative paths
os.chdir(Path(__file__).parent)
sys.path.insert(0, str(Path(__file__).parent))

from env import FlappyBirdEnv
from q_agent import QLearningAgent


def train(num_episodes=200, render=False, save_path="flappy_bird_agent.pkl", resume=True):
    """
    Train the Q-Learning agent.
    
    Args:
        num_episodes: Number of episodes to train
        render: Whether to render during training
        save_path: Where to save the trained agent
        resume: Whether to continue from saved agent (if exists)
    """
    print("=" * 60)
    print(f"Q-LEARNING TRAINING - {num_episodes} Episodes")
    print("=" * 60)
    
    # Initialize
    env = FlappyBirdEnv(render_mode="human" if render else None)
    
    # Try to load existing agent if resume=True
    agent = None
    if resume and Path(save_path).exists():
        try:
            agent = QLearningAgent()
            agent.load(save_path)
            print(f"✓ Resumed from existing agent")
        except Exception as e:
            print(f"⚠ Could not load agent: {e}")
            print(f"  Starting fresh instead...")
            agent = None
    
    # Create fresh agent if not resuming
    if agent is None:
        agent = QLearningAgent(
            num_bins=10,
            learning_rate=0.1,
            discount_factor=0.95,
            epsilon=1.0,
            epsilon_decay=0.995,
            epsilon_min=0.05
        )
        print(f"Starting fresh training...")
    
    # Training loop
    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0.0
        score = 0
        done = False
        step = 0
        
        while not done and step < 1000:
            # Agent chooses action
            action = agent.choose_action(state)
            
            # Environment step
            next_state, reward, done, truncated, info = env.step(action)
            
            # Q-Learning update
            agent.update(state, action, reward, next_state, done)
            
            # Track stats
            total_reward += reward
            score = info['score']
            state = next_state
            step += 1
        
        # End of episode
        agent.record_episode(score, total_reward)
        agent.decay_epsilon()
        
        # Print progress
        if (episode + 1) % 20 == 0 or episode == 0:
            recent_scores = agent.episode_scores[-20:]
            avg_score = np.mean(recent_scores)
            print(f"Episode {episode+1:3d}/{num_episodes} | "
                  f"Score: {score:3d} | "
                  f"Avg (last 20): {avg_score:6.1f} | "
                  f"Epsilon: {agent.epsilon:.3f} | "
                  f"Q-table: {len(agent.q_table)}")
    
    env.close()
    
    # Save agent
    agent.save(save_path)
    
    # Print final stats
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    stats = agent.get_stats()
    print(f"Total episodes: {stats['total_episodes']}")
    print(f"Avg score (all): {stats['avg_score_all']:.1f}")
    print(f"Avg score (recent 100): {stats['avg_score_recent']:.1f}")
    print(f"Max score: {stats['max_score']}")
    print(f"Q-table size: {stats['q_table_size']}")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Q-Learning Flappy Bird agent")
    parser.add_argument("--episodes", type=int, default=200, help="Number of episodes to train")
    parser.add_argument("--render", action="store_true", help="Render during training")
    parser.add_argument("--save", type=str, default="flappy_bird_agent.pkl", help="Save path")
    parser.add_argument("--fresh", action="store_true", help="Start fresh (don't resume from saved agent)")
    
    args = parser.parse_args()
    
    train(
        num_episodes=args.episodes,
        render=args.render,
        save_path=args.save,
        resume=not args.fresh
    )
