"""
Play with trained Q-Learning agent
Watch the agent play Flappy Bird intelligently
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


def play(agent_path="flappy_bird_agent.pkl", num_episodes=3, render=True, fps=90):
    """
    Play episodes with a trained agent.
    
    Args:
        agent_path: Path to saved agent
        num_episodes: Number of episodes to play
        render: Whether to visualize
        fps: Frames per second for rendering
    """
    print("=" * 60)
    print("FLAPPY BIRD - Q-LEARNING AGENT")
    print("=" * 60)
    
    # Load agent
    agent = QLearningAgent()
    agent.load(agent_path)
    agent.set_exploit_mode()  # No exploration, just exploit learned policy
    
    # Initialize environment
    env = FlappyBirdEnv(render_mode="human" if render else None, render_fps=fps)
    
    scores = []
    
    # Play episodes
    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        step = 0
        score = 0
        action_count = [0, 0]  # [idle, flap]
        
        print(f"\nEpisode {episode + 1}/{num_episodes}:")
        
        while not done and step < 1000:
            # Agent chooses action (greedy - no exploration)
            action = agent.choose_action(state)
            action_count[action] += 1
            
            # Environment step
            next_state, reward, done, truncated, info = env.step(action)
            
            score = info['score']
            state = next_state
            step += 1
        
        scores.append(score)
        action_pct = 100 * action_count[1] / (action_count[0] + action_count[1])
        print(f"  Score: {score} | Steps: {step} | Flap %: {action_pct:.1f}%")
    
    env.close()
    
    # Print summary
    print("\n" + "=" * 60)
    print("PLAY SUMMARY")
    print("=" * 60)
    print(f"Episodes played: {num_episodes}")
    print(f"Avg score: {np.mean(scores):.1f}")
    print(f"Max score: {max(scores)}")
    print(f"Min score: {min(scores)}")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Play Flappy Bird with trained Q-Learning agent")
    parser.add_argument("--agent", type=str, default="flappy_bird_agent.pkl", help="Agent path")
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes")
    parser.add_argument("--render", action="store_true", default=True, help="Render game")
    parser.add_argument("--fps", type=int, default=90, help="Frames per second")
    
    args = parser.parse_args()
    
    play(
        agent_path=args.agent,
        num_episodes=args.episodes,
        render=args.render,
        fps=args.fps
    )
