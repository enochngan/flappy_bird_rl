"""
Watch the trained DQN agent play Flappy Bird.
"""
import sys
import os
from pathlib import Path
import numpy as np

os.chdir(Path(__file__).parent)
sys.path.insert(0, str(Path(__file__).parent))

from env import FlappyBirdEnv
from dqn_agent import DQNAgent


def play(agent_path="flappy_bird_dqn.pt", num_episodes=3, fps=90):
    print("=" * 60)
    print("FLAPPY BIRD — DQN AGENT")
    print("=" * 60)

    agent = DQNAgent()
    agent.load(agent_path)
    agent.set_exploit_mode()

    env = FlappyBirdEnv(render_mode="human", render_fps=fps)
    scores = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        step = 0
        score = 0
        action_count = [0, 0]

        print(f"\nEpisode {episode + 1}/{num_episodes}:")

        while not done and step < 2000:
            action = agent.choose_action(state)
            action_count[action] += 1
            next_state, reward, done, _, info = env.step(action)
            score = info["score"]
            state = next_state
            step += 1

        scores.append(score)
        flap_pct = 100 * action_count[1] / max(1, sum(action_count))
        print(f"  Score: {score} | Steps: {step} | Flap%: {flap_pct:.1f}%")

    env.close()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Episodes: {num_episodes}")
    print(f"Avg score: {np.mean(scores):.1f}")
    print(f"Max score: {max(scores)}")
    print(f"Min score: {min(scores)}")
    print("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Play Flappy Bird with DQN agent")
    parser.add_argument("--agent", type=str, default="flappy_bird_dqn.pt")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--fps", type=int, default=90)

    args = parser.parse_args()
    play(agent_path=args.agent, num_episodes=args.episodes, fps=args.fps)
