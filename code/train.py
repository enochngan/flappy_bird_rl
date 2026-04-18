"""
Training script for Q-Learning Flappy Bird agent.
Supports expert pretraining to solve the sparse-reward cold-start problem.
"""
import sys
import os
from pathlib import Path
import numpy as np

os.chdir(Path(__file__).parent)
sys.path.insert(0, str(Path(__file__).parent))

from env import FlappyBirdEnv
from q_agent import QLearningAgent


def expert_action(state):
    """
    Rule-based expert policy.

    state = [bird_y_norm, bird_vel_norm, pipe_x_norm, gap_rel]
      gap_rel < 0  →  gap is above the bird
      gap_rel > 0  →  gap is below the bird
      pipe_x > 0.88 →  no pipe visible, hover at center
    """
    bird_y, bird_vel, pipe_x, gap_rel = state

    # No pipe visible (passed or not yet spawned) → hover near screen center
    if pipe_x > 0.88 or pipe_x < 0.04:
        if bird_y > 0.55 and bird_vel > -0.25:
            return 1   # drifting low, flap
        if bird_vel > 0.35:
            return 1   # falling too fast
        return 0

    # Pipe approaching: steer toward gap
    if gap_rel < -0.08:
        # Gap is above — flap if not already rising fast enough
        return 1 if bird_vel > -0.38 else 0
    if gap_rel > 0.08:
        # Gap is below — fall, but catch overshoot
        return 1 if bird_vel > 0.45 else 0
    # Near gap center — gentle control
    return 1 if bird_vel > 0.25 else 0                  # let gravity do the work


def pretrain(env, agent, num_episodes=300, noise=0.05):
    """
    Run the expert policy for `num_episodes` episodes, storing
    transitions into the Q-table via normal Q-learning updates.

    `noise` = probability of ignoring the expert and acting randomly,
    so the agent sees states slightly off the expert trajectory too.
    """
    print(f"\nPretraining with expert policy ({num_episodes} episodes, noise={noise})...")
    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        step = 0
        score = 0
        total_reward = 0.0

        while not done and step < 1000:
            if np.random.random() < noise:
                action = np.random.randint(0, 2)
            else:
                action = expert_action(state)

            next_state, reward, done, _, info = env.step(action)
            agent.update(state, action, reward, next_state, done)
            total_reward += reward
            score = info['score']
            state = next_state
            step += 1

        agent.record_episode(score, total_reward)

        if (episode + 1) % 50 == 0:
            recent = agent.episode_scores[-50:]
            print(f"  Pretrain {episode+1:3d}/{num_episodes} | "
                  f"Score: {score} | Avg: {np.mean(recent):.1f} | "
                  f"Q-table: {len(agent.q_table)}")

    print(f"Pretraining done. Q-table size: {len(agent.q_table)}\n")


def train(num_episodes=3000, render=False, save_path="flappy_bird_agent.pkl",
          resume=False, pretrain_episodes=400):
    print("=" * 60)
    print(f"Q-LEARNING TRAINING - {num_episodes} Episodes")
    if pretrain_episodes:
        print(f"Expert pretraining: {pretrain_episodes} episodes first")
    print("=" * 60)

    env = FlappyBirdEnv(render_mode="human" if render else None)

    # Load existing agent only if explicitly resuming
    agent = None
    if resume and Path(save_path).exists():
        try:
            agent = QLearningAgent()
            agent.load(save_path)
            print("Resumed from existing agent")
        except Exception as e:
            print(f"Could not load agent: {e} — starting fresh")
            agent = None

    if agent is None:
        agent = QLearningAgent(
            num_bins=12,
            learning_rate=0.15,
            discount_factor=0.95,
            epsilon=1.0,
            epsilon_decay=0.9985,   # slower decay → more exploration
            epsilon_min=0.05,
        )

    # --- Expert pretraining ---
    if pretrain_episodes and not resume:
        # Temporarily disable agent exploration; expert provides the actions
        agent.epsilon = 0.0
        pretrain(env, agent, num_episodes=pretrain_episodes, noise=0.05)
        # After pretraining, start Q-learning with moderate exploration
        agent.epsilon = 0.4

    # --- Q-Learning loop ---
    print(f"Starting Q-learning from epsilon={agent.epsilon:.2f}...")
    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0.0
        score = 0
        done = False
        step = 0

        while not done and step < 1000:
            action = agent.choose_action(state)
            next_state, reward, done, _, info = env.step(action)
            agent.update(state, action, reward, next_state, done)
            total_reward += reward
            score = info['score']
            state = next_state
            step += 1

        agent.record_episode(score, total_reward)
        agent.decay_epsilon()

        if (episode + 1) % 20 == 0 or episode == 0:
            recent_scores = agent.episode_scores[-20:]
            print(f"Episode {episode+1:4d}/{num_episodes} | "
                  f"Score: {score:3d} | "
                  f"Avg (last 20): {np.mean(recent_scores):5.1f} | "
                  f"Epsilon: {agent.epsilon:.3f} | "
                  f"Q-table: {len(agent.q_table)}")

    env.close()
    agent.save(save_path)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    stats = agent.get_stats()
    print(f"Total episodes:          {stats['total_episodes']}")
    print(f"Avg score (all):         {stats['avg_score_all']:.1f}")
    print(f"Avg score (recent 100):  {stats['avg_score_recent']:.1f}")
    print(f"Max score:               {stats['max_score']}")
    print(f"Q-table size:            {stats['q_table_size']}")
    print("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train Q-Learning Flappy Bird agent")
    parser.add_argument("--episodes", type=int, default=3000)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--save", type=str, default="flappy_bird_agent.pkl")
    parser.add_argument("--resume", action="store_true", help="Resume from saved agent")
    parser.add_argument("--pretrain", type=int, default=400,
                        help="Expert pretraining episodes (0 to skip)")

    args = parser.parse_args()
    train(
        num_episodes=args.episodes,
        render=args.render,
        save_path=args.save,
        resume=args.resume,
        pretrain_episodes=args.pretrain,
    )
