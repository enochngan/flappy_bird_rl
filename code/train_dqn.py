"""
DQN training script for Flappy Bird.
Uses behavioral cloning for warm-start, then DQN fine-tuning.
"""
import sys
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

os.chdir(Path(__file__).parent)
sys.path.insert(0, str(Path(__file__).parent))

from env import FlappyBirdEnv
from dqn_agent import DQNAgent


# ── Expert policy ─────────────────────────────────────────────────────────────

def expert_action(state):
    bird_y, bird_vel, pipe_x, gap_rel = state

    if pipe_x > 0.88 or pipe_x < 0.04:
        if bird_y > 0.55 and bird_vel > -0.25:
            return 1
        if bird_vel > 0.35:
            return 1
        return 0

    if gap_rel < -0.08:
        return 1 if bird_vel > -0.38 else 0
    if gap_rel > 0.08:
        return 1 if bird_vel > 0.45 else 0
    return 1 if bird_vel > 0.25 else 0


# ── Behavioral cloning warm-start ─────────────────────────────────────────────

def behavioral_clone(env, agent, num_episodes=300, noise=0.05, bc_epochs=30):
    """
    Collect expert rollouts, then directly train the network with
    cross-entropy to mimic the expert (supervised classification).
    This avoids the bootstrapping problem of TD learning on sparse rewards.
    """
    print(f"\nCollecting expert data ({num_episodes} episodes)...")
    all_states = []
    all_actions = []
    ep_scores = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        step = 0
        score = 0

        while not done and step < 1000:
            if np.random.random() < noise:
                action = np.random.randint(0, 2)
            else:
                action = expert_action(state)

            next_state, reward, done, _, info = env.step(action)
            all_states.append(state.copy())
            all_actions.append(action)
            # Also fill replay buffer for later DQN updates
            agent.buffer.push(state, action, reward, next_state, done)
            score = info["score"]
            state = next_state
            step += 1

        ep_scores.append(score)

    print(f"  Expert avg score: {np.mean(ep_scores):.1f}  max: {max(ep_scores)}")
    print(f"  Dataset size: {len(all_states)} transitions, buffer: {len(agent.buffer)}")

    # ── Supervised training (behavioral cloning) ──────────────────────────────
    print(f"  Training network via behavioral cloning ({bc_epochs} epochs)...")
    states_t  = torch.tensor(np.array(all_states,  dtype=np.float32), device=agent.device)
    actions_t = torch.tensor(np.array(all_actions, dtype=np.int64),   device=agent.device)

    dataset = torch.utils.data.TensorDataset(states_t, actions_t)
    loader  = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

    for epoch in range(bc_epochs):
        total_loss = 0.0
        correct = 0
        for s_batch, a_batch in loader:
            logits = agent.policy_net(s_batch)
            loss = F.cross_entropy(logits, a_batch)
            agent.optimizer.zero_grad()
            loss.backward()
            agent.optimizer.step()
            total_loss += loss.item()
            correct += (logits.argmax(dim=1) == a_batch).sum().item()

        if (epoch + 1) % 10 == 0:
            acc = 100 * correct / len(all_states)
            print(f"    BC epoch {epoch+1:2d}/{bc_epochs} | "
                  f"loss: {total_loss/len(loader):.4f} | acc: {acc:.1f}%")

    # Sync target network with newly trained policy
    agent.target_net.load_state_dict(agent.policy_net.state_dict())
    print("  Behavioral cloning done.\n")


# ── Main training loop ────────────────────────────────────────────────────────

def train(
    num_episodes=2000,
    render=False,
    save_path="flappy_bird_dqn.pt",
    resume=False,
    pretrain_episodes=300,
):
    print("=" * 60)
    print(f"DQN TRAINING - {num_episodes} Episodes")
    if pretrain_episodes:
        print(f"Behavioral cloning warm-start: {pretrain_episodes} expert episodes")
    print("=" * 60)

    env = FlappyBirdEnv(render_mode="human" if render else None)

    agent = DQNAgent(
        lr=5e-4,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.997,
        epsilon_min=0.01,
        batch_size=64,
        target_update_freq=500,
        buffer_capacity=50_000,
    )

    if resume and Path(save_path).exists():
        try:
            agent.load(save_path)
            print("Resumed from existing checkpoint")
        except Exception as e:
            print(f"Could not load checkpoint: {e} — starting fresh")

    # ── Behavioral cloning warm-start ──────────────────────────────────────────
    if pretrain_episodes and not resume:
        behavioral_clone(env, agent, num_episodes=pretrain_episodes,
                         noise=0.05, bc_epochs=30)
        # After cloning, explore but don't erase what was learned too quickly
        agent.epsilon = 0.3

    # ── DQN fine-tuning loop ───────────────────────────────────────────────────
    print(f"Starting DQN from epsilon={agent.epsilon:.2f}...")
    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0.0
        score = 0
        done = False
        step = 0
        ep_loss = []

        while not done and step < 2000:
            action = agent.choose_action(state)
            next_state, reward, done, _, info = env.step(action)

            agent.store(state, action, reward, next_state, done)
            loss = agent.update()
            if loss is not None:
                ep_loss.append(loss)

            total_reward += reward
            score = info["score"]
            state = next_state
            step += 1

        agent.record_episode(score, total_reward)
        agent.decay_epsilon()

        if (episode + 1) % 20 == 0 or episode == 0:
            recent_scores = agent.episode_scores[-20:]
            avg_loss = np.mean(ep_loss) if ep_loss else 0.0
            print(f"Episode {episode+1:4d}/{num_episodes} | "
                  f"Score: {score:3d} | "
                  f"Avg20: {np.mean(recent_scores):5.1f} | "
                  f"Eps: {agent.epsilon:.3f} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"Buf: {len(agent.buffer)}")

        if (episode + 1) % 200 == 0:
            agent.save(save_path)

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
    print("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train DQN Flappy Bird agent")
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--save", type=str, default="flappy_bird_dqn.pt")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--pretrain", type=int, default=300,
                        help="Expert episodes for behavioral cloning (0 to skip)")

    args = parser.parse_args()
    train(
        num_episodes=args.episodes,
        render=args.render,
        save_path=args.save,
        resume=args.resume,
        pretrain_episodes=args.pretrain,
    )
