# Flappy Bird RL

A Flappy Bird environment built with Pygame and Gymnasium, used to compare reinforcement learning approaches: tabular Q-Learning and Deep Q-Networks (DQN) with behavioral cloning.

---

## Installation

Requires Python 3.10+.

```bash
pip install pygame gymnasium torch numpy
```

All scripts must be run from inside the `code/` directory:

```bash
cd code
```

---

## Playing the Game Yourself

```bash
python3 human_play.py
```

- **Space** or **click** to flap
- The game waits for your first input before starting
- Press any key to restart after a game over
- Score = number of pipes cleared

---

## Tabular Q-Learning Agent

### Train

```bash
python3 train.py --episodes 3000
```

Runs 400 episodes of expert pretraining first, then 3000 episodes of Q-learning. Saves the agent to `flappy_bird_agent.pkl`.

| Flag | Default | Description |
|---|---|---|
| `--episodes` | 3000 | Number of Q-learning episodes |
| `--pretrain` | 400 | Expert pretraining episodes (0 to skip) |
| `--save` | `flappy_bird_agent.pkl` | Output file |
| `--resume` | off | Continue training from existing save |
| `--render` | off | Show the game window while training |

### Watch the Agent Play

```bash
python3 play.py --episodes 5
```

| Flag | Default | Description |
|---|---|---|
| `--episodes` | 3 | Number of episodes to play |
| `--agent` | `flappy_bird_agent.pkl` | Agent file to load |
| `--fps` | 90 | Rendering speed |

---

## DQN Agent (Behavioral Cloning)

### Train

```bash
python3 train_dqn.py --episodes 2000
```

Runs 300 episodes of behavioral cloning first (supervised imitation of the expert), then 2000 episodes of DQN fine-tuning. Saves the agent to `flappy_bird_dqn.pt`.

| Flag | Default | Description |
|---|---|---|
| `--episodes` | 2000 | Number of DQN episodes |
| `--pretrain` | 300 | Expert episodes for behavioral cloning (0 to skip) |
| `--save` | `flappy_bird_dqn.pt` | Output file |
| `--resume` | off | Continue training from existing checkpoint |
| `--render` | off | Show the game window while training |

### Watch the Agent Play

```bash
python3 play_dqn.py --episodes 5
```

| Flag | Default | Description |
|---|---|---|
| `--episodes` | 3 | Number of episodes to play |
| `--agent` | `flappy_bird_dqn.pt` | Agent file to load |
| `--fps` | 90 | Rendering speed |

---

## Project Structure

```
flappy_bird_rl/
├── code/
│   ├── env.py            # Gymnasium environment wrapper
│   ├── sprites.py        # Pygame sprites (bird, pipes, background)
│   ├── settings.py       # Window size and framerate constants
│   ├── q_agent.py        # Tabular Q-learning agent
│   ├── dqn_agent.py      # DQN agent (neural network + replay buffer)
│   ├── train.py          # Train tabular Q-learning agent
│   ├── train_dqn.py      # Train DQN agent
│   ├── play.py           # Watch tabular agent play
│   ├── play_dqn.py       # Watch DQN agent play
│   ├── human_play.py     # Play the game yourself
│   ├── flappy_bird_agent.pkl   # Saved tabular agent
│   └── flappy_bird_dqn.pt      # Saved DQN agent
├── graphics/
│   └── flappy_bird_sprites_fixed/   # Game sprites (bird, pipes, background)
└── sounds/
    ├── jump.wav
    └── music.wav
```


## AI Usage