import sys
import os
from pathlib import Path
os.chdir(Path(__file__).parent)
sys.path.insert(0, str(Path.cwd()))
from env import FlappyBirdEnv

print('Testing environment rendering...')
print('Expected: See pygame window with bird, ground, black background')
env = FlappyBirdEnv(render_mode='human', render_fps=60)
state, _ = env.reset()

print(f'Initial state: bird_y={state[0]:.2f}, velocity={state[1]:.2f}, pipe_x={state[2]:.2f}, gap={state[3]:.2f}')

for step in range(200):
    action = 0
    state, reward, done, _, info = env.step(action)
    if done:
        print(f'Episode ended at step {step}, score: {info["score"]}')
        break

env.close()
print('Done - did you see the game window?')
