"""
Better debug - trace the obstacle spawning logic with step-count method
"""
import sys
import os
from pathlib import Path
import pygame
os.chdir(Path(__file__).parent)
sys.path.insert(0, str(Path(__file__).parent))

from env import FlappyBirdEnv

env = FlappyBirdEnv(render_mode=None)
state, _ = env.reset()

print("=" * 60)
print("DEBUG: TRACING OBSTACLE SPAWNING (STEP-COUNT METHOD)")
print("=" * 60)

for step in range(200):
    action = 0
    
    state, reward, done, truncated, info = env.step(action)
    
    obstacles_count = len([s for s in env.collision_sprites if hasattr(s, 'sprite_type') and s.sprite_type == 'obstacle'])
    
    if step % 10 == 0 or obstacles_count > 0 or done:
        print(f"Step {step:3d}: steps_since_last={env.steps_since_last_obstacle:2d} | obstacles={obstacles_count} | active={env.active}")
    
    if done:
        print(f"  → Episode ended at step {step} with {obstacles_count} obstacles visible")
        state, _ = env.reset()

env.close()
print("Done!")
