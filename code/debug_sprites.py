"""
Debug visualization - see what sprites are active
"""
import sys
import os
from pathlib import Path
os.chdir(Path(__file__).parent)
sys.path.insert(0, str(Path(__file__).parent))

from env import FlappyBirdEnv

env = FlappyBirdEnv(render_mode="human", render_fps=60)
state, _ = env.reset()

print("=" * 60)
print("DEBUG: CHECKING SPRITE GENERATION")
print("=" * 60)

for step in range(300):
    action = 0 if step % 20 < 10 else 1  # Alternate flap/idle
    
    state, reward, done, truncated, info = env.step(action)
    
    if step % 20 == 0:
        # Count sprites
        all_count = len(env.all_sprites.sprites())
        collision_count = len(env.collision_sprites.sprites())
        obstacle_count = len([s for s in env.collision_sprites if hasattr(s, 'sprite_type') and s.sprite_type == 'obstacle'])
        
        print(f"Step {step:3d}: All={all_count} | Collision={collision_count} | Obstacles={obstacle_count} | Score={info['score']} | Bird Y={info['bird_y']:.0f}")
        print(f"        State: {state}")
    
    if done:
        print(f"Step {step}: Episode ended")
        state, _ = env.reset()

env.close()
