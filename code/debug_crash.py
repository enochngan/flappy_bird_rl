"""Debug the crash - run longer and track where it fails"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from env import FlappyBirdEnv

print("Running extended test to find crash point...\n")

try:
    env = FlappyBirdEnv(render_mode="human", render_fps=120)
    state, _ = env.reset()
    
    for step in range(500):
        action = env.action_space.sample()
        state, reward, done, truncated, info = env.step(action)
        
        if step % 50 == 0:
            print(f"Step {step}: OK - Score: {info['score']}")
        
        if done:
            print(f"Step {step}: CRASHED - Score: {info['score']} (resetting)")
            state, _ = env.reset()
    
    env.close()
    print("\n✓ Test completed successfully!")
    
except Exception as e:
    print(f"\n✗ CRASH at some point: {e}")
    import traceback
    traceback.print_exc()
