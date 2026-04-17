"""
Gymnasium environment wrapper for Flappy Bird - Q-Learning Ready
"""
import gymnasium as gym
import numpy as np
import pygame
import sys
from pathlib import Path
import time

# Add code directory to path
sys.path.insert(0, str(Path(__file__).parent / "code"))

from settings import WINDOW_WIDTH, WINDOW_HEIGHT, FRAMERATE
from sprites import BG, Ground, Plane, Pipe


class FlappyBirdEnv(gym.Env):
    """Gymnasium environment for Flappy Bird with Q-Learning interface."""
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": FRAMERATE}
    
    def __init__(self, render_mode=None, render_fps=120):
        """Initialize the Flappy Bird environment."""
        super().__init__()
        
        self.render_mode = render_mode
        self.render_fps = render_fps
        self.clock = None
        
        # Action space: 0 = do nothing, 1 = flap
        self.action_space = gym.spaces.Discrete(2)
        
        # State space: [bird_y, bird_velocity, next_pipe_x, gap_center]
        # Normalized to [0, 1]
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(4,), dtype=np.float32
        )
        
        # Initialize Pygame
        if not pygame.get_init():
            pygame.init()
        
        self.display_surface = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Flappy Bird - Q-Learning")
        self.clock = pygame.time.Clock()
        
        # Game state
        self.all_sprites = None
        self.collision_sprites = None
        self.plane = None
        self.scale_factor = None
        self.music = None
        self.score = 0
        self.start_offset = 0
        self.active = False
        self.frame_count = 0
        self.steps_since_last_obstacle = 0  # Spawn obstacles every N steps
        
        # For getting next obstacle info
        self.obstacles = []
        
        self._setup_game()
    
    def _setup_game(self):
        """Setup game sprites and environment."""
        # Clear existing sprites
        if self.all_sprites is not None:
            self.all_sprites.empty()
            self.collision_sprites.empty()
        
        self.all_sprites = pygame.sprite.Group()
        self.collision_sprites = pygame.sprite.Group()
        
        # Load and scale background (use relative paths - expecting to run from code/)
        bg_height = pygame.image.load('../graphics/environment/background.bmp').get_height()
        self.scale_factor = WINDOW_HEIGHT / bg_height
        
        # Create sprites
        BG(self.all_sprites, self.scale_factor)
        Ground([self.all_sprites, self.collision_sprites], self.scale_factor)
        self.plane = Plane(self.all_sprites, self.scale_factor / 1.7)
        
        # Obstacles spawn every 30 steps at 60 FPS = ~0.5 seconds
        self.steps_since_last_obstacle = 0
        
        # Music
        if self.music is None:
            try:
                self.music = pygame.mixer.Sound('../sounds/music.wav')
            except:
                self.music = None
        if self.music:
            self.music.play(loops=-1)
        
        self.active = True
        self.score = 0
        self.start_offset = pygame.time.get_ticks()
        self.frame_count = 0
    
    def _get_state(self):
        """Extract state from game: [bird_y, bird_velocity, next_pipe_x, gap_position_relative_to_bird]"""
        if not self.active or self.plane is None:
            return np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
        
        # Bird Y position (normalized 0-1)
        bird_y_norm = self.plane.rect.centery / WINDOW_HEIGHT
        
        # Bird velocity (normalized, clamped to [-1, 1])
        bird_vel_norm = np.clip(self.plane.direction / 600, -1, 1)
        
        # Find next obstacle in front of bird
        next_obstacle = None
        next_obstacle_x = 1.0  # Default: far away
        gap_rel = 0.0  # Default: gap is at bird level
        
        obstacles = [s for s in self.collision_sprites if hasattr(s, 'sprite_type') and s.sprite_type == 'obstacle']
        
        if obstacles:
            # Find the first obstacle to the right of the bird
            for obs in obstacles:
                if obs.rect.left > self.plane.rect.centerx:
                    next_obstacle = obs
                    break
            
            if next_obstacle is None and obstacles:
                # No obstacle ahead, use the first one
                next_obstacle = obstacles[0]
        
        if next_obstacle:
            # Obstacle X position (normalized 0-1, clamped)
            next_obstacle_x = np.clip(next_obstacle.rect.left / WINDOW_WIDTH, 0, 1)
            
            # Gap position relative to bird
            # Tells agent: is gap above (negative) or below (positive)?
            gap_y = self._estimate_gap_center(next_obstacle)
            gap_rel = np.clip((gap_y - self.plane.rect.centery) / (WINDOW_HEIGHT/2), -1, 1)
        
        return np.array([bird_y_norm, bird_vel_norm, next_obstacle_x, gap_rel], dtype=np.float32)
    
    def _estimate_gap_center(self, obstacle):
        """Estimate the gap center between obstacle pair."""
        # The gap is approximately between the two obstacles
        # Look for obstacles at the same X position
        obstacles_at_x = [
            s for s in self.collision_sprites 
            if hasattr(s, 'sprite_type') and s.sprite_type == 'obstacle'
            and abs(s.rect.left - obstacle.rect.left) < 10
        ]
        
        if len(obstacles_at_x) >= 2:
            # Find top and bottom obstacles
            heights = [(s.rect.bottom if s.rect.bottom < WINDOW_HEIGHT/2 else 0, s) for s in obstacles_at_x]
            heights = sorted(heights, key=lambda x: x[0])
            
            if len(heights) >= 2:
                top_obs = heights[0][1]
                bottom_obs = heights[1][1]
                
                # Gap center is between them
                gap_center = (top_obs.rect.bottom + bottom_obs.rect.top) / 2
                return gap_center
        
        # Default: use obstacle center
        return obstacle.rect.centery
    
    def _get_reward(self):
        """Calculate reward for this step."""
        if not self.active:
            return -5.0  # Crash penalty
        
        # Base reward for surviving
        reward = 0.1
        
        # Bonus for navigating (score increments)
        if self.score > 0 and self.score % 100 == 0:
            reward += 1.0  # Major milestone bonus
        
        return reward
    
    def step(self, action):
        """Execute one step of the environment."""
        if not self.active:
            return self._get_state(), 0, True, False, {}
        
        # Execute action
        if action == 1:  # Flap
            self.plane.jump()
        
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        
        # Manually spawn obstacles based on step count (every 30 steps = ~500ms at 60 FPS)
        if self.active and self.steps_since_last_obstacle >= 30:
            Pipe([self.all_sprites, self.collision_sprites], self.scale_factor * 1.1)
            self.steps_since_last_obstacle = 0
        
        self.steps_since_last_obstacle += 1
        
        # Update game logic
        dt = (1.0 / self.render_fps)
        self.display_surface.fill('black')
        self.all_sprites.update(dt)
        self.all_sprites.draw(self.display_surface)
        
        # Check collisions
        if pygame.sprite.spritecollide(self.plane, self.collision_sprites, False, pygame.sprite.collide_mask) \
           or self.plane.rect.top <= 0:
            for sprite in self.collision_sprites.sprites():
                if hasattr(sprite, 'sprite_type') and sprite.sprite_type == 'obstacle':
                    sprite.kill()
            self.active = False
            self.plane.kill()
        
        # Update score
        if self.active:
            self.score = (pygame.time.get_ticks() - self.start_offset) // 100
        
        # Get reward and state
        reward = self._get_reward()
        state = self._get_state()
        done = not self.active
        
        # Render
        if self.render_mode == "human":
            pygame.display.update()
            self.clock.tick(self.render_fps)
        
        self.frame_count += 1
        
        return state, reward, done, False, {"score": self.score, "bird_y": self.plane.rect.centery if self.plane else 0}
    
    def reset(self, seed=None, options=None):
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        # Clear obstacles
        for sprite in self.collision_sprites.sprites():
            if hasattr(sprite, 'sprite_type') and sprite.sprite_type == 'obstacle':
                sprite.kill()
        
        self._setup_game()
        
        return self._get_state(), {}
    
    def render(self):
        """Render the environment."""
        if self.render_mode == "human":
            pygame.display.update()
            self.clock.tick(self.render_fps)
    
    def close(self):
        """Close the environment."""
        pygame.quit()


if __name__ == "__main__":
    # Quick test
    env = FlappyBirdEnv(render_mode="human")
    state, _ = env.reset()
    print(f"Initial state: {state}")
    
    for _ in range(100):
        action = env.action_space.sample()  # Random action
        state, reward, done, truncated, info = env.step(action)
        print(f"Step: action={action}, reward={reward:.2f}, score={info['score']}")
        
        if done:
            state, _ = env.reset()
    
    env.close()
