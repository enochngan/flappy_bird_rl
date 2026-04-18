"""
Gymnasium environment wrapper for Flappy Bird - Q-Learning Ready
"""
import gymnasium as gym
import numpy as np
import pygame
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "code"))

from settings import WINDOW_WIDTH, WINDOW_HEIGHT, FRAMERATE
from sprites import BG, Ground, Plane, Pipe


class FlappyBirdEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": FRAMERATE}

    def __init__(self, render_mode=None, render_fps=120):
        super().__init__()

        self.render_mode = render_mode
        self.render_fps = render_fps
        self.clock = None

        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(4,), dtype=np.float32
        )

        if not pygame.get_init():
            pygame.init()

        self.display_surface = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Flappy Bird - Q-Learning")
        self.clock = pygame.time.Clock()

        self.all_sprites = None
        self.collision_sprites = None
        self.plane = None
        self.scale_factor = None
        self.music = None
        self.score = 0
        self.active = False
        self.frame_count = 0
        self.steps_since_last_obstacle = 0
        self.pipes_cleared = 0        # how many pipe pairs the bird has passed
        self._pipe_xs_seen = set()    # tracks pipe x-positions already counted

        self._setup_game()

    def _setup_game(self):
        if self.all_sprites is not None:
            self.all_sprites.empty()
            self.collision_sprites.empty()

        self.all_sprites = pygame.sprite.Group()
        self.collision_sprites = pygame.sprite.Group()

        bg_height = pygame.image.load('../graphics/environment/background.png').get_height()
        self.scale_factor = WINDOW_HEIGHT / bg_height

        BG(self.all_sprites, self.scale_factor)
        Ground([self.all_sprites, self.collision_sprites], self.scale_factor)
        self.plane = Plane(self.all_sprites, self.scale_factor / 1.7)

        # Spawn first pipe already on-screen so the agent encounters it
        # before dying (~step 65 arrival vs ~step 106 death without flapping)
        Pipe([self.all_sprites, self.collision_sprites], self.scale_factor * 1.1,
             x_start=WINDOW_WIDTH // 2)
        self.steps_since_last_obstacle = 0
        self.pipes_cleared = 0
        self._pipe_xs_seen = set()

        if self.music is None:
            try:
                self.music = pygame.mixer.Sound('../sounds/music.wav')
            except Exception:
                self.music = None
        if self.music:
            self.music.play(loops=-1)

        self.active = True
        self.score = 0
        self.frame_count = 0

    def _get_state(self):
        if not self.active or self.plane is None:
            return np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)

        bird_y_norm = np.clip(self.plane.rect.centery / WINDOW_HEIGHT, 0, 1)
        bird_vel_norm = np.clip(self.plane.direction / 600, -1, 1)

        next_obstacle_x = 1.0
        gap_rel = 0.0

        obstacles = [
            s for s in self.collision_sprites
            if hasattr(s, 'sprite_type') and s.sprite_type == 'obstacle'
        ]

        next_obstacle = None
        if obstacles:
            ahead = [o for o in obstacles if o.rect.right > self.plane.rect.centerx]
            if ahead:
                next_obstacle = min(ahead, key=lambda o: o.rect.left)
            else:
                next_obstacle = min(obstacles, key=lambda o: o.rect.left)

        if next_obstacle:
            next_obstacle_x = np.clip(next_obstacle.rect.left / WINDOW_WIDTH, 0, 1)
            gap_y = self._estimate_gap_center(next_obstacle)
            gap_rel = np.clip(
                (gap_y - self.plane.rect.centery) / (WINDOW_HEIGHT / 2), -1, 1
            )

        return np.array([bird_y_norm, bird_vel_norm, next_obstacle_x, gap_rel], dtype=np.float32)

    def _estimate_gap_center(self, obstacle):
        obstacles_at_x = [
            s for s in self.collision_sprites
            if hasattr(s, 'sprite_type') and s.sprite_type == 'obstacle'
            and abs(s.rect.left - obstacle.rect.left) < 20
        ]

        if len(obstacles_at_x) >= 2:
            sorted_obs = sorted(obstacles_at_x, key=lambda o: o.rect.centery)
            top_obs = sorted_obs[0]
            bottom_obs = sorted_obs[-1]
            return (top_obs.rect.bottom + bottom_obs.rect.top) / 2

        return obstacle.rect.centery

    def _count_cleared_pipes(self):
        """Return +1 for each pipe pair the bird just cleared this step."""
        bonus = 0
        obstacles = [
            s for s in self.collision_sprites
            if hasattr(s, 'sprite_type') and s.sprite_type == 'obstacle'
        ]
        for obs in obstacles:
            key = id(obs)
            if obs.rect.right < self.plane.rect.centerx and key not in self._pipe_xs_seen:
                self._pipe_xs_seen.add(key)
                bonus += 1
        return bonus

    def _get_reward(self, pipes_passed):
        if not self.active:
            return -10.0  # crash penalty

        reward = 0.1  # survival reward

        # Big reward for clearing a pipe pair
        reward += pipes_passed * 5.0

        # Shape reward: penalize being far from the gap center
        obstacles = [
            s for s in self.collision_sprites
            if hasattr(s, 'sprite_type') and s.sprite_type == 'obstacle'
        ]
        ahead = [o for o in obstacles if o.rect.right > self.plane.rect.centerx]
        if ahead and self.plane:
            nearest = min(ahead, key=lambda o: o.rect.left)
            gap_y = self._estimate_gap_center(nearest)
            dist = abs(self.plane.rect.centery - gap_y) / WINDOW_HEIGHT
            reward -= dist * 0.5  # up to -0.5 per step for being far from gap

        return reward

    def step(self, action):
        if not self.active:
            return self._get_state(), 0, True, False, {}

        if action == 1:
            self.plane.jump()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        if self.active and self.steps_since_last_obstacle >= 120:
            Pipe([self.all_sprites, self.collision_sprites], self.scale_factor * 1.1)
            self.steps_since_last_obstacle = 0

        self.steps_since_last_obstacle += 1

        dt = 1.0 / self.render_fps
        self.all_sprites.update(dt)
        if self.render_mode == "human":
            self.display_surface.fill('black')
            self.all_sprites.draw(self.display_surface)

        # Check how many pipes were cleared before collision check kills the plane
        pipes_passed = self._count_cleared_pipes()

        # Check collisions
        hit = pygame.sprite.spritecollide(
            self.plane, self.collision_sprites, False, pygame.sprite.collide_mask
        )
        if hit or self.plane.rect.top <= 0:
            for sprite in self.collision_sprites.sprites():
                if hasattr(sprite, 'sprite_type') and sprite.sprite_type == 'obstacle':
                    sprite.kill()
            self.active = False
            self.plane.kill()

        # Score = pipes cleared (meaningful regardless of rendering speed)
        if pipes_passed:
            self.pipes_cleared += pipes_passed
        self.score = self.pipes_cleared
        self.frame_count += 1

        reward = self._get_reward(pipes_passed)
        state = self._get_state()
        done = not self.active

        if self.render_mode == "human":
            pygame.display.update()
            self.clock.tick(self.render_fps)

        return state, reward, done, False, {"score": self.score, "frame": self.frame_count}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        for sprite in self.collision_sprites.sprites():
            if hasattr(sprite, 'sprite_type') and sprite.sprite_type == 'obstacle':
                sprite.kill()
        self._setup_game()
        return self._get_state(), {}

    def render(self):
        if self.render_mode == "human":
            pygame.display.update()
            self.clock.tick(self.render_fps)

    def close(self):
        pygame.quit()
