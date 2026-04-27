"""
Play Flappy Bird yourself — Space or click to flap.
"""
import sys
import os
from pathlib import Path

os.chdir(Path(__file__).parent)
sys.path.insert(0, str(Path(__file__).parent))

import pygame
from settings import WINDOW_WIDTH, WINDOW_HEIGHT, FRAMERATE
from sprites import BG, Ground, Plane, Pipe


def run():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Flappy Bird — Human Mode")
    clock = pygame.time.Clock()

    font = pygame.font.SysFont(None, 48)
    small_font = pygame.font.SysFont(None, 32)

    def setup():
        all_sprites = pygame.sprite.Group()
        collision_sprites = pygame.sprite.Group()
        scale_factor = WINDOW_HEIGHT / 480
        BG(all_sprites, scale_factor)
        plane = Plane(all_sprites, scale_factor / 1.7)
        return all_sprites, collision_sprites, plane, scale_factor

    all_sprites, collision_sprites, plane, scale_factor = setup()

    score = 0
    best_score = 0
    steps_since_pipe = 0
    pipes_seen = set()
    alive = True
    started = False  # wait for first flap before gravity kicks in

    while True:
        dt = clock.tick(FRAMERATE) / 1000

        flap = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type in (pygame.KEYDOWN, pygame.MOUSEBUTTONDOWN):
                if not alive:
                    # restart
                    all_sprites, collision_sprites, plane, scale_factor = setup()
                    score = 0
                    steps_since_pipe = 0
                    pipes_seen = set()
                    alive = True
                    started = False
                else:
                    flap = True
                    started = True

        if alive and started:
            if flap:
                plane.jump()

            # Spawn pipes
            if steps_since_pipe >= 200:
                Pipe([all_sprites, collision_sprites], scale_factor * 1.1)
                steps_since_pipe = 0
            steps_since_pipe += 1

            # Update
            all_sprites.update(dt)

            # Collision
            bird_hitbox = plane.rect.inflate(-20, -15)
            hit = pygame.sprite.spritecollide(
                plane, collision_sprites, False,
                lambda p, obs: bird_hitbox.colliderect(obs.rect)
            )
            if hit or plane.rect.top <= 0 or plane.rect.bottom >= WINDOW_HEIGHT:
                alive = False
                best_score = max(best_score, score)

            # Count pipes the bird has passed
            for s in collision_sprites:
                if getattr(s, 'counts_for_score', False):
                    if s.rect.right < plane.rect.centerx and s.pipe_id not in pipes_seen:
                        pipes_seen.add(s.pipe_id)
                        score += 1

        elif alive and not started:
            # Only scroll BG/ground before first flap, freeze plane
            for s in all_sprites:
                if not isinstance(s, Plane):
                    s.update(dt)

        screen.fill('black')
        all_sprites.draw(screen)

        # HUD
        score_surf = font.render(str(score), True, 'white')
        screen.blit(score_surf, score_surf.get_rect(midtop=(WINDOW_WIDTH // 2, 20)))

        if not started and alive:
            msg = small_font.render("Press Space or Click to start", True, 'white')
            screen.blit(msg, msg.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 + 80)))

        if not alive:
            over = font.render("Game Over", True, 'red')
            best = small_font.render(f"Score: {score}  Best: {best_score}", True, 'white')
            restart = small_font.render("Press any key to restart", True, 'white')
            screen.blit(over, over.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 - 40)))
            screen.blit(best, best.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 + 10)))
            screen.blit(restart, restart.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 + 55)))

        pygame.display.update()


if __name__ == "__main__":
    run()
