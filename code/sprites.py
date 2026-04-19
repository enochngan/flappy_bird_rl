import pygame 
from settings import *
from random import choice, randint

class BG(pygame.sprite.Sprite):
	def __init__(self,groups,scale_factor):
		super().__init__(groups)
		bg_image = pygame.image.load('../graphics/flappy_bird_sprites_fixed/background_day.png').convert()

		# Stretch background to fill the window dimensions for clean tiling
		tile_w = WINDOW_WIDTH
		tile_h = WINDOW_HEIGHT
		full_sized_image = pygame.transform.scale(bg_image,(tile_w, tile_h))

		self.image = pygame.Surface((tile_w * 2, tile_h))
		self.image.blit(full_sized_image,(0,0))
		self.image.blit(full_sized_image,(tile_w,0))

		self.rect = self.image.get_rect(topleft = (0,0))
		self.pos = pygame.math.Vector2(self.rect.topleft)

	def update(self,dt):
		self.pos.x -= 180 * dt
		if self.rect.centerx <= 0:
			self.pos.x = 0
		self.rect.x = round(self.pos.x)

class Ground(pygame.sprite.Sprite):
	def __init__(self,groups,scale_factor):
		super().__init__(groups)
		self.sprite_type = 'ground'
		
		# image
		ground_surf = pygame.image.load('../graphics/environment/ground.png').convert_alpha()
		self.image = pygame.transform.scale(ground_surf,pygame.math.Vector2(ground_surf.get_size()) * scale_factor)
		
		# position
		self.rect = self.image.get_rect(bottomleft = (0,WINDOW_HEIGHT))
		self.pos = pygame.math.Vector2(self.rect.topleft)

		# mask
		self.mask = pygame.mask.from_surface(self.image)

	def update(self,dt):
		self.pos.x -= 220 * dt
		if self.rect.centerx <= 0:
			self.pos.x = 0

		self.rect.x = round(self.pos.x)

class Plane(pygame.sprite.Sprite):
	def __init__(self,groups,scale_factor):
		super().__init__(groups)

		# image 
		self.import_frames(scale_factor)
		self.frame_index = 0
		self.image = self.frames[self.frame_index]

		# rect
		self.rect = self.image.get_rect(midleft = (WINDOW_WIDTH / 20,WINDOW_HEIGHT / 2))
		self.pos = pygame.math.Vector2(self.rect.topleft)

		# movement
		self.gravity = 1000
		self.direction = 0

		# mask
		self.mask = pygame.mask.from_surface(self.image)

		# sound
		try:
			self.jump_sound = pygame.mixer.Sound('../sounds/jump.wav')
			self.jump_sound.set_volume(0.3)
		except:
			self.jump_sound = None

	def import_frames(self,scale_factor):
		self.frames = []
		surf = pygame.image.load('../graphics/flappy_bird_sprites_fixed/109939-logo-pic-bird-flappy-free-transparent-image-hq.png').convert_alpha()
		scaled_surface = pygame.transform.scale(surf, (100, 65))
		for _ in range(3):
			self.frames.append(scaled_surface)

	def apply_gravity(self,dt):
		self.direction += self.gravity * dt
		self.pos.y += self.direction * dt
		self.rect.y = round(self.pos.y)

	def jump(self):
		if self.jump_sound:
			self.jump_sound.play()
		self.direction = -400

	def animate(self,dt):
		self.frame_index += 10 * dt
		if self.frame_index >= len(self.frames):
			self.frame_index = 0
		self.image = self.frames[int(self.frame_index)]

	def rotate(self):
		rotated_plane = pygame.transform.rotozoom(self.image,-self.direction * 0.06,1)
		self.image = rotated_plane
		self.mask = pygame.mask.from_surface(self.image)

	def update(self,dt):
		self.apply_gravity(dt)
		self.animate(dt)
		self.rotate()

class Pipe(pygame.sprite.Sprite):
	"""Bottom pipe. Always spawns a matching top pipe (PipePartner)."""
	def __init__(self, groups, scale_factor, x_start=None):
		super().__init__(groups)
		self.sprite_type = 'obstacle'
		self.counts_for_score = True   # only bottom pipe scores

		gap_y = randint(int(WINDOW_HEIGHT * 0.25), int(WINDOW_HEIGHT * 0.75))
		self.gap_height = 250
		self.gap_y = gap_y

		surf = pygame.image.load('../graphics/flappy_bird_sprites_fixed/pipe_green.png').convert_alpha()
		pipe_w = 80
		# Height: from gap edge all the way past the bottom of the screen
		pipe_top_y = gap_y + self.gap_height // 2
		pipe_h = WINDOW_HEIGHT - pipe_top_y + 60
		self.image = pygame.transform.scale(surf, (pipe_w, pipe_h))
		self.image = pygame.transform.flip(self.image, False, True)

		x = x_start if x_start is not None else WINDOW_WIDTH + 50
		self.rect = self.image.get_rect(midtop=(x, pipe_top_y))
		self.pos = pygame.math.Vector2(self.rect.topleft)
		self.mask = pygame.mask.from_surface(self.image)

		# Always spawn the matching top pipe
		PipePartner(groups, scale_factor, gap_y, self.gap_height, x_start=x)

	def update(self, dt):
		self.pos.x -= 240 * dt
		self.rect.x = round(self.pos.x)
		if self.rect.right <= -100:
			self.kill()


class PipePartner(pygame.sprite.Sprite):
	"""Top pipe, always paired with a Pipe."""
	def __init__(self, groups, scale_factor, gap_y, gap_height, x_start=None):
		super().__init__(groups)
		self.sprite_type = 'obstacle'
		self.counts_for_score = False  # top pipe never scores

		surf = pygame.image.load('../graphics/flappy_bird_sprites_fixed/pipe_green.png').convert_alpha()
		pipe_w = 80
		# Height: from gap edge all the way past the top of the screen
		pipe_bottom_y = gap_y - gap_height // 2
		pipe_h = pipe_bottom_y + 60
		self.image = pygame.transform.scale(surf, (pipe_w, pipe_h))

		x = x_start if x_start is not None else WINDOW_WIDTH + 50
		self.rect = self.image.get_rect(midbottom=(x, pipe_bottom_y))
		self.pos = pygame.math.Vector2(self.rect.topleft)
		self.mask = pygame.mask.from_surface(self.image)

		self.gap_y = gap_y
		self.gap_height = gap_height

	def update(self, dt):
		self.pos.x -= 240 * dt
		self.rect.x = round(self.pos.x)
		if self.rect.right <= -100:
			self.kill()


class Obstacle(pygame.sprite.Sprite):
	def __init__(self,groups,scale_factor):
		super().__init__(groups)
		self.sprite_type = 'obstacle'

		orientation = choice(('up','down'))
		surf = pygame.image.load(f'../graphics/obstacles/{choice((0,1))}.png').convert_alpha()
		self.image = pygame.transform.scale(surf,pygame.math.Vector2(surf.get_size()) * scale_factor)
		
		x = WINDOW_WIDTH + randint(40,100)

		if orientation == 'up':
			y = WINDOW_HEIGHT + randint(10,50)
			self.rect = self.image.get_rect(midbottom = (x,y))
		else:
			y = randint(-50,-10)
			self.image = pygame.transform.flip(self.image,False,True)
			self.rect = self.image.get_rect(midtop = (x,y))

		self.pos = pygame.math.Vector2(self.rect.topleft)

		# mask
		self.mask = pygame.mask.from_surface(self.image)

	def update(self,dt):
		self.pos.x -= 400 * dt
		self.rect.x = round(self.pos.x)
		if self.rect.right <= -100:
			self.kill()
