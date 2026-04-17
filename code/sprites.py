import pygame 
from settings import *
from random import choice, randint

class BG(pygame.sprite.Sprite):
	def __init__(self,groups,scale_factor):
		super().__init__(groups)
		bg_image = pygame.image.load('../graphics/environment/background.bmp').convert()

		full_height = bg_image.get_height() * scale_factor
		full_width = bg_image.get_width() * scale_factor
		full_sized_image = pygame.transform.scale(bg_image,(full_width,full_height))
		
		self.image = pygame.Surface((full_width * 2,full_height))
		self.image.blit(full_sized_image,(0,0))
		self.image.blit(full_sized_image,(full_width,0))

		self.rect = self.image.get_rect(topleft = (0,0))
		self.pos = pygame.math.Vector2(self.rect.topleft)

	def update(self,dt):
		self.pos.x -= 300 * dt
		if self.rect.centerx <= 0:
			self.pos.x = 0
		self.rect.x = round(self.pos.x)

class Ground(pygame.sprite.Sprite):
	def __init__(self,groups,scale_factor):
		super().__init__(groups)
		self.sprite_type = 'ground'
		
		# image
		ground_surf = pygame.image.load('../graphics/environment/ground.bmp').convert_alpha()
		self.image = pygame.transform.scale(ground_surf,pygame.math.Vector2(ground_surf.get_size()) * scale_factor)
		
		# position
		self.rect = self.image.get_rect(bottomleft = (0,WINDOW_HEIGHT))
		self.pos = pygame.math.Vector2(self.rect.topleft)

		# mask
		self.mask = pygame.mask.from_surface(self.image)

	def update(self,dt):
		self.pos.x -= 360 * dt
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
		self.gravity = 600
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
		for i in range(3):
			surf = pygame.image.load(f'../graphics/plane/red{i}.bmp').convert_alpha()
			scaled_surface = pygame.transform.scale(surf,pygame.math.Vector2(surf.get_size())* scale_factor)
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
	def __init__(self,groups,scale_factor):
		super().__init__(groups)
		self.sprite_type = 'obstacle'
		
		# Get pipe image
		surf = pygame.image.load(f'../graphics/obstacles/{choice((0,1))}.bmp').convert_alpha()
		
		# Random gap position (height where bird can pass through)
		gap_y = randint(int(WINDOW_HEIGHT * 0.3), int(WINDOW_HEIGHT * 0.7))
		self.gap_height = 150  # Fixed gap size
		self.gap_y = gap_y
		
		# Determine if this is top or bottom pipe
		self.is_top = choice([True, False])
		
		# Create scaled image
		self.image = pygame.transform.scale(surf, pygame.math.Vector2(surf.get_size()) * scale_factor)
		
		# Position (starting off-screen to the right)
		x = WINDOW_WIDTH + 50
		
		if self.is_top:
			# Top pipe: bottom of pipe is at gap_y - gap_height/2
			y = gap_y - self.gap_height // 2
			self.rect = self.image.get_rect(midbottom = (x, y))
		else:
			# Bottom pipe: top of pipe is at gap_y + gap_height/2
			y = gap_y + self.gap_height // 2
			self.image = pygame.transform.flip(self.image, False, True)
			self.rect = self.image.get_rect(midtop = (x, y))
		
		self.pos = pygame.math.Vector2(self.rect.topleft)
		
		# Mask for collision detection
		self.mask = pygame.mask.from_surface(self.image)
		
		# Create the matching pipe (top if this is bottom, bottom if this is top)
		# by creating a new Pipe in the same groups
		if not self.is_top:  # Only create partner if this is a bottom pipe
			PipePartner(groups, scale_factor, gap_y, self.gap_height)
	
	def update(self, dt):
		self.pos.x -= 400 * dt
		self.rect.x = round(self.pos.x)
		if self.rect.right <= -100:
			self.kill()


class PipePartner(pygame.sprite.Sprite):
	"""Partner pipe (top) created automatically when a bottom pipe is created"""
	def __init__(self, groups, scale_factor, gap_y, gap_height):
		super().__init__(groups)
		self.sprite_type = 'obstacle'
		self.is_partner = True
		
		# Get pipe image
		surf = pygame.image.load(f'../graphics/obstacles/{choice((0,1))}.bmp').convert_alpha()
		
		# Create scaled image
		self.image = pygame.transform.scale(surf, pygame.math.Vector2(surf.get_size()) * scale_factor)
		
		# Position (starting off-screen to the right)
		x = WINDOW_WIDTH + 50
		y = gap_y - gap_height // 2
		self.rect = self.image.get_rect(midbottom = (x, y))
		
		self.pos = pygame.math.Vector2(self.rect.topleft)
		
		# Mask for collision detection
		self.mask = pygame.mask.from_surface(self.image)
		
		self.gap_y = gap_y
		self.gap_height = gap_height
	
	def update(self, dt):
		self.pos.x -= 400 * dt
		self.rect.x = round(self.pos.x)
		if self.rect.right <= -100:
			self.kill()


class Obstacle(pygame.sprite.Sprite):
	def __init__(self,groups,scale_factor):
		super().__init__(groups)
		self.sprite_type = 'obstacle'

		orientation = choice(('up','down'))
		surf = pygame.image.load(f'../graphics/obstacles/{choice((0,1))}.bmp').convert_alpha()
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
