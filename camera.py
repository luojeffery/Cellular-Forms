from enum import IntEnum
import math
from pyglm import *
from OpenGL.GL import *
from pyglm import glm


class Camera_Movement(IntEnum):
	FORWARD = 0
	BACKWARD = 1
	LEFT = 2
	RIGHT = 3
	UP = 4
	DOWN = 5


# Default camera values
YAW = -90.0
PITCH = 0.0
SPEED = 5.0
SENSITIVITY = 0.1
ZOOM = 45.0


class Camera:
	def __init__(self, position=glm.vec3(0.0, 0.0, 0.0), up=glm.vec3(0.0, 1.0, 0.0), yaw=YAW, pitch=PITCH):
		self.Position = position
		self.WorldUp = up
		self.Yaw = yaw
		self.Pitch = pitch
		self.Front = glm.vec3(0.0, 0.0, -1.0)
		self.MovementSpeed = SPEED
		self.MouseSensitivity = SENSITIVITY
		self.Zoom = ZOOM

		self.update_camera_vectors()

	@classmethod
	def from_scalars(cls, pos_x, pos_y, pos_z, up_x, up_y, up_z, yaw, pitch):
		return cls(glm.vec3(pos_x, pos_y, pos_z), glm.vec3(up_x, up_y, up_z), yaw, pitch)

	def get_view_matrix(self):
		return glm.lookAt(self.Position, self.Position + self.Front, self.Up)

	def process_keyboard(self, direction, delta_time):
		velocity = self.MovementSpeed * delta_time
		if direction == Camera_Movement.FORWARD:
			self.Position += self.Front * velocity
		if direction == Camera_Movement.BACKWARD:
			self.Position -= self.Front * velocity
		if direction == Camera_Movement.LEFT:
			self.Position -= self.Right * velocity
		if direction == Camera_Movement.RIGHT:
			self.Position += self.Right * velocity
		if direction == Camera_Movement.UP:
			self.Position += self.Up * velocity
		if direction == Camera_Movement.DOWN:
			self.Position -= self.Up * velocity

	def process_mouse_movement(self, xoffset, yoffset, constrain_pitch=True):
		xoffset *= self.MouseSensitivity
		yoffset *= self.MouseSensitivity

		self.Yaw += xoffset
		self.Pitch += yoffset

		if constrain_pitch:
			self.Pitch = max(min(self.Pitch, 89.0), -89.0)

		self.update_camera_vectors()

	def process_mouse_scroll(self, yoffset):
		self.Zoom -= yoffset
		self.Zoom = max(min(self.Zoom, 45.0), 1.0)

	def update_camera_vectors(self):
		front = glm.vec3(
			math.cos(glm.radians(self.Yaw)) * math.cos(glm.radians(self.Pitch)),
			math.sin(glm.radians(self.Pitch)),
			math.sin(glm.radians(self.Yaw)) * math.cos(glm.radians(self.Pitch))
		)
		self.Front = glm.normalize(front)
		self.Right = glm.normalize(glm.cross(self.Front, self.WorldUp))
		self.Up = glm.normalize(glm.cross(self.Right, self.Front))