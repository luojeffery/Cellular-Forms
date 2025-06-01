from pyglm import glm
import shader
from dataclasses import dataclass
from OpenGL.GL import *
import numpy as np



@dataclass
class Texture:
	id: int
	type: str
	path: str


class Mesh:
	def __init__(self, vertices, indices, textures):
		self.vertices = vertices
		self.indices = np.array(indices)
		self.textures: list[Texture] = textures
		self.VAO = None
		self.VBO = None
		self.EBO = None
		self.setup_mesh()

	def setup_mesh(self):
		# Create buffers and assign them to the VAO
		self.VAO = glGenVertexArrays(1)
		self.VBO = glGenBuffers(1)
		self.EBO = glGenBuffers(1)

		glBindVertexArray(self.VAO)

		glBindBuffer(GL_ARRAY_BUFFER, self.VBO)
		glBufferData(GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, GL_STATIC_DRAW)

		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.EBO)
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, self.indices.nbytes, self.indices, GL_STATIC_DRAW)

		# Vertex positions
		glEnableVertexAttribArray(0)
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * ctypes.sizeof(GLfloat), ctypes.c_void_p(0))

		# Vertex normals
		glEnableVertexAttribArray(1)
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * ctypes.sizeof(GLfloat), ctypes.c_void_p(glm.sizeof(glm.vec3)))

		# Vertex texture coordinates
		glEnableVertexAttribArray(2)
		glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * ctypes.sizeof(GLfloat), ctypes.c_void_p(glm.sizeof(glm.vec3) * 2))

		glBindBuffer(GL_ARRAY_BUFFER, 0)
		glBindVertexArray(0)

	def draw(self, shader):
		diffuse_num = 1
		specular_num = 1
		normal_num = 1
		height_num = 1
		for i in range(len(self.textures)):
			number = ""
			glActiveTexture(GL_TEXTURE0 + i)
			texture_name = self.textures[i].type
			if texture_name == "texture_diffuse":
				diffuse_num += 1
				number = diffuse_num
			elif texture_name == "texture_specular":
				specular_num += 1
				number = specular_num
			elif texture_name == "texture_normal":
				normal_num += 1
				number = normal_num
			elif texture_name == "texture_height":
				height_num += 1
				number = height_num

			glUniform1i(glGetUniformLocation(shader.program_id, f"{texture_name}{number}"), i)
			glBindTexture(GL_TEXTURE_2D, self.textures[i].id)

		glBindVertexArray(self.VAO)
		glDrawElements(GL_TRIANGLES, self.indices.size, GL_UNSIGNED_INT, None)
		glBindVertexArray(0)

		glActiveTexture(GL_TEXTURE0)

	def draw_instanced(self, instance_count):
		glBindVertexArray(self.VAO)
		glDrawElementsInstanced(GL_TRIANGLES, self.indices.size, GL_UNSIGNED_INT, None, instance_count)
		glBindVertexArray(0)
