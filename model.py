import os
from PIL import Image
import trimesh
from mesh import *


class Model:
	def __init__(self, filepath):
		self.meshes = []
		self.textures_loaded = []
		self.directory = os.path.dirname(filepath)
		self.load_model(filepath)

	def load_model(self, filepath):
		scene = trimesh.load_scene(filepath)
		meshes = scene.geometry.values()
		for mesh in meshes:
			self.meshes.append(self.process_mesh(mesh))

	def process_mesh(self, mesh):
		# process vertices
		positions = np.array(mesh.vertices)
		normals = np.array(mesh.vertex_normals)
		if len(mesh.visual.uv) > 0:
			uvs = np.array(mesh.visual.uv)
		else:
			uvs = np.zeros(positions.shape[0], 2)

		vertices = np.hstack((positions, normals, uvs)).astype(np.float32).flatten()
		# process faces (flatten them)
		indices = np.array(mesh.faces).astype(np.uint32).flatten()

		# process materials
		textures = []
		if mesh.visual.material:
			material = mesh.visual.material
			if "file_path" in material.image.info:
				path = material.image.info["file_path"]  # this is always diffuse
				type = "texture_diffuse"
				id = self.texture_from_file(path)
				textures.append(Texture(id, type, path))

		return Mesh(vertices, indices, textures)

	def texture_from_file(self, path) -> int:
		try:
			img = Image.open(path)
		except IOError:
			print(f"[Error] Could not load texture at path: {path}")
			return 0

		mode = img.mode
		if mode == 'L':
			format_gl = GL_RED
		elif mode == 'RGB':
			format_gl = GL_RGB
		elif mode == 'RGBA':
			format_gl = GL_RGBA
		else:
			img = img.convert('RGBA')
			format_gl = GL_RGBA

		# Pillow gives top-left origin; if you need bottom-left you can flip:
		img_data = np.array(img)  # shape (H, W, channels)
		height, width = img_data.shape[:2]

		# upload to GPU
		texture_id = glGenTextures(1)
		glBindTexture(GL_TEXTURE_2D, texture_id)
		glTexImage2D(GL_TEXTURE_2D, 0, format_gl, width, height, 0, format_gl, GL_UNSIGNED_BYTE, img_data)
		glGenerateMipmap(GL_TEXTURE_2D)

		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

		glBindTexture(GL_TEXTURE_2D, 0)

		return texture_id

	def draw(self, shader):
		for mesh in self.meshes:
			mesh.draw(shader)

	def draw_instanced(self, instances):
		for mesh in self.meshes:
			mesh.draw_instanced(instances)


if __name__ == "__main__":
	mesh = Model(r"objects\backpack\backpack.obj")