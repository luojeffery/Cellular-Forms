from pyglm import glm
from OpenGL.GL import *


def check_compile_errors(shader, shader_type):
	# Check for shader compilation errors
	if shader_type == "PROGRAM":
		success = glGetProgramiv(shader, GL_LINK_STATUS)
		if not success:
			info_log = glGetProgramInfoLog(shader)
			print(f"ERROR::PROGRAM_LINKING_ERROR\n{info_log.decode()}")
	else:
		success = glGetShaderiv(shader, GL_COMPILE_STATUS)
		if not success:
			info_log = glGetShaderInfoLog(shader)
			print(f"ERROR::SHADER_COMPILATION_ERROR of type: {shader_type}\n{info_log.decode()}")
			glDeleteShader(shader)


class Shader:
	def __init__(self, vertex_shader_path, fragment_shader_path, geometry_shader_path=None):
		self.vertex_shader_path = vertex_shader_path
		self.fragment_shader_path = fragment_shader_path
		self.geometry_shader_path = geometry_shader_path
		self.program_id = None
		self.load()

	def load(self):
		# Load and compile shaders, link program

		# Load code content
		with open(self.vertex_shader_path, 'r') as file:
			vertex_shader_code = file.read()
		with open(self.fragment_shader_path, 'r') as file:
			fragment_shader_code = file.read()

		if self.geometry_shader_path:
			with open(self.geometry_shader_path, 'r') as file:
				geometry_shader_code = file.read()

		# Compile shaders
		vertex_shader = glCreateShader(GL_VERTEX_SHADER)
		glShaderSource(vertex_shader, vertex_shader_code)
		glCompileShader(vertex_shader)
		check_compile_errors(vertex_shader, "VERTEX")

		fragment_shader = glCreateShader(GL_FRAGMENT_SHADER)
		glShaderSource(fragment_shader, fragment_shader_code)
		glCompileShader(fragment_shader)
		check_compile_errors(fragment_shader, "FRAGMENT")

		if self.geometry_shader_path:
			geometry_shader = glCreateShader(GL_GEOMETRY_SHADER)
			glShaderSource(geometry_shader, geometry_shader_code)
			glCompileShader(geometry_shader)
			check_compile_errors(geometry_shader, "GEOMETRY")

		# Link program
		self.program_id = glCreateProgram()
		glAttachShader(self.program_id, vertex_shader)
		glAttachShader(self.program_id, fragment_shader)
		if self.geometry_shader_path:
			glAttachShader(self.program_id, geometry_shader)
		glLinkProgram(self.program_id)
		check_compile_errors(self.program_id, "PROGRAM")

		# Delete shaders as they're linked into program now and no longer necessary
		glDeleteShader(vertex_shader)
		glDeleteShader(fragment_shader)
		if self.geometry_shader_path:
			glDeleteShader(geometry_shader)

	def use(self):
		# Use the shader program
		glUseProgram(self.program_id)

	def set_bool(self, name, value):
		glUniform1i(glGetUniformLocation(self.program_id, name), value)

	def set_int(self, name, value):
		glUniform1i(glGetUniformLocation(self.program_id, name), value)

	def set_float(self, name, value):
		glUniform1f(glGetUniformLocation(self.program_id, name), value)

	def setVec2(self, name: str, *args):
		if len(args) == 1 and type(args[0]) == glm.vec2:
			glUniform2fv(glGetUniformLocation(self.program_id, name), 1, glm.value_ptr(args[0]))

		elif len(args) == 2 and all(map(lambda x: type(x) == float, args)):
			glUniform2f(glGetUniformLocation(self.program_id, name), *args)

	def setVec3(self, name: str, *args):
		if len(args) == 1 and type(args[0]) == glm.vec3:
			glUniform3fv(glGetUniformLocation(self.program_id, name), 1, glm.value_ptr(args[0]))

		elif len(args) == 3 and all(map(lambda x: type(x) == float, args)):
			glUniform3f(glGetUniformLocation(self.program_id, name), *args)

	def setVec4(self, name: str, *args) -> None:
		if len(args) == 1 and type(args[0]) == glm.vec4:
			glUniform4fv(glGetUniformLocation(self.program_id, name), 1, glm.value_ptr(args[0]))

		elif len(args) == 3 and all(map(lambda x: type(x) == float, args)):
			glUniform4f(glGetUniformLocation(self.program_id, name), *args)

	def setMat2(self, name: str, mat: glm.mat2) -> None:
		glUniformMatrix2fv(glGetUniformLocation(self.program_id, name), 1, GL_FALSE, glm.value_ptr(mat))

	def setMat3(self, name: str, mat: glm.mat3) -> None:
		glUniformMatrix3fv(glGetUniformLocation(self.program_id, name), 1, GL_FALSE, glm.value_ptr(mat))

	def setMat4(self, name: str, mat: glm.mat4) -> None:
		glUniformMatrix4fv(glGetUniformLocation(self.program_id, name), 1, GL_FALSE, glm.value_ptr(mat))
