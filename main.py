import sys

import glfw
import imgui
import numpy as np
from OpenGL.GL import *
from pyglm import *
import random
import os

from pyglm import glm
from imgui.integrations.glfw import GlfwRenderer
from shader import Shader
from model import Model
from camera import Camera, Camera_Movement

# Settings
SCR_WIDTH = 1000
SCR_HEIGHT = 750
use_ssao = True
only_ao = True

# Camera
camera = Camera(glm.vec3(0.0, 0.0, 5.0))
last_x = SCR_WIDTH / 2
last_y = SCR_HEIGHT / 2
first_mouse = True
cursor_disabled = True
e_pressed = False

# Timing
delta_time = 0.0
last_frame = 0.0


def impl_glfw_init():
	window_name = "SSAO-Cellular Forms"

	if not glfw.init():
		print("Could not initialize OpenGL context")
		sys.exit(1)

	glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
	glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
	glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

	glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL_TRUE)

	window = glfw.create_window(SCR_WIDTH, SCR_HEIGHT, window_name, None, None)
	glfw.make_context_current(window)

	if not window:
		glfw.terminate()
		print("Could not initialize Window")
		sys.exit(1)

	return window


def our_lerp(a, b, f):
	return a + f * (b - a)


def main():
	global delta_time, last_frame, use_ssao, only_ao

	imgui.create_context()
	window = impl_glfw_init()
	impl = GlfwRenderer(window)

	glfw.set_framebuffer_size_callback(window, framebuffer_size_callback)
	glfw.set_cursor_pos_callback(window, mouse_callback)
	glfw.set_scroll_callback(window, scroll_callback)
	glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_DISABLED)

	# Configure OpenGL
	glEnable(GL_DEPTH_TEST)

	# Load shaders
	shader_geometry_pass = Shader("ssao_geometry.vs", "ssao_geometry.fs")
	shader_lighting_pass = Shader("ssao.vs", "ssao_lighting.fs")
	shader_ssao = Shader("ssao.vs", "ssao.fs")
	shader_ssao_blur = Shader("ssao.vs", "ssao_blur.fs")

	# Load models
	backpack = Model("objects/backpack/backpack.obj")

	# Configure G-Buffer
	g_buffer = glGenFramebuffers(1)
	glBindFramebuffer(GL_FRAMEBUFFER, g_buffer)

	# Position color buffer
	g_position = glGenTextures(1)
	glBindTexture(GL_TEXTURE_2D, g_position)
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, SCR_WIDTH, SCR_HEIGHT, 0, GL_RGBA, GL_FLOAT, None)
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, g_position, 0)

	# Normal color buffer
	g_normal = glGenTextures(1)
	glBindTexture(GL_TEXTURE_2D, g_normal)
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, SCR_WIDTH, SCR_HEIGHT, 0, GL_RGBA, GL_FLOAT, None)
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, g_normal, 0)

	# Albedo color buffer
	g_albedo = glGenTextures(1)
	glBindTexture(GL_TEXTURE_2D, g_albedo)
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, SCR_WIDTH, SCR_HEIGHT, 0, GL_RGBA, GL_UNSIGNED_BYTE, None)
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, GL_TEXTURE_2D, g_albedo, 0)

	# Attachments
	attachments = [GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2]
	glDrawBuffers(3, attachments)

	# Depth buffer
	rbo_depth = glGenRenderbuffers(1)
	glBindRenderbuffer(GL_RENDERBUFFER, rbo_depth)
	glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, SCR_WIDTH, SCR_HEIGHT)
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rbo_depth)

	if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
		print("Framebuffer not complete!")
	glBindFramebuffer(GL_FRAMEBUFFER, 0)

	# SSAO Framebuffers
	ssao_fbo = glGenFramebuffers(1)
	ssao_blur_fbo = glGenFramebuffers(1)

	# SSAO color buffer
	glBindFramebuffer(GL_FRAMEBUFFER, ssao_fbo)
	ssao_color_buffer = glGenTextures(1)
	glBindTexture(GL_TEXTURE_2D, ssao_color_buffer)
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, SCR_WIDTH, SCR_HEIGHT, 0, GL_RED, GL_FLOAT, None)
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, ssao_color_buffer, 0)

	# SSAO blur buffer
	glBindFramebuffer(GL_FRAMEBUFFER, ssao_blur_fbo)
	ssao_color_buffer_blur = glGenTextures(1)
	glBindTexture(GL_TEXTURE_2D, ssao_color_buffer_blur)
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, SCR_WIDTH, SCR_HEIGHT, 0, GL_RED, GL_FLOAT, None)
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, ssao_color_buffer_blur, 0)
	glBindFramebuffer(GL_FRAMEBUFFER, 0)

	# Generate sample kernel
	ssao_kernel = []
	for _ in range(64):
		sample = glm.normalize(glm.vec3(
			random.uniform(-1, 1),
			random.uniform(-1, 1),
			random.uniform(0, 1)
		))
		scale = random.uniform(0, 1)
		scale *= our_lerp(0.1, 1.0, scale * scale)
		sample *= scale
		ssao_kernel.append(sample)

	# Generate noise texture
	noise_texture = glGenTextures(1)
	glBindTexture(GL_TEXTURE_2D, noise_texture)
	noise_data = []
	for _ in range(16):
		noise_data.extend([random.uniform(-1, 1), random.uniform(-1, 1), 0.0])
	noise_data = np.array(noise_data, dtype=np.float32)
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, 4, 4, 0, GL_RGB, GL_FLOAT, noise_data)
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)

	# Lighting info
	light_pos = glm.vec3(2.0, 4.0, -2.0)
	light_color = glm.vec3(0.2, 0.2, 0.7)

	# Shader config
	shader_lighting_pass.use()
	shader_lighting_pass.set_int("gPosition", 0)
	shader_lighting_pass.set_int("gNormal", 1)
	shader_lighting_pass.set_int("gAlbedo", 2)
	shader_lighting_pass.set_int("ssao", 3)

	shader_ssao.use()
	shader_ssao.set_int("gPosition", 0)
	shader_ssao.set_int("gNormal", 1)
	shader_ssao.set_int("texNoise", 2)

	shader_ssao_blur.use()
	shader_ssao_blur.set_int("ssaoInput", 0)

	# Main loop
	while not glfw.window_should_close(window):
		impl.process_inputs()
		imgui.new_frame()

		imgui.set_next_window_position(10, 10)
		flags = imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_COLLAPSE
		imgui.begin("Toggle Settings", flags=flags)
		_, use_ssao = imgui.checkbox("Toggle SSAO", use_ssao)
		_, only_ao = imgui.checkbox("Only AO", only_ao)

		imgui.end()

		current_frame = glfw.get_time()
		delta_time = current_frame - last_frame
		last_frame = current_frame

		process_input(window, camera, delta_time)
		glClearColor(0.0, 0.0, 0.0, 1.0)
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

		# Geometry pass
		glBindFramebuffer(GL_FRAMEBUFFER, g_buffer)
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
		projection = glm.perspective(glm.radians(camera.Zoom), SCR_WIDTH / SCR_HEIGHT, 0.1, 50.0)
		view = camera.get_view_matrix()
		model = glm.mat4(1.0)

		shader_geometry_pass.use()
		shader_geometry_pass.setMat4("projection", projection)
		shader_geometry_pass.setMat4("view", view)

		# Render cube
		model = glm.translate(glm.mat4(1.0), glm.vec3(0.0, 7.0, 0.0))
		model = glm.scale(model, glm.vec3(7.5))
		shader_geometry_pass.setMat4("model", model)
		shader_geometry_pass.set_int("invertedNormals", 1)
		shader_geometry_pass.set_bool("useColor", True)
		shader_geometry_pass.setVec3("color", glm.vec3(0.95))
		render_cube()
		if not only_ao:
			shader_geometry_pass.set_bool("useColor", False)
		shader_geometry_pass.set_int("invertedNormals", 0)

		# Render backpack
		model = glm.translate(glm.mat4(1.0), glm.vec3(0.0, 0.5, 0.0))
		model = glm.rotate(model, glm.radians(-90.0), glm.vec3(1.0, 0.0, 0.0))
		shader_geometry_pass.setMat4("model", model)
		backpack.draw(shader_geometry_pass)
		glBindFramebuffer(GL_FRAMEBUFFER, 0)

		# SSAO pass
		glBindFramebuffer(GL_FRAMEBUFFER, ssao_fbo)
		glClear(GL_COLOR_BUFFER_BIT)
		shader_ssao.use()
		for i in range(64):
			shader_ssao.setVec3(f"samples[{i}]", ssao_kernel[i])
		shader_ssao.setMat4("projection", projection)
		glActiveTexture(GL_TEXTURE0)
		glBindTexture(GL_TEXTURE_2D, g_position)
		glActiveTexture(GL_TEXTURE1)
		glBindTexture(GL_TEXTURE_2D, g_normal)
		glActiveTexture(GL_TEXTURE2)
		glBindTexture(GL_TEXTURE_2D, noise_texture)

		render_quad()
		glBindFramebuffer(GL_FRAMEBUFFER, 0)

		# Blur pass
		glBindFramebuffer(GL_FRAMEBUFFER, ssao_blur_fbo)
		glClear(GL_COLOR_BUFFER_BIT)
		shader_ssao_blur.use()
		glActiveTexture(GL_TEXTURE0)
		glBindTexture(GL_TEXTURE_2D, ssao_color_buffer)
		render_quad()
		glBindFramebuffer(GL_FRAMEBUFFER, 0)

		# Lighting pass
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
		shader_lighting_pass.use()
		light_pos_view = glm.vec3(view * glm.vec4(light_pos, 1.0))
		shader_lighting_pass.setVec3("light.Position", light_pos_view)
		shader_lighting_pass.setVec3("light.Color", light_color)
		shader_lighting_pass.set_float("light.Linear", 0.09)
		shader_lighting_pass.set_float("light.Quadratic", 0.032)
		if use_ssao:
			shader_lighting_pass.set_bool("enableSSAO", True)
		else:
			shader_lighting_pass.set_bool("enableSSAO", False)

		glActiveTexture(GL_TEXTURE0)
		glBindTexture(GL_TEXTURE_2D, g_position)
		glActiveTexture(GL_TEXTURE1)
		glBindTexture(GL_TEXTURE_2D, g_normal)
		glActiveTexture(GL_TEXTURE2)
		glBindTexture(GL_TEXTURE_2D, g_albedo)
		glActiveTexture(GL_TEXTURE3)
		glBindTexture(GL_TEXTURE_2D, ssao_color_buffer_blur)
		render_quad()

		imgui.render()
		impl.render(imgui.get_draw_data())
		glfw.swap_buffers(window)
		glfw.poll_events()

	glfw.terminate()


# Cube and Quad VAOs
cube_vao = 0
cube_vbo = 0
quad_vao = 0
quad_vbo = 0


def render_cube():
	global cube_vao, cube_vbo
	if cube_vao == 0:
		vertices = [
			# back face
			-1.0, -1.0, -1.0, 0.0, 0.0, -1.0, 0.0, 0.0,  # bottom-left
			1.0, 1.0, -1.0, 0.0, 0.0, -1.0, 1.0, 1.0,  # top-right
			1.0, -1.0, -1.0, 0.0, 0.0, -1.0, 1.0, 0.0,  # bottom-right
			1.0, 1.0, -1.0, 0.0, 0.0, -1.0, 1.0, 1.0,  # top-right
			-1.0, -1.0, -1.0, 0.0, 0.0, -1.0, 0.0, 0.0,  # bottom-left
			-1.0, 1.0, -1.0, 0.0, 0.0, -1.0, 0.0, 1.0,  # top-left

			# front face
			-1.0, -1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0,  # bottom-left
			1.0, -1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0,  # bottom-right
			1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0,  # top-right
			1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0,  # top-right
			-1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0,  # top-left
			-1.0, -1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0,  # bottom-left

			# left face
			-1.0, 1.0, 1.0, -1.0, 0.0, 0.0, 1.0, 0.0,  # top-right
			-1.0, 1.0, -1.0, -1.0, 0.0, 0.0, 1.0, 1.0,  # top-left
			-1.0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 1.0,  # bottom-left
			-1.0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 1.0,  # bottom-left
			-1.0, -1.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0,  # bottom-right
			-1.0, 1.0, 1.0, -1.0, 0.0, 0.0, 1.0, 0.0,  # top-right

			# right face
			1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0,  # top-left
			1.0, -1.0, -1.0, 1.0, 0.0, 0.0, 0.0, 1.0,  # bottom-right
			1.0, 1.0, -1.0, 1.0, 0.0, 0.0, 1.0, 1.0,  # top-right
			1.0, -1.0, -1.0, 1.0, 0.0, 0.0, 0.0, 1.0,  # bottom-right
			1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0,  # top-left
			1.0, -1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0,  # bottom-left

			# bottom face
			-1.0, -1.0, -1.0, 0.0, -1.0, 0.0, 0.0, 1.0,  # top-right
			1.0, -1.0, -1.0, 0.0, -1.0, 0.0, 1.0, 1.0,  # top-left
			1.0, -1.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0,  # bottom-left
			1.0, -1.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0,  # bottom-left
			-1.0, -1.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0,  # bottom-right
			-1.0, -1.0, -1.0, 0.0, -1.0, 0.0, 0.0, 1.0,  # top-right

			# top face
			-1.0, 1.0, -1.0, 0.0, 1.0, 0.0, 0.0, 1.0,  # top-left
			1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0,  # bottom-right
			1.0, 1.0, -1.0, 0.0, 1.0, 0.0, 1.0, 1.0,  # top-right
			1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0,  # bottom-right
			-1.0, 1.0, -1.0, 0.0, 1.0, 0.0, 0.0, 1.0,  # top-left
			-1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0  # bottom-left
		]
		vertices = np.array(vertices, dtype=np.float32)

		cube_vao = glGenVertexArrays(1)
		cube_vbo = glGenBuffers(1)
		glBindVertexArray(cube_vao)
		glBindBuffer(GL_ARRAY_BUFFER, cube_vbo)
		glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
		glEnableVertexAttribArray(0)
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * 4, None)
		glEnableVertexAttribArray(1)
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * 4, ctypes.c_void_p(3 * 4))
		glEnableVertexAttribArray(2)
		glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * 4, ctypes.c_void_p(6 * 4))
		glBindVertexArray(0)

	glBindVertexArray(cube_vao)
	glDrawArrays(GL_TRIANGLES, 0, 36)
	glBindVertexArray(0)


def render_quad():
	global quad_vao, quad_vbo
	if quad_vao == 0:
		quad_vertices = [
			-1.0, 1.0, 0.0, 0.0, 1.0,
			-1.0, -1.0, 0.0, 0.0, 0.0,
			1.0, 1.0, 0.0, 1.0, 1.0,
			1.0, -1.0, 0.0, 1.0, 0.0,
		]
		quad_vertices = np.array(quad_vertices, dtype=np.float32)

		quad_vao = glGenVertexArrays(1)
		quad_vbo = glGenBuffers(1)
		glBindVertexArray(quad_vao)
		glBindBuffer(GL_ARRAY_BUFFER, quad_vbo)
		glBufferData(GL_ARRAY_BUFFER, quad_vertices.nbytes, quad_vertices, GL_STATIC_DRAW)
		glEnableVertexAttribArray(0)
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * 4, None)
		glEnableVertexAttribArray(1)
		glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * 4, ctypes.c_void_p(3 * 4))
		glBindVertexArray(0)

	glBindVertexArray(quad_vao)
	glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
	glBindVertexArray(0)


def process_input(window, camera, delta_time):
	global cursor_disabled, last_x, last_y, e_pressed
	if glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS:
		glfw.set_window_should_close(window, True)
	if glfw.get_key(window, glfw.KEY_E) == glfw.PRESS:
		e_pressed = True
	if glfw.get_key(window, glfw.KEY_E) == glfw.RELEASE and e_pressed:
		e_pressed = False
		if cursor_disabled:
			cursor_disabled = False
			glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_NORMAL)
		else:
			cursor_disabled = True
			glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_DISABLED)
		last_x, last_y = glfw.get_cursor_pos(window)

	if not cursor_disabled:
		return
	if glfw.get_key(window, glfw.KEY_W) == glfw.PRESS:
		camera.process_keyboard(Camera_Movement.FORWARD, delta_time)
	if glfw.get_key(window, glfw.KEY_S) == glfw.PRESS:
		camera.process_keyboard(Camera_Movement.BACKWARD, delta_time)
	if glfw.get_key(window, glfw.KEY_A) == glfw.PRESS:
		camera.process_keyboard(Camera_Movement.LEFT, delta_time)
	if glfw.get_key(window, glfw.KEY_D) == glfw.PRESS:
		camera.process_keyboard(Camera_Movement.RIGHT, delta_time)
	if glfw.get_key(window, glfw.KEY_SPACE) == glfw.PRESS:
		camera.process_keyboard(Camera_Movement.UP, delta_time)
	if glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS:
		camera.process_keyboard(Camera_Movement.DOWN, delta_time)


def framebuffer_size_callback(window, width, height):
	glViewport(0, 0, width, height)


def mouse_callback(window, xpos_in, ypos_in):
	global first_mouse, last_x, last_y, cursor_disabled
	if not cursor_disabled:
		return

	xpos = float(xpos_in)
	ypos = float(ypos_in)

	if first_mouse:
		last_x = xpos
		last_y = ypos
		first_mouse = False

	xoffset = xpos - last_x
	yoffset = last_y - ypos  # Reversed since y-coordinates go from bottom to top

	last_x = xpos
	last_y = ypos

	camera.process_mouse_movement(xoffset, yoffset)


def scroll_callback(window, xoffset, yoffset):
	camera.process_mouse_scroll(float(yoffset))


if __name__ == "__main__":
	main()
