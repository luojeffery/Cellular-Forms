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
from compute_shader import Compute_Shader

# Settings
SCR_WIDTH = 1920
SCR_HEIGHT = 1080
use_ssao = True
use_phong = True
NUM_CELLS = 25_000   # Max capacity for growth
INITIAL_CELLS = 2_000
ENABLE_FEEDING = True
# Force debugging toggles
enable_spring = True
enable_bulge = True
enable_planar = True
enable_repulsion = True
spring_strength = 50.0
bulge_strength = 2.0
planar_strength = 2.0
repulsion_strength = 0.008
NUM_VOXELS = 216  # 6x6x6; keep <= 256 because prefix-sum shader is single-workgroup
VOXEL_SIZE = 4
GRID_RES = int(round(NUM_VOXELS ** (1 / 3)))  # Will be 4
MAX_LINKS = NUM_CELLS * 6  # Estimate
MAX_DIVISION_QUEUE = 262144  # Maximum divisions per frame
HEALING_ITERATIONS_AFTER_DIVISION = 6
RECENTER_SAMPLE_CELLS = 65536  # Sampled centroid readback cap to keep CPU-GPU sync manageable

# Camera
camera = Camera(glm.vec3(0.0, 0.0, 5.0), pitch=45.0)
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

	glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
	glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 6)
	glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
	glfw.window_hint(glfw.MAXIMIZED, glfw.TRUE)

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


def read_active_cell_count(global_counts_ssbo):
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, global_counts_ssbo)
	data = glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, 4)
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)
	return np.frombuffer(data, dtype=np.uint32)[0]


def read_active_cell_stats(cell_ssbo, sample_cells):
	cell_dtype = np.dtype([
		('position', np.float32, 3),
		('foodLevel', np.float32),
		('voxelCoord', np.float32, 3),
		('radius', np.float32),
		('linkStartIndex', np.int32),
		('linkCount', np.int32),
		('flatVoxelIndex', np.int32),
		('isActive', np.int32),
	])
	cell_data = read_ssbo(cell_ssbo, cell_dtype, sample_cells)
	active_mask = cell_data['isActive'] == 1
	if not np.any(active_mask):
		return np.zeros(3, dtype=np.float32), 1.0
	positions = cell_data['position'][active_mask]
	centroid = np.mean(positions, axis=0).astype(np.float32)
	distances = np.linalg.norm(positions - centroid, axis=1)
	max_distance = float(np.max(distances)) if distances.size > 0 else 1.0
	return centroid, max(max_distance, 1e-4)


def debug_ssbos(cell_ssbo, link_ssbo, j):
	cell_dtype = np.dtype([
		('position', np.float32, 3),
		('foodLevel', np.float32),
		('voxelCoord', np.float32, 3),
		('radius', np.float32),
		('linkStartIndex', np.int32),
		('linkCount', np.int32),
		('flatVoxelIndex', np.int32),
		('isActive', np.int32),
	])

	num_cells = 1024  # or however many total capacity
	num_links = num_cells * 6  # assuming 6 links per cell

	cell_data = read_ssbo(cell_ssbo, cell_dtype, num_cells)
	link_data = read_ssbo(link_ssbo, np.int32, num_links)

	print("=============================================================")
	print(f"Frame {j}")
	# Print active cells
	for i, cell in enumerate(cell_data):
		if cell['isActive'] == 1:
			print(f"Cell {i}: pos={cell['position']}, food={cell['foodLevel']}, links={cell['linkStartIndex']}+{cell['linkCount']}, voxelCoord={cell['voxelCoord']}, flatVoxelIndex={cell['flatVoxelIndex']}, isActive={cell['isActive']}")

	# Print links
	for i in range(0, len(link_data[:500*6]), 6):
		print(f"Links[{i // 6}]:", link_data[i:i + 6])



def read_ssbo(buffer_id, dtype, count):
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffer_id)
	ptr = glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY)
	if not ptr:
		raise RuntimeError("Failed to map buffer")

	dtype = np.dtype(dtype)
	size = dtype.itemsize * count

	buf_type = (GLubyte * size).from_address(ptr)
	array_view = np.ctypeslib.as_array(buf_type).view(dtype)
	array_copy = np.array(array_view, copy=True)  # <-- Force deep copy here

	glUnmapBuffer(GL_SHADER_STORAGE_BUFFER)
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)

	return array_copy


def main():
	global delta_time, last_frame, use_ssao, use_phong, enable_spring, enable_bulge, enable_planar, enable_repulsion
	global spring_strength, bulge_strength, planar_strength, repulsion_strength
	global SCR_WIDTH, SCR_HEIGHT, last_x, last_y

	imgui.create_context()
	window = impl_glfw_init()
	SCR_WIDTH, SCR_HEIGHT = glfw.get_framebuffer_size(window)
	last_x = SCR_WIDTH / 2
	last_y = SCR_HEIGHT / 2
	impl = GlfwRenderer(window)

	glfw.set_framebuffer_size_callback(window, framebuffer_size_callback)
	glfw.set_cursor_pos_callback(window, mouse_callback)
	glfw.set_scroll_callback(window, scroll_callback)
	glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_DISABLED)

	# Configure OpenGL
	glEnable(GL_DEPTH_TEST)

	# Load shaders
	clear_cell_counts = Compute_Shader("shaders/compute/clear_cell_counts.glsl")
	count_cells_per_voxel = Compute_Shader("shaders/compute/count_cells_per_voxel.glsl")
	prefix_sum_voxel_offsets = Compute_Shader("shaders/compute/prefix_sum_voxel_offsets.glsl")
	fill_voxel_cell_ids = Compute_Shader("shaders/compute/fill_voxel_cell_ids.glsl")
	food_enqueue = Compute_Shader("shaders/compute/food_enqueue.glsl")
	process_division_queue = Compute_Shader("shaders/compute/process_division_queue.glsl")
	sanitize_links = Compute_Shader("shaders/compute/sanitize_links.glsl")
	recompute_link_count = Compute_Shader("shaders/compute/recompute_link_count.glsl")
	link_healing = Compute_Shader("shaders/compute/link_healing.glsl")
	recenter_cells = Compute_Shader("shaders/compute/recenter_cells.glsl")
	simulate = Compute_Shader("shaders/compute/simulate.glsl")

	shader_geometry_pass = Shader("shaders/vertex/vs.ssao_geometry.glsl", "shaders/fragment/fs.ssao_geometry.glsl")
	shader_lighting_pass = Shader("shaders/vertex/vs.ssao.glsl", "shaders/fragment/fs.ssao_lighting.glsl")
	shader_ssao = Shader("shaders/vertex/vs.ssao.glsl", "shaders/fragment/fs.ssao.glsl")
	shader_ssao_blur = Shader("shaders/vertex/vs.ssao.glsl", "shaders/fragment/fs.ssao_blur.glsl")

	# Load models
	sphere = Model("objects/sphere/sphere.obj")
	# Create SSBOs
	# std430 Cell stride is 48 bytes: (vec3+float) + (vec3+float) + 4 ints
	CELL_STRUCT_SIZE = 48
	LINK_ENTRY_SIZE = 4  # int
	UINT_SIZE = 4

	# Allocate buffers
	cell_ssbo = create_ssbo(binding_index=0, size_in_bytes=NUM_CELLS * CELL_STRUCT_SIZE)
	link_ssbo = create_ssbo(binding_index=1, size_in_bytes=NUM_CELLS * 6 * LINK_ENTRY_SIZE)  # assume max 8 links per cell
	cell_count_per_voxel_ssbo = create_ssbo(binding_index=2, size_in_bytes=NUM_VOXELS * UINT_SIZE)
	start_index_per_voxel_ssbo = create_ssbo(binding_index=3, size_in_bytes=NUM_VOXELS * UINT_SIZE)
	# todo: might have to change this size
	flat_voxel_cell_ids_ssbo = create_ssbo(binding_index=4, size_in_bytes=NUM_CELLS * UINT_SIZE)
	global_counts_ssbo = create_ssbo(binding_index=5, size_in_bytes=2 * UINT_SIZE)  # numActiveCells, divisionQueueCount
	division_queue_ssbo = create_ssbo(binding_index=7, size_in_bytes=MAX_DIVISION_QUEUE * UINT_SIZE)
	initialize_cells_hex_sheet_with_links(
		cell_ssbo,
		link_ssbo,
		global_counts_ssbo,
		num_cells=INITIAL_CELLS,
		capacity=NUM_CELLS,
		link_rest_length=0.6,
	)
	# create vao for unit sphere
	#vao, vertex_count = create_indexed_sphere_vao()

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
	light_color = glm.vec3(0.68, 0.68, 0.68)

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

	j = 0
	# Main loop
	while not glfw.window_should_close(window):
		impl.process_inputs()
		imgui.new_frame()

		imgui.set_next_window_position(10, 10)
		imgui.set_next_window_size(260, 320)
		flags = imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_COLLAPSE
		imgui.begin("Toggle Settings", flags=flags)
		_, use_ssao = imgui.checkbox("Toggle SSAO", use_ssao)
		_, use_phong = imgui.checkbox("Phong Shading", use_phong)
		active_cells = read_active_cell_count(global_counts_ssbo)
		imgui.text(f"Active Cells: {active_cells}")
		imgui.separator()
		imgui.text("Force Debugging:")
		_, enable_spring = imgui.checkbox("Spring Force", enable_spring)
		_, spring_strength = imgui.slider_float("Spring Strength", spring_strength, 0.0, 100.0)
		_, enable_bulge = imgui.checkbox("Bulge Force", enable_bulge)
		_, bulge_strength = imgui.slider_float("Bulge Strength", bulge_strength, 0.0, 100.0)
		_, enable_planar = imgui.checkbox("Planar Force", enable_planar)
		_, planar_strength = imgui.slider_float("Planar Strength", planar_strength, 0.0, 100.0)
		_, enable_repulsion = imgui.checkbox("Repulsion Force", enable_repulsion)
		_, repulsion_strength = imgui.slider_float("Repulsion Strength", repulsion_strength, 0.0, 7.0)
		imgui.end()

		current_frame = glfw.get_time()
		delta_time = current_frame - last_frame
		last_frame = current_frame

		process_input(window, camera, delta_time)
		glClearColor(0.0, 0.0, 0.0, 1.0)
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

		# Compute Shaders
		# Dispatch all shaders each frame:
		clear_cell_counts.use()
		glDispatchCompute((NUM_VOXELS + 255) // 256, 1, 1)  # Ensure at least 1 work group
		glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

		count_cells_per_voxel.use()
		glDispatchCompute((NUM_CELLS + 255) // 256, 1, 1)  # Ensure at least 1 work group
		glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

		prefix_sum_voxel_offsets.use()
		glDispatchCompute((NUM_VOXELS + 255) // 256, 1, 1)  # Ensure at least 1 work group
		glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

		clear_cell_counts.use()  # Clear again for fill
		glDispatchCompute((NUM_VOXELS + 255) // 256, 1, 1)  # Ensure at least 1 work group
		glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

		fill_voxel_cell_ids.use()
		fill_voxel_cell_ids.setVec3("gridResolution", glm.vec3(GRID_RES, GRID_RES, GRID_RES))
		glDispatchCompute((NUM_CELLS + 255) // 256, 1, 1)  # Ensure at least 1 work group
		glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

		# Reset division queue count
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, global_counts_ssbo)
		glBufferSubData(GL_SHADER_STORAGE_BUFFER, 4, UINT_SIZE, np.array([0], dtype=np.uint32).tobytes())  # Reset divisionQueueCount
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)
		division_happened = False

		if ENABLE_FEEDING:
			# Food + Enqueue pass
			food_enqueue.use()
			food_enqueue.set_int("numCells", NUM_CELLS)
			food_enqueue.set_float("foodPerFrame", delta_time / 10.0)  # Accumulates to 1.0 over 10 seconds
			food_enqueue.set_float("foodThreshold", 1.0)
			food_enqueue.set_int("maxDivisionQueue", MAX_DIVISION_QUEUE)
			glDispatchCompute((NUM_CELLS + 255) // 256, 1, 1)  # Ensure at least 1 work group
			glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

			# Read back division queue count
			glBindBuffer(GL_SHADER_STORAGE_BUFFER, global_counts_ssbo)
			queue_count_data = glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 4, UINT_SIZE)  # Read divisionQueueCount at offset 4
			division_queue_count = np.frombuffer(queue_count_data, dtype=np.uint32)[0]
			glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)

			# Process division queue
			if division_queue_count > 0:
				division_happened = True
				process_division_queue.use()
				process_division_queue.set_int("numCells", NUM_CELLS)
				process_division_queue.set_float("linkRestLength", 0.6)
				glDispatchCompute(min(int(division_queue_count), MAX_DIVISION_QUEUE) // 256 + 1, 1, 1)
				glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

		# Remove broken or one-way links introduced by concurrent topology edits.
		sanitize_links.use()
		glDispatchCompute((NUM_CELLS + 255) // 256, 1, 1)
		glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

		# Recompute linkCount
		recompute_link_count.use()
		glDispatchCompute((NUM_CELLS + 255) // 256, 1, 1)  # Ensure at least 1 work group
		glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

		# Physics pass
		simulate.use()
		simulate.set_float("linkRestLength", 0.6)  # Increased to match initial sphere spacing better
		simulate.set_float("springFactor", spring_strength if enable_spring else 0.0)
		simulate.set_float("planarFactor", planar_strength if enable_planar else 0.0)
		simulate.set_float("bulgeFactor", bulge_strength if enable_bulge else 0.0)
		simulate.set_float("repulsionFactor", repulsion_strength if enable_repulsion else 0.0)
		simulate.set_float("repulsionRadius", 2)
		simulate.setVec3("gridResolution", glm.vec3(GRID_RES, GRID_RES, GRID_RES))
		simulate.set_float("timeStep", 0.005)
		simulate.set_float("voxelSize", VOXEL_SIZE)
		glDispatchCompute((NUM_CELLS + 255) // 256, 1, 1)  # Ensure at least 1 work group
		glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

		healing_iterations = HEALING_ITERATIONS_AFTER_DIVISION if division_happened else 1
		for _ in range(healing_iterations):
			# Link healing pass
			link_healing.use()
			link_healing.setVec3("gridResolution", glm.vec3(GRID_RES, GRID_RES, GRID_RES))
			link_healing.set_float("healingRadius", 0.9)
			link_healing.set_int("targetNeighbors", 6)
			glDispatchCompute((NUM_CELLS + 255) // 256, 1, 1)  # Ensure at least 1 work group
			glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

			# Healing can race; sanitize and recount after each iteration.
			sanitize_links.use()
			glDispatchCompute((NUM_CELLS + 255) // 256, 1, 1)
			glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

			recompute_link_count.use()
			glDispatchCompute((NUM_CELLS + 255) // 256, 1, 1)  # Ensure at least 1 work group
			glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

		# Keep the whole structure centered so it doesn't drift out of view.
		active_cells_now = int(read_active_cell_count(global_counts_ssbo))
		sample_cells = max(1, min(active_cells_now, RECENTER_SAMPLE_CELLS))
		centroid, color_max_distance = read_active_cell_stats(cell_ssbo, sample_cells)
		centroid_len = np.linalg.norm(centroid)
		if centroid_len > 1e-4:
			recenter_cells.use()
			recenter_cells.setVec3("centerOffset", glm.vec3(float(centroid[0]), float(centroid[1]), float(centroid[2])))
			recenter_cells.setVec3("gridResolution", glm.vec3(GRID_RES, GRID_RES, GRID_RES))
			recenter_cells.set_float("voxelSize", VOXEL_SIZE)
			glDispatchCompute((NUM_CELLS + 255) // 256, 1, 1)
			glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)
		j += 1
		# Geometry pass
		glBindFramebuffer(GL_FRAMEBUFFER, g_buffer)
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
		projection = glm.perspective(glm.radians(camera.Zoom), SCR_WIDTH / SCR_HEIGHT, 0.1, 100.0)
		view = camera.get_view_matrix()
		model = glm.mat4(1.0)

		shader_geometry_pass.use()
		shader_geometry_pass.setMat4("projection", projection)
		shader_geometry_pass.setMat4("view", view)
		shader_geometry_pass.setVec3("cellColorCentroid", glm.vec3(float(centroid[0]), float(centroid[1]), float(centroid[2])))
		shader_geometry_pass.set_float("cellColorMaxDistance", color_max_distance)

		# Render cells
		shader_geometry_pass.set_bool("useSSBO", True)
		shader_geometry_pass.setMat4("model", glm.mat4(1.0))
		glMemoryBarrier(GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT)
		sphere.draw_instanced(active_cells_now)


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
		shader_lighting_pass.set_float("light.Linear", 0.045)
		shader_lighting_pass.set_float("light.Quadratic", 0.0075)
		shader_lighting_pass.set_bool("enableSSAO", use_ssao)
		shader_lighting_pass.set_bool("enablePhong", use_phong)

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


def create_ssbo(binding_index, size_in_bytes):
	ssbo = glGenBuffers(1)
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo)
	glBufferData(GL_SHADER_STORAGE_BUFFER, size_in_bytes, None, GL_DYNAMIC_DRAW)
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, binding_index, ssbo)
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)
	return ssbo


def initialize_cells_hex_sheet_with_links(cell_ssbo, link_ssbo, global_counts_ssbo, num_cells=666, capacity=None, link_rest_length=0.6, max_links=6):
	if capacity is None:
		capacity = num_cells
	dtype = np.dtype([
		('position', np.float32, 3),
		('foodLevel', np.float32),
		('voxelCoord', np.float32, 3),
		('radius', np.float32),
		('linkStartIndex', np.int32),
		('linkCount', np.int32),
		('flatVoxelIndex', np.int32),
		('isActive', np.int32),
	])

	# Allocate full capacity; only first num_cells are active sheet cells
	cells = np.zeros(capacity, dtype=dtype)

	def hex_ring_coords(limit):
		coords = []
		radius = 0
		while len(coords) < limit:
			for q in range(-radius, radius + 1):
				for r in range(-radius, radius + 1):
					s = -q - r
					if max(abs(q), abs(r), abs(s)) == radius:
						coords.append((q, r))
						if len(coords) >= limit:
							return coords
			radius += 1
		return coords

	axial_coords = hex_ring_coords(num_cells)
	coord_to_index = {coord: idx for idx, coord in enumerate(axial_coords)}

	# Hex lattice on XZ plane with nearest-neighbor spacing equal to link_rest_length.
	# Basis vectors: a=(d,0), b=(d/2, d*sqrt(3)/2)
	d = float(link_rest_length)
	z_scale = d * (np.sqrt(3.0) / 2.0)

	positions = np.zeros((num_cells, 3), dtype=np.float32)
	for i, (q, r) in enumerate(axial_coords):
		x = d * (q + 0.5 * r)
		z = z_scale * r
		positions[i] = np.array([x, 0.0, z], dtype=np.float32)

	# Center the sheet at the origin.
	positions -= np.mean(positions, axis=0, keepdims=True)

	neighbor_offsets = [
		(1, 0),
		(-1, 0),
		(0, 1),
		(0, -1),
		(1, -1),
		(-1, 1),
	]

	adjacency = [set() for _ in range(num_cells)]
	for i, (q, r) in enumerate(axial_coords):
		for dq, dr in neighbor_offsets:
			j = coord_to_index.get((q + dq, r + dr))
			if j is not None:
				adjacency[i].add(j)

	for i in range(num_cells):
		pos = positions[i]
		vox_coord = np.floor(pos / VOXEL_SIZE)
		flat_voxel_offset = GRID_RES // 2
		shifted_flat_vox = vox_coord + flat_voxel_offset

		cells[i]['position'] = pos
		cells[i]['foodLevel'] = 0.0
		cells[i]['voxelCoord'] = vox_coord
		cells[i]['radius'] = 0.3
		cells[i]['linkStartIndex'] = i * max_links
		cells[i]['linkCount'] = len(adjacency[i])
		if np.any(shifted_flat_vox < 0) or np.any(shifted_flat_vox >= GRID_RES):
			cells[i]['flatVoxelIndex'] = -1
		else:
			cells[i]['flatVoxelIndex'] = int(shifted_flat_vox[0] + shifted_flat_vox[1] * GRID_RES + shifted_flat_vox[2] * GRID_RES * GRID_RES)
		cells[i]['isActive'] = 1

	for i in range(num_cells, capacity):
		cells[i]['isActive'] = 0
		cells[i]['linkStartIndex'] = i * max_links
		cells[i]['linkCount'] = 0

	# Build the flat link array for full capacity (inactive slots have EMPTY links)
	links = np.full(capacity * max_links, -1, dtype=np.int32)
	for i in range(num_cells):
		neighbors = sorted(list(adjacency[i]))
		for k, n in enumerate(neighbors[:max_links]):
			links[i * max_links + k] = n

	global_counts = np.array([num_cells, 0], dtype=np.uint32)  # numActiveCells, divisionQueueCount

	glBindBuffer(GL_SHADER_STORAGE_BUFFER, cell_ssbo)
	glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, cells.nbytes, cells)
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)

	glBindBuffer(GL_SHADER_STORAGE_BUFFER, link_ssbo)
	glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, links.nbytes, links)
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)

	glBindBuffer(GL_SHADER_STORAGE_BUFFER, global_counts_ssbo)
	glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, global_counts.nbytes, global_counts)


def initialize_cells_hollow_sphere_with_links(cell_ssbo, link_ssbo, global_counts_ssbo, num_cells=512, capacity=None, sphere_radius=1, max_links=6):
	if capacity is None:
		capacity = num_cells
	dtype = np.dtype([
		('position', np.float32, 3),
		('foodLevel', np.float32),
		('voxelCoord', np.float32, 3),
		('radius', np.float32),
		('linkStartIndex', np.int32),
		('linkCount', np.int32),
		('flatVoxelIndex', np.int32),
		('isActive', np.int32),
	])
	# Allocate full capacity; only first num_cells are active sphere cells
	cells = np.zeros(capacity, dtype=dtype)
	golden_angle = np.pi * (3 - np.sqrt(5))

	for i in range(num_cells):
		y = 1 - (i / max(num_cells - 1, 1)) * 2
		radius = np.sqrt(1 - y ** 2)
		theta = golden_angle * i
		x = np.cos(theta) * radius
		z = np.sin(theta) * radius
		pos = np.array([x, y, z]) * sphere_radius
		vox_coord = np.floor(pos / VOXEL_SIZE)
		flat_voxel_offset = GRID_RES // 2
		shifted_flat_vox = vox_coord + flat_voxel_offset
		cells[i]['position'] = pos
		cells[i]['voxelCoord'] = vox_coord
		cells[i]['radius'] = 0.3
		cells[i]['linkStartIndex'] = i * max_links
		cells[i]['linkCount'] = max_links
		cells[i]['flatVoxelIndex'] = int(shifted_flat_vox[0] + shifted_flat_vox[1] * GRID_RES + shifted_flat_vox[2] * GRID_RES * GRID_RES)
		cells[i]['isActive'] = 1
	for i in range(num_cells, capacity):
		cells[i]['isActive'] = 0
		cells[i]['linkStartIndex'] = i * max_links
		cells[i]['linkCount'] = 0

	# Build empty adjacency lists (only for initial sphere)
	adjacency = [set() for _ in range(num_cells)]
	link_counts = [0 for _ in range(num_cells)]

	# Consider all unique pairs once
	positions = cells['position'][:num_cells]
	for i in range(num_cells):
		dists = np.linalg.norm(positions - positions[i], axis=1)
		sorted_indices = np.argsort(dists)
		for j in sorted_indices:
			if i == j:
				continue
			if link_counts[i] >= max_links or link_counts[j] >= max_links:
				continue
			if j in adjacency[i] or i in adjacency[j]:
				continue
			# Make symmetric link
			adjacency[i].add(j)
			adjacency[j].add(i)
			link_counts[i] += 1
			link_counts[j] += 1
			if link_counts[i] >= max_links and link_counts[j] >= max_links:
				break

	# Build the flat link array for full capacity (inactive slots have EMPTY links)
	links = np.full(capacity * max_links, -1, dtype=np.int32)
	for i in range(num_cells):
		neighbors = sorted(list(adjacency[i]))  # Sort for deterministic ordering
		cells[i]['linkCount'] = len(neighbors)
		for k, n in enumerate(neighbors):
			links[i * max_links + k] = n

	"""
	debug for link healing
	arr = np.zeros(2, dtype=dtype)
	arr[0]["position"] = np.array([-5, -3, -5])
	arr[0]["voxelCoord"] = np.array([-3, -2, -3])
	arr[0]["linkStartIndex"] = 16 * 6
	arr[0]["linkCount"] = 0
	arr[0]["radius"] = 0.35
	arr[0]["isActive"] = 1
	arr[1]["position"] = np.array([-5, -3, -6])
	arr[1]["voxelCoord"] = np.array([-3, -2, -3])
	arr[1]["linkStartIndex"] = 17 * 6
	arr[1]["linkCount"] = 0
	arr[1]["radius"] = 0.35
	arr[1]["isActive"] = 1
	arr[1]["flatVoxelIndex"] = 232

	extra_links = np.full(2 * max_links, -1, dtype=np.int32)

	links = np.concatenate((links, extra_links))

	cells = np.concatenate((cells, arr))
	"""
	# -----------------------------
	# 3. Generate global counts for cells (implicitly can calculate links)
	global_counts = np.array([num_cells, 0], dtype=np.uint32)  # numActiveCells, divisionQueueCount
	# -----------------------------
	# 4. Upload to GPU
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, cell_ssbo)
	glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, cells.nbytes, cells)
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)

	glBindBuffer(GL_SHADER_STORAGE_BUFFER, link_ssbo)
	glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, links.nbytes, links)
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)

	glBindBuffer(GL_SHADER_STORAGE_BUFFER, global_counts_ssbo)
	glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, global_counts.nbytes, global_counts)


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
		glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_DYNAMIC_DRAW)
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
	move_delta = delta_time
	if glfw.get_key(window, glfw.KEY_LEFT_CONTROL) == glfw.PRESS or glfw.get_key(window, glfw.KEY_RIGHT_CONTROL) == glfw.PRESS:
		move_delta *= 3.0

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
		camera.process_keyboard(Camera_Movement.FORWARD, move_delta)
	if glfw.get_key(window, glfw.KEY_S) == glfw.PRESS:
		camera.process_keyboard(Camera_Movement.BACKWARD, move_delta)
	if glfw.get_key(window, glfw.KEY_A) == glfw.PRESS:
		camera.process_keyboard(Camera_Movement.LEFT, move_delta)
	if glfw.get_key(window, glfw.KEY_D) == glfw.PRESS:
		camera.process_keyboard(Camera_Movement.RIGHT, move_delta)
	if glfw.get_key(window, glfw.KEY_SPACE) == glfw.PRESS:
		camera.process_keyboard(Camera_Movement.UP, move_delta)
	if glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS:
		camera.process_keyboard(Camera_Movement.DOWN, move_delta)


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
