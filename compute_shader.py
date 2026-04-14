from OpenGL.GL import *

from shader import check_compile_errors
from pyglm import glm


class Compute_Shader:
    def __init__(self, path):
        self.path = path
        self.program_id = None
        self.load()

    def load(self):
        # Load and compile compute shader, link program
        with open(self.path, "r") as file:
            compute_shader_code = file.read()

        # Compile compute shader
        compute_shader = glCreateShader(GL_COMPUTE_SHADER)
        glShaderSource(compute_shader, compute_shader_code)
        glCompileShader(compute_shader)
        check_compile_errors(compute_shader, "COMPUTE")

        # Link program
        self.program_id = glCreateProgram()
        glAttachShader(self.program_id, compute_shader)
        glLinkProgram(self.program_id)
        check_compile_errors(self.program_id, "PROGRAM")

        # Delete shader as it's linked into program now and no longer necessary
        glDeleteShader(compute_shader)

    def use(self):
        glUseProgram(self.program_id)

    def set_bool(self, name, value):
        glUniform1i(glGetUniformLocation(self.program_id, name), value)

    def set_int(self, name, value):
        glUniform1i(glGetUniformLocation(self.program_id, name), value)

    def set_uint(self, name, value):
        glUniform1ui(glGetUniformLocation(self.program_id, name), value)

    def set_float(self, name, value):
        glUniform1f(glGetUniformLocation(self.program_id, name), value)

    def setVec2(self, name: str, *args):
        if len(args) == 1 and type(args[0]) == glm.vec2:
            glUniform2fv(
                glGetUniformLocation(self.program_id, name), 1, glm.value_ptr(args[0])
            )

        elif len(args) == 2 and all(map(lambda x: type(x) == float, args)):
            glUniform2f(glGetUniformLocation(self.program_id, name), *args)

    def setVec3(self, name: str, *args):
        if len(args) == 1 and type(args[0]) == glm.vec3:
            glUniform3fv(
                glGetUniformLocation(self.program_id, name), 1, glm.value_ptr(args[0])
            )

        elif len(args) == 3 and all(map(lambda x: type(x) == float, args)):
            glUniform3f(glGetUniformLocation(self.program_id, name), *args)

    def setVec4(self, name: str, *args) -> None:
        if len(args) == 1 and type(args[0]) == glm.vec4:
            glUniform4fv(
                glGetUniformLocation(self.program_id, name), 1, glm.value_ptr(args[0])
            )

        elif len(args) == 3 and all(map(lambda x: type(x) == float, args)):
            glUniform4f(glGetUniformLocation(self.program_id, name), *args)

    def setMat2(self, name: str, mat: glm.mat2) -> None:
        glUniformMatrix2fv(
            glGetUniformLocation(self.program_id, name), 1, GL_FALSE, glm.value_ptr(mat)
        )

    def setMat3(self, name: str, mat: glm.mat3) -> None:
        glUniformMatrix3fv(
            glGetUniformLocation(self.program_id, name), 1, GL_FALSE, glm.value_ptr(mat)
        )

    def setMat4(self, name: str, mat: glm.mat4) -> None:
        glUniformMatrix4fv(
            glGetUniformLocation(self.program_id, name), 1, GL_FALSE, glm.value_ptr(mat)
        )
