
import fileinput
import os

build_call = """  build_{name}_program (mocl);"""

build_function = """
void build_{name}_program (mesh_opencl_t *mocl) {{
  cl_int errcode;

  #include "{name}_cl.c"
  
  mocl->programs.{name}_program = clCreateProgramWithSource(mocl->context, 1, {name}, NULL, clck_(&errcode));
  clCheck (clBuildProgram (mocl->programs.{name}_program, 0, NULL, NULL, NULL, NULL));

  mocl->kernels.{name} = clCreateKernel (mocl->programs.{name}_program, "{name}", clck_(&errcode));
}}
"""
kernel_declaration = """  cl_kernel {name};"""

program_declaration = """  cl_program {name}_program;"""

release_call = """
  clCheck (clReleaseKernel (mocl->kernels.{name}));
  clCheck (clReleaseProgram (mocl->kernels.{name}_program));
"""

names = [l[:l.index("_cl.c")] for l in fileinput.input()]

print "/* --------------- 8< ----- HEADER ----- >8 ------- */"

print "struct mesh_programs_s {"
for name in names: print program_declaration.format(name=name)
print "};"

print "struct mesh_kernels_s {"
for name in names: print kernel_declaration.format(name=name)
print "};"

print """
typedef struct _mesh_opencl {
  struct mesh_programs_s programs;
  struct mesh_kernels_s kernels;
  cl_command_queue command_queue;
  cl_context context;
} mesh_opencl_t;
"""

print "/* --------------- 8< ----- SOURCE ----- >8 ------- */"

for name in names: print build_function.format(name=name)

print "void build_kernels (mesh_opencl_t *mocl) {"
for name in names:
    print build_call.format(name=name)
print "}\n"

print "void release_kernels (mesh_opencl_t *mocl) {"
for name in names: print release_call.format(name=name),
print "}\n"
