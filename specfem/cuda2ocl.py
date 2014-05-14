#!/usr/bin/python3

import sys

COMMAND_QUEUE = "command_queue"
CONTEXT = "context"
PROGRAM_SUF = "_program"
KERNEL_SUF = "_kern"

DO_STDIN_READ = True
size_arguments = {}
kernel_definitions = {}
progr_name = None
reader = None

def treat_program():
    print("""
/* create {progr} program */
cl_program {progr} = oclGetProgramFromSource ({context}, device_id, source_name, &errcode);
""".format(progr=progr_name, context=CONTEXT), file=sys.stderr)

def treat_prototype(kernel_src):
    prototype = kernel_src[:kernel_src.index("{")]
    if "USE_LAUNCH_BOUNDS" in kernel_src:
        prototype = prototype.partition("#endif")[2]
        
    kern_name = prototype.split("(")[0].split(" ")[-1].strip()

    params_strs = prototype.split("(")[-1].split(")")[0].split(",")
    params = {}
    for param in params_strs:
        space_idx = param.rindex(" ")
        name = param[space_idx:]
        while name[0] == "*":
            name = name[1:]
            space_idx += 1
            
        ttype = param[0:space_idx]
        params[name.strip()] = ttype.strip()

    boast_cuda_name = kern_name + "|"
    
    boast_cuda_name = boast_cuda_name.replace("cuda_kernel", "kernel_cuda")
    boast_cuda_name = boast_cuda_name.replace("kernel|", "kernel_cuda|")
    boast_cuda_name = boast_cuda_name.replace("cuda_device", "kernel_cuda")
    
    boast_cuda_name = boast_cuda_name.replace("multiply", "update")

    boast_cuda_name = boast_cuda_name.replace("UpdateDispVeloc", "update_disp_veloc")
    boast_cuda_name = boast_cuda_name.replace("UpdatePotential", "update_potential")

    boast_cuda_name = boast_cuda_name.replace("compute_kernels", "compute")
    boast_cuda_name = boast_cuda_name.replace("_cudakernel_", "_kernel_")

    boast_cuda_name = boast_cuda_name.replace("Kernel_2_", "")
    boast_cuda_name = boast_cuda_name.replace("impl|", "impl_kernel_cuda|")
    
    boast_cuda_name = boast_cuda_name.replace("_element_", "_element_NOPE NOPE NOPE")

    boast_cuda_name = boast_cuda_name.replace("compute_strain_product_cuda|", "compute_acoustic_kernel_cuda|")
    boast_cuda_name = boast_cuda_name.replace("compute_gradient_kernel_cuda|", "compute_acoustic_kernel_cuda|")

    boast_cuda_name = boast_cuda_name.replace("_cuda|", "|")
    
    kern_name_long = kern_name + KERNEL_SUF

    kernel_definitions[kern_name] = params

    print("""#include "{kern_name}" """.format(kern_name=boast_cuda_name))

    return kern_name, boast_cuda_name.replace("|", "")
    #print("""// prepare kernel {name}
#cl_kernel {kern_name} = clCreateKernel({progr_name}, "{name}", &errcode);
#""".format(kern_name=kern_name_long, progr_name=progr_name, name=kern_name), file=sys.stderr)

def print_size(name):
    if name == "grid": print("size_t global_work_size[2];")
    if name == "threads": print("size_t local_work_size[2];")
    
def prepare_size(line):
    if "(" not in line or "=" in line:
        prepare_indirect_size(line)
    else:
        prepare_direct_size(line)

def prepare_indirect_size(line):
    if "(" not in line:
        assert line.startswith("dim3 ")
        line = line.replace("dim3 ", "")
        line = line.replace(";", "")
        variables = [l.strip() for l in line.split(",")]
        for var in variables:
            size_arguments[var] = None
    else:
        var = line[:line.index("=")].strip()
        assert var in size_arguments.keys()
        params = line[line.index("(")+1 : line.index(")")]
        size_arguments[var] = [p.strip() for p in params]

        print_size(var)
        
def prepare_direct_size(line):
    global size_arguments

    ttype_idx = line.index(" ")
    if line[:ttype_idx] != "dim3":
        print ("// Error: excepted a dim3 variable")
        print ("// Error: excepted a dim3 variable", file=sys.stderr)
        return
    name = line[ttype_idx + 1:].split("(")[0]
    params = line[ttype_idx + 1:].split("(")[1].split(")")[0].split(',')
    size_arguments[name] = [p.strip() for p in params]

    print_size(name)
    
def treat_call(kernel_call):
    name = kernel_call.split("<<<")[0]
    size = kernel_call.split("<<<")[1].split(">>>")[0].split(",")
    size = [s.strip() for s in size]
    
    params = [p.strip() for p in kernel_call.split("(")[1].split(")")[0].split(",")]

    ret = "// start kernel {} execution".format(name)
    assert len(params) == len(kernel_definitions[name])
    kern_name = name + KERNEL_SUF
    
    for i, (pname, ptype) in enumerate(kernel_definitions[name].items()):
        ret += """
errcode = clSetKernelArg({kern_name}, {i}, sizeof({ptype}), (void *) &{pname});""".format(kern_name=kern_name, i=i, ptype=ptype, pname=params[i])

    size_global = size_arguments[size[0]]
    size_local  = size_arguments[size[1]]
    ret += "\n\n"
    ret += "local_work_size[0] = {} ;\n".format(size_local[0])
    ret += "local_work_size[1] = {} ;\n".format(size_local[1])

    ret += "global_work_size[0] = {} ;\n".format(size_global[0])
    ret += "global_work_size[1] = {} ;\n".format(size_global[1])
    
    ret += """
errcode = clEnqueueNDRangeKernel({command_queue}, {kern_name}, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL);""".format(command_queue=COMMAND_QUEUE, kern_name=kern_name, size_global=size_global, size_local=size_local)

    print(ret)

def treat_memcpyToSymb(memcpy_call):
    memcpy_call = memcpy_call[memcpy_call.index("(") + 1:memcpy_call.rindex(")")].split(",")
    print(
        "errcode = clEnqueueWriteBuffer({command_queue}, {buff_name}, CL_TRUE, 0,\n{size}, \n{dest}, 0, NULL, NULL);""".format(command_queue=COMMAND_QUEUE, buff_name=memcpy_call[1], dest=memcpy_call[0], size=memcpy_call[2])
        )
    
def treat_memcpy(memcpy_call):
    if "cudaMemcpyToSymbol" in memcpy_call:
        return treat_memcpyToSymb(memcpy_call)
    
    memcpy_call = memcpy_call[:memcpy_call.rindex(")") + 1]

    while memcpy_call.index("cudaMemcpy") != 0:
        memcpy_call = memcpy_call[memcpy_call.index("(") + 1:memcpy_call.rindex(")", 0, memcpy_call.rindex(")"))+1]
        
    memcpy_call = memcpy_call[memcpy_call.index("(") + 1:memcpy_call.rindex(")")]
    memcpy_call = [p.strip() for p in memcpy_call.split(",")]

    mapping = {"cudaMemcpyDeviceToHost" : "clEnqueueReadBuffer",
               "cudaMemcpyHostToDevice" : "clEnqueueWriteBuffer"}
    
    print("""
errcode = {fct}({command_queue}, {buff_name}, CL_TRUE, 0,\n{size}, \n{dest}, 0, NULL, NULL);""".format(command_queue=COMMAND_QUEUE, buff_name=memcpy_call[0], dest=memcpy_call[1], size=memcpy_call[2], fct=mapping[memcpy_call[3]])
        )

def treat_malloc(malloc_call):
    malloc_call = malloc_call[:malloc_call.rindex(")") + 1]

    while malloc_call.index("cudaMalloc") != 0:
        malloc_call = malloc_call[malloc_call.index("(") + 1:malloc_call.rindex(")", 0, malloc_call.rindex(")"))+1]
        
    malloc_call = malloc_call[malloc_call.index("(") + 1:malloc_call.rindex(")")]
    malloc_call = [p.strip() for p in malloc_call.split(",")]

    buff_name = malloc_call[0]
    if "&" in buff_name:
        buff_name = buff_name.split("&")[1]
    
    print("""
cl_mem {buff_name} = clCreateBuffer({context}, CL_MEM_READ_WRITE, {size}, NULL, &errcode);\
""".format(context=CONTEXT, command_queue=COMMAND_QUEUE, buff_name=buff_name, size=malloc_call[1]))

def treat_free(free_call):
    free_call = "".join([l.strip() for l in free_call])
    print(free_call.replace("cudaFree", "clReleaseMemObject"))

def treat_proto_and_eat(statement):
    if "__device__" in statement:
        return 
    kern_name, boast_name = treat_prototype(statement)
    eat(kern_name, boast_name, statement)

#mp->d_xix_crust_mantle = clCreateBuffer (context, CL_MEM_READ_WRITE, size_padded*sizeof (realw), NULL, &errcode);
#__device__ realw d_hprime_xx[NGLL2];
def treat_device(statement):
    _, _type, var = statement[:statement.index(";")].split()
    varname = var[:var.index("[")]
    varsize = var[var.index("[") + 1:-1]
    print(
"cl_mem {} = clCreateBuffer (context, CL_MEM_READ_ONLY, {}*sizeof({}), NULL, &errcode);".format(varname, varsize, _type)
        )

def do_nop(statement):
    return

keywords = {
  #  "size" : ("dim3", "size definition", prepare_size),
    "proto" : ("__ void", "kernel prototype", treat_proto_and_eat),
  #  "exec" : ("<<<", "kernel execution", treat_call),
  #  "memcp" : ("cudaMemcpy", "memory copy", treat_memcpy),
  #  "malloc" :  ("cudaMalloc", "malloc start", treat_malloc),
  #  "free" : ("cudaFree", "free statement", treat_free),
  #  "getsymbaddr" : ("cudaGetSymbolAddress", "cudaGetSymbolAddress", do_nop),
  #  "__device__" : ("__device__", "__device__ statement", treat_device),
  #  "quit" : ("quit", "quit", lambda x : True)
    }


def eat(kern_name, boast_name, proto):
    proto = proto.replace(kern_name, boast_name)
    fref = open("references/{}.cu".format(boast_name), "w+")
    print("// from {}".format(sys.argv[1]), file=fref)
    print(proto, file=fref)
    
    stack = 1
    while stack != 0:
        nb, line = reader.send(0)
        #line = line.split("//")[0]
        stack += line.count("{")
        stack -= line.count("}")
        print(line, end="", file=fref)
        #print("EAT:{} {}".format(nb, line), end="")
    fref.close()
    
if __name__ == "__main__":
    filename = sys.argv[1]
    
    progr_name = filename
    if "/" in progr_name: progr_name = progr_name[filename.rindex("/") + 1:]
    if "./" in progr_name: progr_name = progr_name[:filename.rindex(".c")]
    if progr_name.endswith("cuda"): progr_name = progr_name[:progr_name.index("cuda")]
    while progr_name.endswith("_"): progr_name = progr_name[:-1]
    
    progr_name +=  PROGRAM_SUF
    #treat_program()

    def Reader():
        with open(filename, "r") as cuda_in:
            nb = 0
            for line in cuda_in:
                nb += 1
                yield (nb, line)

    reader = Reader()
    while True:
        try:
            lines = []
            handled = False
            for nb, line in reader:
                if "//" in line:
                    print(line[line.index("//"):], end="")
                    line = line[:line.index("//")]
                    if not len(lines) and not len(line):
                        continue
                    
                if "/*" in line:
                    #print("COMMENT: {} {}".format(nb, line), end="")
                    print(line, end="")
                    next_line = line[2:]
                    
                    line = line[:line.index("/*")] + "\n"
                    
                    while not "*/" in next_line:
                        nb, next_line = reader.send(0)
                        #print("COMMENT: {} {}".format(nb, next_line), end="")
                        print(next_line, end="")
                    handled = True
                    break
                line = line[:-1].strip()
                lines.append(line)
                if len(lines) == 1 and len(line) == 0: break # empty line
                if len(lines) == 1 and line.startswith("#"): break  # preproc line
                if len(lines) == 1 and line.startswith("}"): break  # preproc line
                if line.endswith("{") or line.endswith(";") : break # statement or declaration
                
            else:
                raise StopIteration()

            if handled:
                continue
            
            lin_statement = "".join([l.strip() for l in lines])
            statement = "\n".join([l.strip() for l in lines])
            
            
            for name, (key, descr, funct) in keywords.items():
                if key in statement:
                    descr = "Handle " + descr
                    #print("\n###{}###".format("#" * len(descr)))
                    #print("## {} ##".format(descr))
                    #print("###{}###\n".format("#" * len(descr)))
                    #print("/* {} */".format(descr))
                    handled = True
                    if funct(lin_statement):
                        exit()
                    #print("\n####################\n")
            if handled:
                statement = "/* {} */".format(lin_statement)
            print("{}".format(statement))
        except StopIteration:
            #print("// Done :)", file=sys.stderr)
            break
        except Exception as e:
            print("// CAUGHT!", file=sys.stderr)
            import pdb;pdb.set_trace()
            import traceback
            print("/* {} */".format(traceback.format_exc()), file=sys.stderr )
            
