#! /usr/bin/env python3

import sys
import os

try:
    import gdb_helper
except:
    pass

def get_prog_lines():
    with open("kern") as fkern:
        for line in fkern:
            yield line

programs = {}
def prepare_program(uid, lines):
    progr_lines = []
    for line in lines:
        progr_lines += line.split("\n")
    programs[uid] = progr_lines
            
def parse_ocl(program_uid, kernel_name):
    def get_lines():
        yield from programs[program_uid]

    return parse(get_lines(), kernel_name)

def parse_cuda(source_name, kernel_name):
    def get_lines():
        with open(source_name, "r") as fsource:
            yield from fsource.readlines()
    
    arglist = parse(get_lines(), kernel_name)
    
    arguments = list(zip(arglist[0::2], arglist[1::2]))

    return arguments

def parse(lines, kernel_name):
    if "<" in kernel_name:
        kernel_name = kernel_name.split("<")[0]
    else:
        kernel_name = "void {}".format(kernel_name)
          
    parameters = []
    while True:
        line = lines.send(None)
        if " __attribute__" in line:
            idx = line.index(" __attribute__")
            closeAt = idx
            parents = 0
            inside = False
            while not (inside and parents == 0):
                if line[closeAt] == "(":
                    inside = True
                    parents += 1
                elif line[closeAt] == ")":
                    parents -= 1
                closeAt += 1

            line = line[:idx] + line[closeAt:]
            
        if kernel_name in line:
            while not "{" in line:
                line += " " + lines.send(None).strip()
            break

    for param in line.split("(")[1].split(")")[0].split(","):
        param = param.strip()
        type_ = " ".join([w for w in param.split()[:-1] if not w.startswith("__")])
        name = param.split()[-1]
        while name.startswith("*"):
            name = name[1:]
            type_ = type_ + " *"
        parameters.append(type_)
        parameters.append(name)

    return parameters


if __name__ == "__main__":
    binary = sys.argv[1] if len(sys.argv) > 1 else "./hello"
    symbols = gdb_helper.get_cuda_kernel_names()
    
    for symb, loc, address in gdb_helper.get_symbol_location(symbols):
        print(symb)
        print(parse_cuda(loc, symb))
