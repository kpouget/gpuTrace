#! /usr/bin/env python3

import sys
import re
import collections

import gdb_helper
import parse_ocl_program

def print_head(_lookup_table):
    print("#include <stddef.h>")
    print("#include <stdio.h>")
    #print("typedef float realw;")
    print("\n#define NB_CUDA_KERNEL {}".format(len(_lookup_table)))
    
def get_parameters(params, fname):
    yield from params

def get_lookup_table(binary):
    lookup_table = collections.OrderedDict()
    
    symbols = gdb_helper.get_cuda_kernel_names(binary)

    for symb, loc, address in gdb_helper.get_symbol_location(symbols, binary):
        params = parse_ocl_program.parse_cuda(loc, symb)
        lookup_table[address] = symb, params
        
    return lookup_table

def get_safe_function_name(fname):
    return fname.replace("<", "__").replace(">", "__")

def get_arg_struct_name(fname):
    return "struct {}_args_s".format(get_safe_function_name(fname))

def print_parameter_structs(_lookup_table):
    def is_pointer_type(ptype):
        return ptype[-1] == '*'
    
    for address, (fname, params) in _lookup_table.items():
        print("%s {" % get_arg_struct_name(fname))
        for ptype, pname in reversed([p for p in get_parameters(params, fname)]):
            print("  {}{}{};".format(ptype, "" if is_pointer_type(ptype) else " ", pname))
        print("};")
        print()
        
PRINT_TYPE_LOOKUP = {
    "int": "%d",
    "float": "%.2f",
    "char *": "%p",
    "const char *": "%s"
    }
        
def print_param_info_struct():
    print("""
struct param_info_s {
  const char *name;
  const char *type;
  size_t offset;
};
""")

def print_spacer():
    print("/****************************/")

def print_lookup_table_struct(_lookup_table):
    lookup_table_items = ",\n".join([i for i in print_and_get_lookup_table_items(_lookup_table)])
    print("""
struct  kernel_lookup_s {
  void *address;
  const char *name;
  size_t nb_params;
  struct param_info_s *params;
} function_lookup_table[] = { \n%s\n};""" % lookup_table_items)
    
    
def print_preprocessor(binary):
    lookup_table = get_lookup_table(binary)

    print_head(lookup_table)

    print_param_info_struct()
    
    print_parameter_structs(lookup_table)
    
    print_lookup_table_struct(lookup_table)

def print_and_get_lookup_table_items(_lookup_table):
    for address, (fname, params) in _lookup_table.items():
        nb_params = len(params)
        yield '  {(void *) %s, "%s", %d, %s}' % \
            (address, fname, nb_params,
             print_and_get_param_info_items(params, fname))

def print_and_get_param_info_items(params, fname):
    
    def get_param_items():
        for ptype, pname in get_parameters(params, fname):
            offset = "(size_t) &(({} *) NULL)->{}".format(get_arg_struct_name(fname), pname)
            yield '{"%s", "%s", %s}' % (pname, ptype, offset)
            
    varname = "{}_params".format(get_safe_function_name(fname))
    print("struct param_info_s %s[] = {\n  %s\n};\n" % (varname, ",\n  ".join([i for i in get_param_items()])))
    return varname


if __name__ == "__main__":
    binary = sys.argv[1]

    print_preprocessor(binary)
