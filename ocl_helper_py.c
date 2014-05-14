#include <python3.3m/Python.h>

#include "ocl_helper.h"

#define FILENAME "parse_ocl_program"


PyObject *pf_parse_ocl, *pf_prep_progr;

void init_helper(void) {
  PyObject *pDict, *pModule;
  
  Py_Initialize();
  PySys_SetPath(
    //L"/home/kevin/travail/sample/cl-specfem3d/SPECFEM3D_GLOBE/ldChecker"
    L"/home/kevin/cl-specfem3d/SPECFEM3D_GLOBE/ldChecker"
    );
  
  pModule = PyImport_ImportModule(FILENAME);
  PyErr_Print();
  
  pDict = PyModule_GetDict(pModule);
  
  pf_parse_ocl = PyDict_GetItemString(pDict, "parse_ocl");
  pf_prep_progr = PyDict_GetItemString(pDict, "prepare_program");
  
  Py_DECREF(pDict);
  Py_DECREF(pModule);
}

void handle_program(void *program,
                    unsigned int count,
                    const char **strings,
                    const size_t *lengths) {
  PyObject *p_progr_uid = PyUnicode_FromFormat("%p", program);
  PyObject *p_program_lines = PyTuple_New(count);
  PyObject *p_params;
  int i;
  
  for (i = 0; i < count; i++) {
    PyObject *line;
    if (!lengths || !lengths[i])
      line = PyUnicode_FromString(strings[i]);
    else
      line = PyUnicode_FromStringAndSize(strings[i], lengths[i]);
    
    PyTuple_SetItem(p_program_lines, i, line);
  }

  p_params = PyTuple_Pack(2, p_progr_uid, p_program_lines);

  PyObject_Call(pf_prep_progr, p_params, NULL);
  PyErr_Print();
  
  
  Py_DECREF(p_params);
  Py_DECREF(p_program_lines);
  Py_DECREF(p_progr_uid);
}

char **handle_create_kernel(void *program, void *kernel, const char *name) {
  PyObject *p_progr_uid = PyUnicode_FromFormat("%p", program);
  PyObject *p_kern_uid = PyUnicode_FromFormat("%p", kernel);
  PyObject *p_kern_name = PyUnicode_FromString(name);
  PyObject *p_result, *p_params = PyTuple_Pack(2, p_progr_uid, p_kern_name);
  Py_ssize_t result_size;
  char **param_types_names;
  int i;
  
  p_result = PyObject_Call(pf_parse_ocl, p_params, NULL);
  PyErr_Print();

  result_size = PyList_Size(p_result);
  param_types_names = malloc(sizeof(char *)*(result_size + 1));
  
  for (i = 0; i < result_size; i++) {
    PyObject *p_ascii_str = PyUnicode_AsASCIIString(PyList_GetItem(p_result, i));
   
    param_types_names[i] = PyBytes_AsString(p_ascii_str);
  }
  param_types_names[result_size] = NULL;
  
  Py_DECREF(p_result);
  Py_DECREF(p_params);
  Py_DECREF(p_progr_uid);
  Py_DECREF(p_kern_uid);
  Py_DECREF(p_kern_name);

  return param_types_names;
}

