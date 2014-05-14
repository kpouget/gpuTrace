#include <CL/cl.h>

#include "ldChecker.h"
#include "ocl_helper.h"

struct ld_ocl_s {
  cl_context context;
  cl_command_queue command_queue;
} ldOclEnv;

struct ld_program_s {
  cl_program handle;
  char *source;
};

struct ld_mem_offset_s {
  cl_mem handle;
  size_t size;
  ld_flags flags;
  size_t offset;
  unsigned int released;
  struct ld_mem_s *parent;
};

struct ld_queue_s {
  cl_command_queue handle;
};


cl_command_queue (*real_clCreateCommandQueue)(cl_context context,
                                              cl_device_id device,
                                              cl_command_queue_properties properties,
                                              cl_int *errcode_ret);
cl_int (*real_clFinish) (cl_command_queue command_queue);
cl_int (*real_clReleaseCommandQueue) (cl_command_queue command_queue);
cl_int (*real_clSetKernelArg) (cl_kernel kernel, cl_uint arg_index,
                               size_t arg_size, const void *arg_value);
cl_mem (*real_clCreateBuffer) (cl_context context, cl_mem_flags flags,
                               size_t size, void *host_ptr,
                               cl_int *errcode_ret);
cl_mem (*real_clCreateSubBuffer) (cl_mem buffer,
                             cl_mem_flags flags,
                             cl_buffer_create_type buffer_create_type,
                             const void *buffer_create_info,
                             cl_int *errcode_ret);
cl_int (*real_clReleaseMemObject) (cl_mem memobj);
cl_kernel (*real_clCreateKernel) (cl_program  program, const char *kernel_name,
                                  cl_int *errcode_ret);
cl_program (*real_clCreateProgramWithSource) (cl_context context,
                                              cl_uint count,
                                              const char **strings,
                                              const size_t *lengths,
                                              cl_int *errcode_ret);
cl_int (*real_clEnqueueNDRangeKernel) (cl_command_queue command_queue,
                                       cl_kernel kernel, cl_uint work_dim,
                                       const size_t *global_work_offset,
                                       const size_t *global_work_size,
                                       const size_t *local_work_size,
                                       cl_uint num_events_in_wait_list,
                                       const cl_event *event_wait_list,
                                       cl_event *event);
cl_int (*real_clEnqueueWriteBuffer) (cl_command_queue command_queue,
                                     cl_mem buffer,
                                     cl_bool blocking_write,
                                     size_t offset,
                                     size_t size,
                                     const void *ptr,
                                     cl_uint num_events_in_wait_list,
                                     const cl_event *event_wait_list,
                                     cl_event *event);
cl_int (*real_clEnqueueReadBuffer) (cl_command_queue command_queue,
                                    cl_mem buffer,
                                    cl_bool blocking_read,
                                    size_t offset,
                                    size_t size,
                                    void *ptr,
                                    cl_uint num_events_in_wait_list,
                                    const cl_event *event_wait_list,
                                    cl_event *event);

struct ld_bindings_s ocl_bindings[] = {
  {"clSetKernelArg", (void **) &real_clSetKernelArg},
  {"clCreateBuffer", (void **) &real_clCreateBuffer},
  {"clCreateSubBuffer", (void **) &real_clCreateSubBuffer},
  {"clReleaseMemObject", (void **) &real_clReleaseMemObject},
  {"clCreateKernel", (void **) &real_clCreateKernel},
  {"clCreateProgramWithSource", (void **) &real_clCreateProgramWithSource},
  {"clEnqueueNDRangeKernel", (void **) &real_clEnqueueNDRangeKernel},
  {"clEnqueueWriteBuffer", (void **) &real_clEnqueueWriteBuffer},
  {"clEnqueueReadBuffer", (void **) &real_clEnqueueReadBuffer},
  {"clCreateCommandQueue", (void **) &real_clCreateCommandQueue},
  {"clFinish", (void **) &real_clFinish},
  {"clReleaseCommandQueue", (void **) &real_clReleaseCommandQueue},
  {NULL, NULL}
};

CREATE_HASHMAP(program, cl_program, 100)
CREATE_HASHMAP(kernel, cl_kernel, 100)
CREATE_HASHMAP(mem, cl_mem, 400)
CREATE_HASHMAP(mem_offset, cl_mem, 200)
CREATE_HASHMAP(queue, cl_command_queue, 10)

/* ************************************************************************* */

int ocl_getBufferContent (struct ld_mem_s *ldBuffer, void *buffer,
                          size_t offset, size_t size);
struct ld_mem_s *create_ocl_buffer (cl_mem handle);

void init_ocl_ldchecker(void) {
  struct callback_s callbacks = {ocl_getBufferContent};
  
  init_ldchecker(callbacks, ocl_bindings);
  create_ocl_buffer(NULL);
  init_helper();
}

cl_command_queue clCreateCommandQueue(cl_context context,
                                      cl_device_id device,
                                      cl_command_queue_properties properties,
                                      cl_int *errcode_ret)
{
  init_ocl_ldchecker();
  
  ldOclEnv.command_queue = real_clCreateCommandQueue(context, device,
                                                     properties, errcode_ret);
  ldOclEnv.context = context;
  
  return ldOclEnv.command_queue;
}

cl_int clFinish (cl_command_queue command_queue) {
  return real_clFinish(command_queue);
}

cl_int clReleaseCommandQueue (cl_command_queue command_queue) {
  return real_clReleaseCommandQueue (command_queue);

}
/* ************************************************************************* */

cl_program clCreateProgramWithSource (cl_context context,
                                      cl_uint count,
                                      const char **strings,
                                      const size_t *lengths,
                                      cl_int *errcode_ret)
{
  struct ld_program_s *program = get_next_program_spot();

  program->handle = real_clCreateProgramWithSource(context, count, strings,
                                                   lengths, errcode_ret);

  handle_program(program->handle, count, strings, lengths);
  
  return program->handle;
}

/* ************************************************************************* */

static inline ld_flags ocl_buffer_flags_to_ld (cl_mem_flags flags) {
  ld_flags ldFlags = 0;
  if (flags & CL_MEM_WRITE_ONLY) ldFlags |= LD_FLAG_WRITE_ONLY;
  if (flags & CL_MEM_READ_ONLY) ldFlags |= LD_FLAG_READ_ONLY;
  if (flags & CL_MEM_READ_WRITE) ldFlags |= LD_FLAG_READ_WRITE;

  return ldFlags;
}

struct ld_mem_s *create_ocl_buffer(cl_mem handle) {
  static unsigned int buffer_uid = -1;
  struct ld_mem_s *ldBuffer = get_next_mem_spot();

  ldBuffer->handle = handle;
  ldBuffer->uid = buffer_uid++;
  
  return ldBuffer;
}

cl_mem clCreateBuffer (cl_context context, cl_mem_flags flags, size_t size,
                       void *host_ptr, cl_int *errcode_ret)
{
  struct ld_mem_s *buffer;

  if (flags & CL_MEM_ALLOC_HOST_PTR) {
    goto unhandled;
  }
  
  buffer = create_ocl_buffer(real_clCreateBuffer(context, flags, size, host_ptr, errcode_ret));
  
  buffer->size = size;
  buffer->flags = ocl_buffer_flags_to_ld(flags);
  
  buffer_created_event(buffer);
  
  return buffer->handle;

unhandled:
  return real_clCreateBuffer(context, flags, size, host_ptr, errcode_ret);
}

/* ************************************************************************* */

cl_mem clCreateSubBuffer (cl_mem buffer,
                          cl_mem_flags flags,
                          cl_buffer_create_type buffer_create_type,
                          const void *buffer_create_info,
                          cl_int *errcode_ret)
{
  struct ld_mem_s *ldBuffer = find_mem_entry(buffer);
  struct ld_mem_offset_s *ldSubBuffer = NULL;
  cl_mem subbuffer;
  int i;
  
  assert(ldBuffer);
  
  subbuffer = real_clCreateSubBuffer(buffer, flags, buffer_create_type,
                                     buffer_create_info, errcode_ret);

  if (subbuffer == buffer) {
    return subbuffer;
  }
  for (i = 0; i < mem_offset_elt_count; i++) {
    if (mem_offset_map[i].released) {
      ldSubBuffer = &mem_offset_map[i];
    }                                                                 
  }

  if (!ldSubBuffer) {
    ldSubBuffer = get_next_mem_offset_spot();
  }
  
  assert(ldSubBuffer);

  ldSubBuffer->handle = subbuffer;
  ldSubBuffer->flags = ocl_buffer_flags_to_ld(flags);
  ldSubBuffer->offset = ((cl_buffer_region *) buffer_create_info)->origin;
  ldSubBuffer->size = ((cl_buffer_region*) buffer_create_info)->size;
  ldSubBuffer->parent = ldBuffer;
  ldSubBuffer->released = 0;
  
  subbuffer_created_event(ldBuffer, ldSubBuffer->offset);
  
  return subbuffer;
}

/* ************************************************************************* */

cl_int clReleaseMemObject (cl_mem memobj) {
  struct ld_mem_offset_s *ldSubBuffer = find_mem_offset_entry(memobj);
  struct ld_mem_s *ldBuffer = find_mem_entry(memobj);

  if (ldSubBuffer) {
    ldSubBuffer->released = 1;
  }

  if (ldBuffer) {
    buffer_released(ldBuffer);
  }
  
  //ignore other buffer types
  
  return real_clReleaseMemObject (memobj);
}

/* ************************************************************************* */

cl_kernel clCreateKernel (cl_program  program,
                          const char *kernel_name,
                          cl_int *errcode_ret)
{
  int nb_params, i;
  char **types_and_names;

  struct ld_kernel_s *ldKernel = get_next_kernel_spot();
 
  ldKernel->handle = real_clCreateKernel(program, kernel_name, errcode_ret);
  ldKernel->name = kernel_name;

  types_and_names = handle_create_kernel(program, ldKernel->handle, kernel_name);
  for (nb_params = 0; types_and_names[nb_params]; nb_params++);
  nb_params /= 2;

  ldKernel->nb_params = nb_params;
  ldKernel->params = malloc(sizeof(struct ld_kern_param_s) * nb_params);

  for (i = 0; i < nb_params; i++) {    
    ldKernel->params[i].name = types_and_names[i*2 + 1];
    ldKernel->params[i].type = types_and_names[i*2];
  }

  kernel_created_event(ldKernel);
  
  return ldKernel->handle;
}

/* ************************************************************************* */

int ocl_getBufferContent (struct ld_mem_s *ldBuffer, void *buffer,
                          size_t offset, size_t size)
{
  //debug("*** Read %zub from buffer #%d at +%zub *** \n", size, ldBuffer->uid, offset);
  cl_int err = real_clEnqueueReadBuffer(ldOclEnv.command_queue,
                                        ldBuffer->handle, CL_TRUE,
                                        offset, size, buffer,
                                        0, NULL, NULL);
  assert(err == CL_SUCCESS);
    
  return err == CL_SUCCESS;
}

cl_int clSetKernelArg (cl_kernel kernel,
                       cl_uint arg_index,
                       size_t arg_size,
                       const void *arg_value) {
  struct ld_kernel_s *ldKernel = find_kernel_entry(kernel);
  struct ld_kern_param_s *ldParam;

  assert(ldKernel);
  ldParam = &ldKernel->params[arg_index];
  
  if (ldParam->is_pointer) {
    struct ld_mem_s *ldBuffer = find_mem_entry(arg_value == NULL ? NULL : *(cl_mem *) arg_value);
    size_t offset = 0;
    
    if (!ldBuffer) {
      struct ld_mem_offset_s *ldSubBuffer = find_mem_offset_entry(*(cl_mem *) arg_value);

      assert(ldSubBuffer);
      ldBuffer = ldSubBuffer->parent;
      offset = ldSubBuffer->offset;
    }
    
    assert(ldBuffer);
    kernel_set_buffer_arg_event (ldKernel, ldParam, arg_index, ldBuffer, offset);
  } else {
    kernel_set_scalar_arg_event (ldKernel, ldParam, arg_index, (const void **) arg_value);
  }
  
  return real_clSetKernelArg(kernel, arg_index, arg_size, arg_value);
}

/* ************************************************************************* */

cl_int clEnqueueNDRangeKernel (cl_command_queue command_queue,
                               cl_kernel kernel,
                               cl_uint work_dim,
                               const size_t *global_work_offset,
                               const size_t *global_work_size,
                               const size_t *local_work_size,
                               cl_uint num_events_in_wait_list,
                               const cl_event *event_wait_list,
                               cl_event *event)
{
  static struct work_size_s work_sizes;
  
  struct ld_kernel_s *ldKernel = find_kernel_entry(kernel);
  int i;
  cl_int errcode;

  assert(ldKernel);
  for (i = 0; i < work_dim; i++) {
    work_sizes.local[i] = local_work_size[i];
    work_sizes.global[i] = global_work_size[i]/work_sizes.local[i];
  }
  
  kernel_executed_event(ldKernel, &work_sizes, work_dim);
  
  errcode = real_clEnqueueNDRangeKernel(command_queue, kernel, work_dim,
                                        global_work_offset, global_work_size,
                                        local_work_size, num_events_in_wait_list,
                                        event_wait_list, event);

#if FORCE_FINISH_KERNEL
  real_clFinish(command_queue);
#endif

  kernel_finished_event(ldKernel, &work_sizes, work_dim);
  
  return errcode;
}

/* ************************************************************************* */

struct ld_mem_s *readWriteMemory(cl_mem buffer, void **ptr, int direction,
                                 size_t size, size_t offset)
{
  struct ld_mem_s *ldBuffer = find_mem_entry(buffer);
  size_t real_offset = offset;
  
  if (!ldBuffer) {
    struct ld_mem_offset_s *ldSubBuffer = find_mem_offset_entry(buffer);

    assert(ldSubBuffer);
    ldBuffer = ldSubBuffer->parent;
    real_offset += ldSubBuffer->offset;
  }

  assert(ldBuffer);

  buffer_copy_event(ldBuffer, direction, (void **) ptr, size, real_offset);

  return ldBuffer;
}

cl_int clEnqueueWriteBuffer (cl_command_queue command_queue,
                             cl_mem buffer,
                             cl_bool blocking_write,
                             size_t offset,
                             size_t size,
                             const void *ptr,
                             cl_uint num_events_in_wait_list,
                             const cl_event *event_wait_list,
                             cl_event *event)
{
  readWriteMemory(buffer, (void **) ptr, LD_WRITE, size, offset);
  
  return real_clEnqueueWriteBuffer(command_queue, buffer, CL_TRUE /* blocking_write */,
                                   offset, size, ptr, num_events_in_wait_list,
                                   event_wait_list, event);
}

/* ************************************************************************* */

cl_int clEnqueueReadBuffer (cl_command_queue command_queue,
                            cl_mem buffer,
                            cl_bool blocking_read,
                            size_t offset,
                            size_t size,
                            void *ptr,
                            cl_uint num_events_in_wait_list,
                            const cl_event *event_wait_list,
                            cl_event *event)
{
  cl_int errcode;
  
  errcode = real_clEnqueueReadBuffer(command_queue, buffer, CL_TRUE /* blocking_read */,
                                     offset, size, ptr, num_events_in_wait_list,
                                     event_wait_list, event);

  readWriteMemory(buffer, ptr, LD_READ, size, offset);
  
  
  return errcode;
}
