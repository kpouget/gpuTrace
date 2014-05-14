#include <stdlib.h>
#include <stdarg.h>

#include "ldChecker.h"

#define __USE_GNU
#include <dlfcn.h>

#define MAX_LEN_ONE_NUMBER 16

#if ONLY_MPI_ROOT_OUTPUT == 1
#define CHECK_MPI_ROOT_OUTPUT()      \
  if (myrank != -1 && myrank != 0) { \
    return;                          \
  }
#else
#define CHECK_MPI_ROOT_OUTPUT()
#endif

struct type_info_s TYPE_FORMATTERS[] = {
  {"unsigned int", "%u", sizeof(unsigned int),  TYPE_INFO_UINT},
  {"int", "%d", sizeof(int), TYPE_INFO_INT},
  {"float", "%e", sizeof(float), TYPE_INFO_FLOAT},
  {"realw", "%e", sizeof(float), TYPE_INFO_FLOAT},
  {"double", "%lf", sizeof(double), TYPE_INFO_DOUBLE},
  {NULL, "%p", sizeof(void *),  TYPE_INFO_DEFAULT} /* default */
};

#include <mpi.h>
int (*real_MPI_Comm_rank)(MPI_Comm comm, int *rank);

struct ld_bindings_s ld_bindings[] = {
  {"MPI_Comm_rank", (void **) &real_MPI_Comm_rank},
  {NULL, NULL}
};

void dbg_crash_event(void) {}
void dbg_notify_event(void) {}

static struct callback_s _callbacks;

void init_bindings(struct ld_bindings_s *bindings) {
  int i = 0;
  while (bindings[i].name) {
    *bindings[i].real_address = dlsym(RTLD_NEXT, bindings[i].name);
    if (*bindings[i].real_address == NULL) {
      error("in `dlsym` of %s: %s\n", bindings[i].name, dlerror());
    }
    i++;
  }
}

static
void local_init_ldchecker(void) {
  static int inited = 0;
  if (inited) {
    return;
  }
  init_bindings(ld_bindings);
}

void init_ldchecker(struct callback_s callbacks, struct ld_bindings_s *lib_bindings) {
  static int inited = 0;
  
  if (inited) {
    return;
  }
  local_init_ldchecker();
  
  init_bindings(lib_bindings);
  
  _callbacks = callbacks;

  inited = 1;
}

int myrank = -1;
int MPI_Comm_rank(MPI_Comm comm, int *rank) {
  local_init_ldchecker();
  
  int ret = real_MPI_Comm_rank (comm, rank);
  myrank = *rank;
  warning("Rank is %d\n", myrank);

  return ret;
}


void subbuffer_created_event(struct ld_mem_s *buffer, size_t offset) {
#if PRINT_KERNEL_BUFFER_CREATION == 1
  info("Subbuffer created from buffer #%d, offset=%zu\n", buffer->uid, offset);
#endif
}

void buffer_created_event(struct ld_mem_s *ldBuffer) {
  int pow = 0;
  size_t h_size = ldBuffer->size;
  while (h_size > 1024) {
    h_size /= 1024;
    pow++;
    if (pow == 3) break;
  }

  ldBuffer->has_values = 0;
  ldBuffer->values_outdated = 0;
  ldBuffer->released = 0;
  
#if PRINT_KERNEL_BUFFER_CREATION
  {
    static char *H_UNITS[] = {"", "K", "M", "G"};
    info("New buffer #%d, %zu%sb, %s (%p)\n", ldBuffer->uid,
         h_size, H_UNITS[pow],
         buffer_flag_to_string(ldBuffer->flags),
         ldBuffer->handle);
  }
#endif
}

void kernel_created_event(struct ld_kernel_s *ldKernel) {
  int i;
#if PRINT_KERNEL_BUFFER_CREATION == 1
  info("New kernel: %s\n", ldKernel->name);
#endif
  for (i = 0; i < ldKernel->nb_params; i++) {
    const char *type = ldKernel->params[i].type;
    
    ldKernel->params[i].is_pointer = is_pointer_type(type);
    
    ldKernel->params[i].type_info = get_type_info(type);
    ldKernel->params[i].has_current_value = 0;
    ldKernel->params[i].current_buffer = NULL;
    ldKernel->params[i].offset = 0;
  }

  ldKernel->exec_counter = 0;
}


#define BVALUE_SIZE 400
static void kernel_set_arg_event (struct ld_kernel_s *ldKernel,
                                  struct ld_kern_param_s *ldParam,
                                  int arg_index)
{
  if (ldParam->has_current_value) {
    warning("current value already set (%s#%d)\n",
            ldKernel->name, arg_index);
  }
  ldParam->has_current_value = 1;

}

void setBufferValue (char *value, struct ld_kern_param_s *ldParam);

static
void arg_to_param_value (char *value, struct ld_kern_param_s *ldParam) {
  snprintf(ldParam->current_value, CURRENT_VALUE_BUFFER_SZ, "%s%s%s=<%s>",
           ldParam->type, ldParam->is_pointer ? "" : " ", 
           ldParam->name, value);
}

static
void buffer_to_param_value (struct ld_kern_param_s *ldParam) {
  static char value[BVALUE_SIZE];

  setBufferValue(value, ldParam);

  arg_to_param_value(value, ldParam);
}

void kernel_set_buffer_arg_event (struct ld_kernel_s *ldKernel,
                                  struct ld_kern_param_s *ldParam,
                                  int arg_index,
                                  struct ld_mem_s *ldBuffer,
                                  size_t offset)
{
  ldParam->current_buffer = ldBuffer;
  ldParam->offset = offset;
  
  buffer_to_param_value(ldParam);
  
  kernel_set_arg_event(ldKernel, ldParam, arg_index);
      
  if (!ldBuffer->flags & LD_FLAG_READ_ONLY) {
      ldBuffer->values_outdated = 1;
  }
}

void kernel_set_scalar_arg_event (struct ld_kernel_s *ldKernel,
                                  struct ld_kern_param_s *ldParam,
                                  int arg_index,
                                  const void **arg_value)
{
  static char value[BVALUE_SIZE];

  if (ldParam->type_info->type == TYPE_INFO_FLOAT) {
    const float float_value =  *(float *) arg_value;
    
    snprintf(value, BVALUE_SIZE, ldParam->type_info->format, float_value);
  } else {
    snprintf(value, BVALUE_SIZE, ldParam->type_info->format, *arg_value);
  }
  
  arg_to_param_value(value, ldParam);
  
  kernel_set_arg_event(ldKernel, ldParam, arg_index);
}

static
int updateLdBufferLocalValue (struct ld_mem_s *ldBuffer) {
#define MIN(a, b) (a < b ? a : b)
  size_t size =  MIN(ldBuffer->size, FIRST_BYTES_TO_READ) ;

  if (!ldBuffer->handle) {
    ldBuffer->has_values = 0;
    return 1;
  }
  
  return _callbacks.getBufferContent (ldBuffer, ldBuffer->first_values, 0, size);
}

static
char *print_a_number (const char *ptr, const struct type_info_s *type_info) {
  static char value[MAX_LEN_ONE_NUMBER];
  
  switch(type_info->type) {
  case TYPE_INFO_FLOAT:
    snprintf(value, MAX_LEN_ONE_NUMBER, type_info->format, *(float *) ptr);
    break;
  case TYPE_INFO_DOUBLE:
    snprintf(value, MAX_LEN_ONE_NUMBER, type_info->format, *(double *) ptr);
    break;
  case TYPE_INFO_INT:
    snprintf(value, MAX_LEN_ONE_NUMBER, type_info->format, *(int *) ptr);
    break;
  case TYPE_INFO_UINT:
    snprintf(value, MAX_LEN_ONE_NUMBER, type_info->format, *(unsigned int *) ptr);
    break;
  default:
    snprintf(value, MAX_LEN_ONE_NUMBER, type_info->format, *(void **) ptr);
  }

  return value;
}

void print_full_buffer(struct ld_mem_s *ldBuffer,
                       const struct type_info_s *type_info)
{
  size_t size = ldBuffer->size; /* maybe get size of written data ?  */
  size_t tsize = type_info->size;
  size_t bytes_written = 0;
  
  void *buffer;
  char *ptr;

  if (ldBuffer->released || !ldBuffer->has_values) {
    gpu_trace("(nothing relevant apparently)");
    return;
  }

#if defined(FULL_BUFFER_SIZE_LIMIT) && FULL_BUFFER_SIZE_LIMIT != 0
  if (size > FULL_BUFFER_SIZE_LIMIT) {
    size = FULL_BUFFER_SIZE_LIMIT;
  }
#endif

  buffer = malloc(size);
  if (!buffer) {
    warning("couldn't allocate a buffer of %zub to print "
            "the content of Buffer #%d\n", size, ldBuffer->uid);
  }
  
  if (!_callbacks.getBufferContent(ldBuffer, buffer, 0, size)) {
    warning("failed to retrieve the content of Buffer #%d\n",
            ldBuffer->uid);

    goto finish;
  }
  
  ptr = (char *) buffer;
  while (bytes_written < size) {
    gpu_trace(print_a_number (ptr, type_info));
    
    ptr += tsize;
    bytes_written += tsize;
    gpu_trace(" ");
  }

 finish:
  free(buffer);
}

#define SPACER "     "
static void kernel_print_current_parameters(struct ld_kernel_s *ldKernel,
                                            const struct work_size_s *work_sizes,
                                            int work_dim, int finish)
{
  int i, j;
  
#ifdef FILTER_BY_KERNEL_EXEC_CPT
  if (ldKernel->exec_counter >= FILTER_BY_KERNEL_EXEC_CPT) {
    return;
  }
#endif

#ifdef FILTER_BY_KERNEL_NAME
  if (strstr(ldKernel->name, FILTER_BY_KERNEL_NAME) == NULL) {
    return;
  }
#endif
  
  if (!finish) {
    gpu_trace("%d@%s", ldKernel->exec_counter, ldKernel->name);
    
    for (i = 0; i < 2; i++) {
      const size_t *work_size = i == 0 ? work_sizes->local : work_sizes->global;
    
      gpu_trace("<");
      for (j = 0; j < work_dim; j++) {
        gpu_trace("%zu%s", work_size[j], j != work_dim - 1 ? "," : "");
      }
      gpu_trace(">");
    }
    gpu_trace("(");
  }
#if PRINT_KERNEL_NAME_ONLY == 1
  if (!finish) {
    gpu_trace(");\n");
  }
  return;
#endif

  if (finish) {
    gpu_trace("\n%s----", SPACER);
  }
  
  for (i = 0; i < ldKernel->nb_params; i++) {
#if PRINT_KERNEL_AFTER_EXEC_IGNORE_CONST != 1
    if (!ldKernel->params[i].type_info->type_name)
      continue;
    if (finish
        && (strstr(ldKernel->params[i].type_info->type_name, "const ") == NULL
            || !ldKernel->params[i].is_pointer))
    {
      continue;
    }
#endif

    gpu_trace("\n%s", SPACER);
#define FORCE_REFRESH 1
    if (ldKernel->params[i].is_pointer && (FORCE_REFRESH || finish)) {
      updateLdBufferLocalValue(ldKernel->params[i].current_buffer);
      buffer_to_param_value(&ldKernel->params[i]);
    }
    if (finish) {
      gpu_trace("<out> ");
    }
    if (!ldKernel->params[i].has_current_value) {
      gpu_trace("<param #%d unset>", i);
    } else {
      gpu_trace(ldKernel->params[i].current_value);
    }
#if PRINT_KERNEL_ARG_FULL_BUFFER
    if (ldKernel->params[i].is_pointer) {
      gpu_trace("\n");
      print_full_buffer(ldKernel->params[i].current_buffer,
                        ldKernel->params[i].type_info);
    }
#endif
  }
  if (finish) {
    gpu_trace("\n);\n");
  }
}

void kernel_executed_event(struct ld_kernel_s *ldKernel,
                           const struct work_size_s *work_sizes,
                           int work_dim)
{
  int i;
  
  ldKernel->exec_counter++;
  
#if PRINT_KERNEL_BEFORE_EXEC == 1
  kernel_print_current_parameters(ldKernel, work_sizes, work_dim, 0);
#endif
  for (i = 0; i < ldKernel->nb_params; i++) {
    if (ldKernel->params[i].is_pointer)
      ldKernel->params[i].current_buffer->has_values = 1;
  }
}

void kernel_finished_event(struct ld_kernel_s *ldKernel,
                           const struct work_size_s *work_sizes,
                           int work_dim)
{
  int i;
  
#if PRINT_KERNEL_AFTER_EXEC == 1
  kernel_print_current_parameters(ldKernel, work_sizes, work_dim, 1);
#endif
  
  for (i = 0; i < ldKernel->nb_params; i++) {
    ldKernel->params[i].has_current_value = 0;
  }
  
}

void buffer_copy_event(struct ld_mem_s *ldBuffer, int is_read, void **ptr,
                       size_t size, size_t offset)
{
  if (!is_read && ldBuffer->flags & LD_FLAG_WRITE_ONLY) {
    warning("writing in write-only buffer#%d\n", ldBuffer->uid);
  }
  if (is_read && ldBuffer->flags & LD_FLAG_READ_ONLY) {
    warning("reading in read-only buffer #%d\n", ldBuffer->uid);
  }

  //need to pay attention to size and offset
  memcpy(ldBuffer->first_values, ptr, FIRST_BYTES_TO_READ);
  ldBuffer->has_values = 1;
  ldBuffer->values_outdated = 0;
  
#if PRINT_BUFFER_TRANSFER == 1
  {
    static int cpt = 0;
    float *fptr = (float *) ptr;
    int i;


    gpu_trace("%d) Buffer #%d %s, %zub at +%zub: ", cpt++,
              ldBuffer->uid, is_read ? "read" : "written",
              size, offset);
#if PRINT_BUFFER_TRANSFER_FIRST_BYTES_AS_FLOAT
    gpu_trace("{");
    int firsts = 4;
    for (i = 0; i < firsts; i++) {
      if (sizeof(float) * i >= size) {
        continue;
      }
      gpu_trace("%e, ", fptr[i]);
    }
    
    gpu_trace("}\n");
#endif
  }
#endif
  
  if (offset + size > ldBuffer->size) {
    warning("%s too many bits: %zub at +%zu, buffer is %zu\n",
            is_read ? "reading" : "writing",
            size, offset, ldBuffer->size);
  }
}

/** ** **/
void setBufferValue (char *value, struct ld_kern_param_s *ldParam)
{
  int to_write = BVALUE_SIZE;
  unsigned int bytes_written = 0;
  struct ld_mem_s *ldBuffer = ldParam->current_buffer;
  char *read_ptr = (char *) ldBuffer->first_values;

#define WRITE_PTR value + (BVALUE_SIZE - to_write)
  
  to_write -= snprintf(WRITE_PTR, to_write, "buffer #%d", ldBuffer->uid);

  if (ldParam->offset != 0) {
    to_write -= snprintf(WRITE_PTR, to_write, "+%zub", ldParam->offset);
  }

  to_write -= snprintf(WRITE_PTR, to_write, " ");

#if BUFFER_ZERO_IS_NULL
  /* not the best way to do, but at least it works with specfem */
  if (ldBuffer->uid <= 0) {
    snprintf(WRITE_PTR, to_write, "NULL");
    return;
  } else
#endif
  if (ldBuffer->released) {
    snprintf(WRITE_PTR, to_write, "released");
    return;
  } else if (!ldBuffer->has_values) {
    snprintf(WRITE_PTR, to_write, "no value");
    return;
  } else if (ldBuffer->values_outdated) {
    ldBuffer->values_outdated = !updateLdBufferLocalValue(ldBuffer);
    
    if (ldBuffer->values_outdated) {
      snprintf(WRITE_PTR, to_write, "outdated value");
      return;
    }
  }

#define TRAILER "..."
  while (bytes_written < FIRST_BYTES_TO_READ
         && bytes_written < ldBuffer->size)
  {
    char *number = print_a_number (read_ptr, ldParam->type_info);
    
    to_write -= snprintf(WRITE_PTR, to_write, "%s", number);

    if (to_write < strlen(TRAILER) + MAX_LEN_ONE_NUMBER) {
      break;
    }
    
    if (bytes_written < ldBuffer->size) {
      to_write -= snprintf(WRITE_PTR, to_write, ", ");
    }
    
    read_ptr += ldParam->type_info->size;
    bytes_written += ldParam->type_info->size;
  }
  
  if (bytes_written < ldBuffer->size) {
    snprintf(WRITE_PTR, to_write, "...");
  }
}

void buffer_released (struct ld_mem_s *ldBuffer) {
  if (!ldBuffer) {
    warning("releasing unknown buffer ...\n");
    return;
  }
  
#if PRINT_BUFFER_RELEASE
  info("Release buffer #%d\n", ldBuffer->uid);
#endif
  ldBuffer->handle = NULL;
  ldBuffer->released = 1;
  ldBuffer->has_values = 0;
  ldBuffer->values_outdated = 0;
}

void debug(const char *format, ...) {
  va_list args;

  CHECK_MPI_ROOT_OUTPUT();
  
  va_start(args, format);

  printf("DEBUG: ");
  vprintf(format, args);
  
  va_end(args);
}

void info(const char *format, ...) {
  va_list args;

  CHECK_MPI_ROOT_OUTPUT();
  
  va_start(args, format);

  printf("INFO: ");
  vprintf(format, args);
  
  va_end(args);
}

void warning(const char *format, ...) {
  va_list args;

  CHECK_MPI_ROOT_OUTPUT();
  
  va_start(args, format);

  printf("WARNING: ");
  vprintf(format, args);

  va_end(args);
}

void error(const char *format, ...) {
  va_list args;
  
  va_start(args, format);
  
  printf("ERROR: ");
  vprintf(format, args);

  va_end(args);
  
  dbg_crash_event();
  while(1);
  exit(1);
}

void gpu_info(const char *format, ...) {
  va_list args;

  CHECK_MPI_ROOT_OUTPUT();
  
  va_start(args, format);
  
  vfprintf(stdout, format, args);
  
  va_end(args);
}

void gpu_trace(const char *format, ...) {
  va_list args;

  CHECK_MPI_ROOT_OUTPUT();
  
  va_start(args, format);
  
  vfprintf(stdout, format, args);
  
  va_end(args);
}
