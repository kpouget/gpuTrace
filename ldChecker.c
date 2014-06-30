#include <stdlib.h>
#include <stdarg.h>
#include <stdio.h>

#include "ldChecker.h"

#define __USE_GNU
#include <dlfcn.h>

#define SET_PARAM_FILE_NAME(__kern__, __param__, __out__, __str__)      \
  sprintf(__str__, "%s.%s.%s",                                          \
          __kern__->name, __param__->name,                              \
          __out__ ? "out" : "in")

#define MAX_LEN_ONE_NUMBER 16

struct type_info_s TYPE_FORMATTERS[] = {
  {"unsigned int", "%u", sizeof(unsigned int),  TYPE_INFO_UINT},
  {"int", "%d", sizeof(int), TYPE_INFO_INT},
  {"float", "%e", sizeof(float), TYPE_INFO_FLOAT},
  {"realw", "%e", sizeof(float), TYPE_INFO_FLOAT},
  {"double", "%lf", sizeof(double), TYPE_INFO_DOUBLE},
  {NULL, "%p", sizeof(void *),  TYPE_INFO_DEFAULT} /* default */
};

#if WITH_MPI == 1
#include <mpi.h>
int (*real_MPI_Comm_rank)(MPI_Comm comm, int *rank);
#endif

struct ld_bindings_s ld_bindings[] = {
#if WITH_MPI == 1
  {"MPI_Comm_rank", (void **) &real_MPI_Comm_rank},
#endif
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

#if WITH_MPI == 1
int myrank = -1;
int MPI_Comm_rank(MPI_Comm comm, int *rank) {
  local_init_ldchecker();
  
  int ret = real_MPI_Comm_rank (comm, rank);
  myrank = *rank;
  warning("Rank is %d\n", myrank);

  return ret;
}
#endif

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
#if ENABLE_KERNEL_PROFILING == 1
  ldKernel->exec_span_ns = 0;
#endif
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

  memcpy (ldParam->current_binary_value, arg_value, ldParam->type_info->size);
  
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

void print_scalar_param_to_file (struct ld_kernel_s *ldKernel,
                                 struct ld_kern_param_s *ldParam)
{
  static char filename[80];
  FILE *fp;
  
  SET_PARAM_FILE_NAME(ldKernel, ldParam, 0, filename);
  fp = fopen(filename, "w");
  if (!fp) {
    perror("Failed to open file:");
    error("Failure with file %s", filename);
  }

  fwrite (ldParam->current_binary_value, ldParam->type_info->size, 1, fp);

  fclose (fp);
}

static void print_kernel_problem_to_file(struct ld_kernel_s *ldKernel,
                                         const struct work_size_s *work_sizes,
                                         int work_dim)
{
  static char filename[80];
  FILE *fp;
  int i, j;
  
  sprintf(filename, "%s.problem_size", ldKernel->name);
  
  fp = fopen(filename, "w");
  if (!fp) {
    perror("Failed to open file:");
    error("Failure with file %s", filename);
  }
  
  for (i = 0; i < 2; i++) {
    const size_t *work_size = i == 0 ? work_sizes->local : work_sizes->global;
    
    fprintf(fp, "<");
    for (j = 0; j < work_dim; j++) {
      fprintf(fp, "%zu%s", work_size[j], j != work_dim - 1 ? "," : "");
    }
    fprintf(fp, ">");
  }

  fclose (fp);
}

void print_full_buffer(struct ld_kernel_s *ldKernel,
                       struct ld_kern_param_s *ldParam,
                       struct ld_mem_s *ldBuffer,
                       const struct type_info_s *type_info,
                       int finish)
{
  size_t size = ldBuffer->size; /* maybe get size of written data ?  */
  size_t tsize = type_info->size;
  size_t bytes_written = 0;
  
  void *buffer;
  char *ptr;

  if (ldBuffer->released || !ldBuffer->has_values) {
    if (finish) {
      return;
    }
    size = 0;
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

#if PRINT_KERNEL_PARAMS_TO_FILE == 1
  static char filename[80];
  SET_PARAM_FILE_NAME(ldKernel, ldParam, finish, filename);
  FILE *fp = fopen(filename, "w");
  if (!fp) {
    perror("Failed to open file:");
    error("Failure with file %s", filename);
  }
#else
  gpu_trace("\n");
#endif
  gpu_trace("\nWrite %s\n", filename);
  ptr = (char *) buffer;
  while (bytes_written < size) {
#if PRINT_KERNEL_PARAMS_TO_FILE == 1
    fwrite (ptr, tsize, 1, fp);
#else
    gpu_trace(print_a_number (ptr, type_info));
    gpu_trace(" ");
#endif
    
    ptr += tsize;
    bytes_written += tsize;
  }

#if PRINT_FULL_PARAMS_TO_FILE == 1
  fclose (fp);
#endif
  
finish:
  free(buffer);
}

#if PRINT_KERNEL_BEFORE_EXEC == 1 || PRINT_KERNEL_AFTER_EXEC == 1
#define SPACER "     "
static void kernel_print_current_parameters(struct ld_kernel_s *ldKernel,
                                            const struct work_size_s *work_sizes,
                                            int work_dim, int finish)
{
  int i, j;
  
#if FILTER_BY_KERNEL_EXEC_CPT == 1
  if (ldKernel->exec_counter > KERNEL_EXEC_CPT_UPPER_BOUND) {
    return;
  }
#endif

#if FILTER_BY_KERNEL_NAME == 1
  if (strstr(ldKernel->name, KERNEL_NAME_FILTER) == NULL) {
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

#if PRINT_KERNEL_ARG_FULL_BUFFER == 1 && PRINT_KERNEL_PARAMS_TO_FILE == 1
    print_kernel_problem_to_file(ldKernel, work_sizes, work_dim);
#endif
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
    if (finish
        && (!ldKernel->params[i].is_pointer // (it's a scalar)
            || strstr(ldKernel->params[i].type, "const ") != NULL // (it's a const buffer)
             ))
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
    
#if PRINT_KERNEL_ARG_FULL_BUFFER == 1
    if (ldKernel->params[i].is_pointer) {
      print_full_buffer(ldKernel, &ldKernel->params[i],
                        ldKernel->params[i].current_buffer,
                        ldKernel->params[i].type_info,
                        finish);
    }
#if PRINT_KERNEL_PARAMS_TO_FILE == 1
    else if (!finish){
      print_scalar_param_to_file (ldKernel, &ldKernel->params[i]);
    }
#endif
#endif
  }
  if (finish) {
    gpu_trace("\n);\n");
    exit(0);
  }
}
#endif

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

  size = FIRST_BYTES_TO_READ < size ? FIRST_BYTES_TO_READ : size;
  //need to pay attention to offset
  memcpy(ldBuffer->first_values, ptr, size);
  ldBuffer->has_values = 1;
  ldBuffer->values_outdated = 0;
  
#if PRINT_BUFFER_TRANSFER == 1
  {
    static int cpt = 0;
   
    gpu_trace("%d) Buffer #%d %s, %zub at +%zub", cpt++,
              ldBuffer->uid, is_read ? "read" : "written",
              size, offset);
#if PRINT_BUFFER_TRANSFER_FIRST_BYTES_AS_FLOAT
    gpu_trace(": {");
    int firsts = 4;
    int i;
    float *fptr = (float *) ptr;
    for (i = 0; i < firsts; i++) {
      if (sizeof(float) * i >= size) {
        continue;
      }
      gpu_trace("%e, ", fptr[i]);
    }
    
    gpu_trace("}");
#endif
    gpu_trace("\n");
  }
#endif
  
  if (offset + size > ldBuffer->size) {
    warning("%s too many bits from buffer #%d: %zub at +%zu, buffer is %zu\n",
            is_read ? "reading" : "writing", 
            ldBuffer->uid,
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
  ldBuffer->handle = NULL; // handles can be reused, so dont keep the old one
  ldBuffer->released = 1;
  ldBuffer->has_values = 0;
  ldBuffer->values_outdated = 0;
}

void debug(const char *format, ...) {
  va_list args;

  ONLY_MPI_MASTER();
  
  va_start(args, format);

  printf("DEBUG: ");
  vprintf(format, args);
  
  va_end(args);
}

void info(const char *format, ...) {
  va_list args;

  ONLY_MPI_MASTER();
  
  va_start(args, format);

  printf("INFO: ");
  vprintf(format, args);
  
  va_end(args);
}

void warning(const char *format, ...) {
  va_list args;

  ONLY_MPI_MASTER();
  
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

  exit(1);
}

void gpu_info(const char *format, ...) {
  va_list args;

  ONLY_MPI_MASTER();
  
  va_start(args, format);
  
  vfprintf(stdout, format, args);
  
  va_end(args);
}

void gpu_trace(const char *format, ...) {
  va_list args;

  ONLY_MPI_MASTER();
  
  va_start(args, format);
  
  vfprintf(stdout, format, args);
  
  va_end(args);
}
