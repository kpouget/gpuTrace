

void init_helper(void);

char **handle_create_kernel(void *program, void *kernel, const char *name);

void handle_program(void *program,
                    unsigned int count,
                    const char **strings,
                    const size_t *lengths);
