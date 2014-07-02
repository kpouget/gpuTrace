CWD_DIR := $(shell pwd)

MPI_INC := -I/usr/include/openmpi-x86_64/

OCL_SO_NAME := ldChecker-ocl.so
CUDA_SO_NAME := ldChecker-cuda.so

PY_CFLAGS := $(python3-config --cflags) 
PY_LDFLAGS := $(python3-config --ldflags)

SO_CFLAGS := -fPIC 
SO_LDFLAGS := -fPIC -rdynamic -shared  $(PY_LDFLAGS) 

CFLAGS := -g -O0 -std=gnu99 -Werror -Wall -pedantic -Wno-format-security $(MPI_INC)

OCL_LD_PRELOAD_ENV := LD_PRELOAD=$(CWD_DIR)/$(OCL_SO_NAME):libpython3.3m.so
CUDA_LD_PRELOAD_ENV := LD_PRELOAD=$(CWD_DIR)/$(CUDA_SO_NAME)

all : $(OCL_SO_NAME) # $(CUDA_SO_NAME)

################
### OpenCL  ####
################

PYTHON_MOD_PATH=$(shell pwd)

$(OCL_SO_NAME) : instr-ocl.o ocl_helper_py.o ldChecker.o
	gcc -o $@ $^ $(SO_LDFLAGS)

instr-ocl.o : instr-ocl.c ocl_helper.h ldChecker.h
	gcc -o $@ -c $< $(CFLAGS) $(SO_CFLAGS)

ocl_helper_py.o : ocl_helper_py.c ocl_helper.h
	gcc -o $@ -c $< $(CFLAGS) $(PY_CFLAGS) $(SO_CFLAGS) -DPYTHON_MOD_PATH=$(PYTHON_MOD_PATH)

ldChecker.o : ldChecker.c ldChecker.h
	gcc -o $@ -c $< $(CFLAGS) $(SO_CFLAGS) -I/usr/lib/openmpi/include

clean : clean-cuda clean-python
	rm -fv *.o *.so

clean-python : 
	rm -rf __pycache__

MPI_RUN = #mpirun -n 6 -x 
### OCL example ###

SERGE_OCL_DIR := ~/travail/sample/OpenCL/Apriori_GPU/
SERGE_OCL_APPLI := ./AprioriPBI src/chess.dat 3000

SPECFEM_OCL_DIR := /home/kevin/final/async.ocl/ #/home/kevin/travail/sample/specfem-build
SPECFEM_OCL_APPLI := bin/xspecfem3D

OCL_DIR := $(SPECFEM_OCL_DIR)
OCL_APPLI := $(SPECFEM_OCL_APPLI)

run_ocl : $(OCL_SO_NAME)
	cd $(OCL_DIR) && $(MPI_RUN) $(OCL_LD_PRELOAD_ENV) $(OCL_APPLI) || echo "Failed"

debug_ocl : $(OCL_SO_NAME)
	cd $(OCL_DIR) && gdb -ex 'set environment $(OCL_LD_PRELOAD_ENV)' --args $(OCL_APPLI)

##############
### CUDA  ####
##############

CUDA_DIR := /home/kevin/final/async.cuda/
CUDA_APPLI := bin/xspecfem3D

CUDA_BIN := $(CUDA_DIR)$(CUDA_APPLI)

CUDA_INC := -I/usr/local/cuda-5.5/targets/x86_64-linux/include/ -I/usr/local/cuda-5.5/include

CUDA_CFLAGS := $(CUDA_INC)

NVCC := /usr/local/cuda/bin/nvcc --cudart=shared
INSTR_CFLAGS := -finstrument-functions 

NV_INSTR := --compiler-options $(INSTR_CFLAGS)
NV_TEMP_CFLAGS = --keep-dir=int --keep 

cuda_kernel_instruments_gen.c : preprocess.py $(CUDA_BIN)
	python3 $< $(CUDA_BIN) > $@
	sed -i 's/realw_const_p/const float */g' $@
	sed -i 's/realw_p/float \*/g' $@
	sed -i 's/realw\*/float \*/g' $@
	sed -i 's/realw/float/g' $@
	sed -i 's/int\*/int \*/g' $@
	sed -i 's/\* /\*/g' $@

$(CUDA_SO_NAME) : instr-cuda.c cuda_kernel_instruments_gen.c ldChecker.o 
	gcc -o $@ $< ldChecker.o $(CFLAGS) $(SO_CFLAGS) $(SO_LDFLAGS) $(CUDA_CFLAGS)

CUDA_EXAMPLE_CFLAGS := ${CFLAGS}
CUDA_EXAMPLE_CFLAGS := $(shell echo "${CUDA_EXAMPLE_CFLAGS}" | sed -e 's/-std=gnu99//g')
CUDA_EXAMPLE_CFLAGS := $(shell echo "${CUDA_EXAMPLE_CFLAGS}" | sed -e 's/-Werror//g')
CUDA_EXAMPLE_CFLAGS := $(shell echo "${CUDA_EXAMPLE_CFLAGS}" | sed -e 's/-Wall//g')
CUDA_EXAMPLE_CFLAGS := $(shell echo "${CUDA_EXAMPLE_CFLAGS}" | sed -e 's/-pedantic//g')

clean-cuda : clean-cuda-example
	rm -rf  *.i *.ii *.cudafe* *.fatbin* *.hash *.module_id *.o *.ptx *.cubin 

### Cuda example ###

hello : hello.cu
	$(NVCC) $(CUDA_EXAMPLE_CFLAGS) $< -o $@ $(NV_INSTR)

run_cuda : $(CUDA_SO_NAME)
	cd $(CUDA_DIR) && $(MPI_RUN) $(CUDA_LD_PRELOAD_ENV) $(CUDA_APPLI) || echo "Failed"

debug_cuda : $(CUDA_SO_NAME)
	cd $(CUDA_DIR) && gdb -ex 'set environment $(CUDA_LD_PRELOAD_ENV)' --args $(CUDA_APPLI)

clean-cuda-example :
	rm -rf hello hello.tmp int/

build-keep: hello.cu
	mkdir -p int
	$(NVCC) $(CUDA_EXAMPLE_CFLAGS) $< -o ./hello $(NV_INSTR) $(NV_TEMP_CFLAGS)


#################################

ocl: $(OCL_SO_NAME)

cuda: $(CUDA_SO_NAME)

all_files: file_cuda.out file_ocl.out

all_heads: head_cuda.out head_ocl.out

file_%.out: %
	make run_$< 1> $@ 2> $@.err

FL_DIGITS=3
head_%.out : file_%.out 
	head -5000 $< > $@
	sed -i '/setting cuda devices/d' $@
	sed -i '/image2d_t/d' $@
	sed -i '/Entering directory/d' $@
	sed -i '/LD_PRELOAD=/d' $@
	sed -i 's/,1>/>/g' $@
	sed -i 's/,1>/>/g' $@
	sed -i 's/0\.000000e+00, 0\.000000e+00, 0.*/0/g' $@

new_head_%.out : %
	touch file_$<.out
	make head_$<.out
