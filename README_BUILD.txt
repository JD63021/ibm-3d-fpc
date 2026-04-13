ibmhypregpu modular layout

Files
- common.hpp: includes, macros, Params, sparse helpers, indexing helpers
- immersed_geometry.hpp: cut-cell fractions, masks, index maps
- immersed_operators.hpp: host-side immersed operator assembly, embeddings, VTK helpers
- hypre_struct_utils.hpp: Struct grid/stencil/matrix/vector helpers
- gpu_dense_ops.hpp: CUDA dense kernels for momentum/pressure-side repeated work
- solver_helpers.hpp: audit and solver helper routines
- main.cu: app entry point
- build.sh: one-step build script

Manual compile:
  export PETSC_DIR=/home/jd/src/petsc
  export PETSC_ARCH=arch-linux-cuda-amgx-opt
  export CUDA_HOME=/usr/local/cuda
  export PATH="$CUDA_HOME/bin:$PATH"
  hash -r

  export HYPRE_INC="$PETSC_DIR/$PETSC_ARCH/include"
  export HYPRE_LIB="$PETSC_DIR/$PETSC_ARCH/lib"

  "$CUDA_HOME/bin/nvcc" -O3 -std=c++17 -ccbin mpicxx \
    -I"$HYPRE_INC" \
    -I"$CUDA_HOME/include" \
    -c main.cu \
    -o ibmhypregpu_app.o

  mpicxx ibmhypregpu_app.o \
    -O3 -std=c++17 \
    -L"$HYPRE_LIB" \
    -L"$CUDA_HOME/lib64" \
    -Wl,-rpath,"$HYPRE_LIB" \
    -Wl,-rpath,"$CUDA_HOME/lib64" \
    -lHYPRE -lcudart \
    -o ibmhypregpu_app

Run example:
  mpirun -n 1 ./ibmhypregpu_app \
    -nx 32 -ny 12 -nz 12 \
    -nu 1e-3 -Um 0.45 -dt 2e-3 \
    -nsteps 20 -print-every 1 \
    -device 0 -use-zero-initial-guess 1 \
    -vel-maxit 250 -vel-kdim 50 -vel-tol 1e-6 -vel-jacobi-iters 1 \
    -p-maxit 250 -p-kdim 50 -p-tol 1e-9 \
    -p-pc-maxit 1 -p-pc-relax-type 1 -p-pc-rap-type 1 \
    -write-vtk 1 -vtk-every 5 \
    -vtk st3d_channel_cylinder_gpu_pressure_dense.vtk
