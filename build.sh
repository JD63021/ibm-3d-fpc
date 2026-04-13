#!/usr/bin/env bash
set -euo pipefail

: "${PETSC_DIR:?Set PETSC_DIR}"
: "${PETSC_ARCH:?Set PETSC_ARCH}"
: "${CUDA_HOME:=/usr/local/cuda}"

export PATH="$CUDA_HOME/bin:$PATH"
hash -r

HYPRE_INC="$PETSC_DIR/$PETSC_ARCH/include"
HYPRE_LIB="$PETSC_DIR/$PETSC_ARCH/lib"

"$CUDA_HOME/bin/nvcc" -O3 -std=c++17 -ccbin mpicxx   -I"$HYPRE_INC"   -I"$CUDA_HOME/include"   -c main.cu   -o ibmhypregpu_app.o

mpicxx ibmhypregpu_app.o   -O3 -std=c++17   -L"$HYPRE_LIB"   -L"$CUDA_HOME/lib64"   -Wl,-rpath,"$HYPRE_LIB"   -Wl,-rpath,"$CUDA_HOME/lib64"   -lHYPRE -lcudart   -o ibmhypregpu_app
