# ibmhypregpu

A structured-grid 3D incompressible flow solver for the Schaefer–Turek channel benchmark with an immersed full-height cylinder, using HYPRE Struct solvers in CUDA device mode.

This code was developed to study a clean GPU-oriented path for immersed-boundary flow on a regular grid while keeping the implementation understandable and modular.

## What this code does

The solver computes 3D incompressible flow in a rectangular channel containing an immersed circular cylinder extruded through the full height. The flow is advanced in time using a projection-type approach:

- momentum solves for `u`, `v`, and `w`
- pressure Poisson solve
- velocity correction / projection

The immersed boundary is represented through cut-cell style geometric fractions and active masks on a structured grid. The sparse immersed operators are assembled once on the CPU, embedded into HYPRE Struct stencils, and then reused over many time steps.

## Current design

This version is intentionally a hybrid design:

- **CPU side**
  - immersed geometry construction
  - cut-cell / active-mask logic
  - sparse operator assembly for the immersed problem
  - one-time embedding of operators into Struct stencil format

- **GPU side**
  - HYPRE Struct solves in **device mode**
  - repeated dense momentum-side operations
  - repeated dense pressure/projection-side operations

So the code is **not full GPU assembly**, but it does use the GPU for the expensive repeated solves and dense timestep work.

## Main numerical ingredients

- structured Cartesian grid
- immersed full-height cylinder in a 3D channel
- explicit treatment of convection
- implicit diffusion through linear solves
- projection step for incompressibility
- HYPRE Struct solvers
  - momentum: GMRES + Jacobi
  - pressure: GMRES + PFMG
- reusable solver objects across time steps
- optional VTK output at user-defined frequency

## Why this code exists

The main goal is to keep the solver:

- understandable enough to study
- modular enough to extend
- close to GPU-friendly structured-grid workflows
- able to handle immersed geometry without switching to an unstructured mesh stack

It is a practical middle ground between:
- very lean regular structured solvers
- and much more complex fully unstructured immersed formulations

## Repository structure

A typical layout is:

- `main.cu`  
  Main driver, argument parsing, setup, timestep loop

- `common.hpp`  
  Common utilities, parameter structures, macros, indexing helpers

- `immersed_geometry.hpp`  
  Geometry fractions, active masks, immersed-cylinder construction

- `immersed_operators.hpp`  
  Sparse immersed operator assembly and row-major stencil embedding

- `hypre_struct_utils.hpp`  
  HYPRE Struct grid, stencil, matrix, vector, and solver helper routines

- `gpu_dense_ops.hpp`  
  CUDA kernels for repeated dense momentum and pressure/projection work

- `solver_helpers.hpp`  
  Glue logic for timestep operations, output, and helper routines

## Build

Set your PETSc/HYPRE/CUDA environment first:

```bash
export PETSC_DIR=/home/jd/src/petsc
export PETSC_ARCH=arch-linux-cuda-amgx-opt
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
hash -r

export HYPRE_INC="$PETSC_DIR/$PETSC_ARCH/include"
export HYPRE_LIB="$PETSC_DIR/$PETSC_ARCH/lib"
