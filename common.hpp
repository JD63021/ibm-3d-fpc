#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <numeric>
#include <string>
#include <limits>

#include <mpi.h>
#include "HYPRE.h"
#include "HYPRE_struct_ls.h"
#include "HYPRE_krylov.h"
#include <cuda_runtime.h>

#define CUDA_CALL(call) do { \
  cudaError_t err__ = (call); \
  if (err__ != cudaSuccess) { \
    int rank__ = 0; MPI_Comm_rank(MPI_COMM_WORLD, &rank__); \
    std::fprintf(stderr, "[%d] CUDA ERROR %s:%d: %s\n", rank__, __FILE__, __LINE__, cudaGetErrorString(err__)); \
    MPI_Abort(MPI_COMM_WORLD, (int)err__); \
  } \
} while (0)

#define HYPRE_CALL(call) do { \
  HYPRE_Int ierr__ = (call); \
  if (ierr__) { \
    int rank__ = 0; MPI_Comm_rank(MPI_COMM_WORLD, &rank__); \
    std::fprintf(stderr, "[%d] HYPRE ERROR %s:%d code=%d\n", rank__, __FILE__, __LINE__, (int)ierr__); \
    MPI_Abort(MPI_COMM_WORLD, (int)ierr__); \
  } \
} while (0)

static void HYPRE_SOLVE_CALL(HYPRE_Int ierr, const char *what)
{
  if (!ierr) return;
  if (ierr == 256) { HYPRE_ClearError(HYPRE_ERROR_CONV); return; }
  int rank = 0; MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  std::fprintf(stderr, "[%d] HYPRE SOLVE ERROR in %s at %s:%d code=%d\n", rank, what, __FILE__, __LINE__, (int)ierr);
  MPI_Abort(MPI_COMM_WORLD, (int)ierr);
}

struct Params {
  int nx = 32;
  int ny = 12;
  int nz = 12;
  double Lx = 2.2;
  double Ly = 0.41;
  double Lz = 0.41;
  double rho = 1.0;
  double nu  = 1.0e-3;
  double Um  = 0.45;
  double dt  = 2.0e-3;
  int nsteps = 10;
  int print_every = 1;
  int inlet_ramp_steps = 200;
  int use_zero_initial_guess = 1;
  int device = 0;
  int nsub_beta = 10;
  double beta_tol  = 1.0e-8;
  double alpha_tol = 1.0e-10;
  double xcirc = 0.2;
  double ycirc = 0.2;
  double Rcirc = 0.05;

  int vel_maxit = 250;
  int vel_kdim = 50;
  double vel_tol = 1.0e-6;
  int vel_print = 0;
  int vel_jacobi_iters = 1;

  int p_maxit = 250;
  int p_kdim = 50;
  double p_tol = 1.0e-9;
  int p_print = 0;
  int p_pc_maxit = 1;
  int p_pc_relax_type = 1;
  int p_pc_rap_type = 1;

  int write_vtk = 1;
  int vtk_every = 0;
  std::string vtk_filename = "st3d_channel_cylinder_struct_mac_cuda.vtk";
};

static void ParseArgs(int argc, char **argv, Params &par)
{
  for (int a = 1; a < argc; ++a) {
    if (!std::strcmp(argv[a], "-nx") && a + 1 < argc) par.nx = std::atoi(argv[++a]);
    else if (!std::strcmp(argv[a], "-ny") && a + 1 < argc) par.ny = std::atoi(argv[++a]);
    else if (!std::strcmp(argv[a], "-nz") && a + 1 < argc) par.nz = std::atoi(argv[++a]);
    else if (!std::strcmp(argv[a], "-Lx") && a + 1 < argc) par.Lx = std::atof(argv[++a]);
    else if (!std::strcmp(argv[a], "-Ly") && a + 1 < argc) par.Ly = std::atof(argv[++a]);
    else if (!std::strcmp(argv[a], "-Lz") && a + 1 < argc) par.Lz = std::atof(argv[++a]);
    else if (!std::strcmp(argv[a], "-rho") && a + 1 < argc) par.rho = std::atof(argv[++a]);
    else if (!std::strcmp(argv[a], "-nu") && a + 1 < argc) par.nu = std::atof(argv[++a]);
    else if (!std::strcmp(argv[a], "-Um") && a + 1 < argc) par.Um = std::atof(argv[++a]);
    else if (!std::strcmp(argv[a], "-dt") && a + 1 < argc) par.dt = std::atof(argv[++a]);
    else if (!std::strcmp(argv[a], "-nsteps") && a + 1 < argc) par.nsteps = std::atoi(argv[++a]);
    else if (!std::strcmp(argv[a], "-print-every") && a + 1 < argc) par.print_every = std::atoi(argv[++a]);
    else if (!std::strcmp(argv[a], "-inlet-ramp-steps") && a + 1 < argc) par.inlet_ramp_steps = std::atoi(argv[++a]);
    else if (!std::strcmp(argv[a], "-use-zero-initial-guess") && a + 1 < argc) par.use_zero_initial_guess = std::atoi(argv[++a]);
    else if (!std::strcmp(argv[a], "-device") && a + 1 < argc) par.device = std::atoi(argv[++a]);
    else if (!std::strcmp(argv[a], "-nsub-beta") && a + 1 < argc) par.nsub_beta = std::atoi(argv[++a]);
    else if (!std::strcmp(argv[a], "-beta-tol") && a + 1 < argc) par.beta_tol = std::atof(argv[++a]);
    else if (!std::strcmp(argv[a], "-alpha-tol") && a + 1 < argc) par.alpha_tol = std::atof(argv[++a]);
    else if (!std::strcmp(argv[a], "-xcirc") && a + 1 < argc) par.xcirc = std::atof(argv[++a]);
    else if (!std::strcmp(argv[a], "-ycirc") && a + 1 < argc) par.ycirc = std::atof(argv[++a]);
    else if (!std::strcmp(argv[a], "-Rcirc") && a + 1 < argc) par.Rcirc = std::atof(argv[++a]);
    else if (!std::strcmp(argv[a], "-vel-maxit") && a + 1 < argc) par.vel_maxit = std::atoi(argv[++a]);
    else if (!std::strcmp(argv[a], "-vel-kdim") && a + 1 < argc) par.vel_kdim = std::atoi(argv[++a]);
    else if (!std::strcmp(argv[a], "-vel-tol") && a + 1 < argc) par.vel_tol = std::atof(argv[++a]);
    else if (!std::strcmp(argv[a], "-vel-print") && a + 1 < argc) par.vel_print = std::atoi(argv[++a]);
    else if (!std::strcmp(argv[a], "-vel-jacobi-iters") && a + 1 < argc) par.vel_jacobi_iters = std::atoi(argv[++a]);
    else if (!std::strcmp(argv[a], "-p-maxit") && a + 1 < argc) par.p_maxit = std::atoi(argv[++a]);
    else if (!std::strcmp(argv[a], "-p-kdim") && a + 1 < argc) par.p_kdim = std::atoi(argv[++a]);
    else if (!std::strcmp(argv[a], "-p-tol") && a + 1 < argc) par.p_tol = std::atof(argv[++a]);
    else if (!std::strcmp(argv[a], "-p-print") && a + 1 < argc) par.p_print = std::atoi(argv[++a]);
    else if (!std::strcmp(argv[a], "-p-pc-maxit") && a + 1 < argc) par.p_pc_maxit = std::atoi(argv[++a]);
    else if (!std::strcmp(argv[a], "-p-pc-relax-type") && a + 1 < argc) par.p_pc_relax_type = std::atoi(argv[++a]);
    else if (!std::strcmp(argv[a], "-p-pc-rap-type") && a + 1 < argc) par.p_pc_rap_type = std::atoi(argv[++a]);
    else if (!std::strcmp(argv[a], "-write-vtk") && a + 1 < argc) par.write_vtk = std::atoi(argv[++a]);
    else if (!std::strcmp(argv[a], "-vtk-every") && a + 1 < argc) par.vtk_every = std::atoi(argv[++a]);
    else if (!std::strcmp(argv[a], "-vtk") && a + 1 < argc) par.vtk_filename = argv[++a];
  }
}

static void print_device_info(int device)
{
  cudaDeviceProp prop;
  CUDA_CALL(cudaGetDeviceProperties(&prop, device));
  std::printf("GPU                   : %s\n", prop.name);
}

struct SparseMatrix {
  int nrows = 0;
  int ncols = 0;
  std::vector<std::unordered_map<int,double>> rows; // 1-based columns
  SparseMatrix() = default;
  SparseMatrix(int m, int n) : nrows(m), ncols(n), rows((size_t)m) {}
};

static void add_entry(SparseMatrix &A, int row, int col, double val)
{
  if (row < 1 || row > A.nrows || col < 1 || col > A.ncols) return;
  if (std::abs(val) < 1.0e-30) return;
  A.rows[(size_t)(row - 1)][col] += val;
}

static void compress_zeros(SparseMatrix &A, double tol = 1.0e-14)
{
  for (auto &r : A.rows) {
    for (auto it = r.begin(); it != r.end(); ) {
      if (std::abs(it->second) <= tol) it = r.erase(it);
      else ++it;
    }
  }
}

static int nnz(const SparseMatrix &A)
{
  long long s = 0;
  for (const auto &r : A.rows) s += (long long)r.size();
  return (int)s;
}

static int max_nnz_row(const SparseMatrix &A)
{
  int m = 0;
  for (const auto &r : A.rows) m = std::max(m, (int)r.size());
  return m;
}

static SparseMatrix multiply(const SparseMatrix &A, const SparseMatrix &B)
{
  SparseMatrix C(A.nrows, B.ncols);
  for (int i = 0; i < A.nrows; ++i) {
    for (const auto &akv : A.rows[(size_t)i]) {
      int k = akv.first;
      double a = akv.second;
      if (k < 1 || k > B.nrows) continue;
      const auto &Brow = B.rows[(size_t)(k - 1)];
      for (const auto &bkv : Brow) add_entry(C, i + 1, bkv.first, a * bkv.second);
    }
  }
  compress_zeros(C);
  return C;
}

static SparseMatrix subtract_sparse(const SparseMatrix &A, const SparseMatrix &B)
{
  SparseMatrix C = A;
  if (C.nrows == 0 && C.ncols == 0) C = SparseMatrix(B.nrows, B.ncols);
  for (int i = 1; i <= B.nrows; ++i)
    for (const auto &kv : B.rows[(size_t)(i - 1)]) add_entry(C, i, kv.first, -kv.second);
  compress_zeros(C);
  return C;
}

static SparseMatrix transpose_sparse(const SparseMatrix &A)
{
  SparseMatrix T(A.ncols, A.nrows);
  for (int i = 1; i <= A.nrows; ++i)
    for (const auto &kv : A.rows[(size_t)(i - 1)]) add_entry(T, kv.first, i, kv.second);
  compress_zeros(T);
  return T;
}

static double matrix_one_norm(const SparseMatrix &A)
{
  std::vector<double> cs((size_t)A.ncols + 1, 0.0);
  for (int i = 1; i <= A.nrows; ++i)
    for (const auto &kv : A.rows[(size_t)(i - 1)]) cs[(size_t)kv.first] += std::abs(kv.second);
  double m = 0.0; for (int j = 1; j <= A.ncols; ++j) m = std::max(m, cs[(size_t)j]);
  return m;
}

static double matrix_inf_norm(const SparseMatrix &A)
{
  double m = 0.0;
  for (int i = 1; i <= A.nrows; ++i) {
    double s = 0.0;
    for (const auto &kv : A.rows[(size_t)(i - 1)]) s += std::abs(kv.second);
    m = std::max(m, s);
  }
  return m;
}

static inline int id2_flat(int i1, int j1, int nx) { return i1 + (j1 - 1) * nx; }
static inline size_t idxP3(int i0, int j0, int k0, int nx, int ny) { return ((size_t)k0 * (size_t)ny + (size_t)j0) * (size_t)nx + (size_t)i0; }
__host__ __device__ static inline size_t idxU3(int i0, int j0, int k0, int nx, int ny) { return ((size_t)k0 * (size_t)ny + (size_t)j0) * (size_t)(nx + 1) + (size_t)i0; }
__host__ __device__ static inline size_t idxV3(int i0, int j0, int k0, int nx, int ny) { return ((size_t)k0 * (size_t)(ny + 1) + (size_t)j0) * (size_t)nx + (size_t)i0; }
__host__ __device__ static inline size_t idxW3(int i0, int j0, int k0, int nx, int ny) { return ((size_t)k0 * (size_t)ny + (size_t)j0) * (size_t)nx + (size_t)i0; }
static inline size_t idxUu(int iu0, int j0, int k0, int nxu, int ny) { return ((size_t)k0 * (size_t)ny + (size_t)j0) * (size_t)nxu + (size_t)iu0; }
static inline size_t idxVv(int i0, int j0, int k0, int nx, int nyv) { return ((size_t)k0 * (size_t)nyv + (size_t)j0) * (size_t)nx + (size_t)i0; }
static inline size_t idxWw(int i0, int j0, int kw0, int nx, int ny) { return ((size_t)kw0 * (size_t)ny + (size_t)j0) * (size_t)nx + (size_t)i0; }
static inline size_t idxPfull(int i0, int j0, int k0, int nx, int ny) { return ((size_t)k0 * (size_t)ny + (size_t)j0) * (size_t)nx + (size_t)i0; }

static inline double inletProfile(double y, double z, double H, double Um)
{
  return 16.0 * Um * y * z * (H - y) * (H - z) / std::pow(H, 4.0);
}

static inline bool fluid2_at(const std::vector<char> &a, int i1, int j1, int nx)
{
  return a[(size_t)id2_flat(i1, j1, nx) - 1] != 0;
}
static inline bool fluid3_at(const std::vector<char> &a, int i1, int j1, int k1, int nx, int ny)
{
  return a[idxP3(i1-1,j1-1,k1-1,nx,ny)] != 0;
}
static inline double alpha_u2_at(const std::vector<double> &a, int i1, int j1, int nx)
{
  return a[(size_t)(i1 - 1) + (size_t)(j1 - 1) * (size_t)(nx + 1)];
}
static inline double alpha_v2_at(const std::vector<double> &a, int i1, int j1, int nx)
{
  return a[(size_t)(i1 - 1) + (size_t)(j1 - 1) * (size_t)nx];
}

static double segmentCircleDistance(double xP, double yP, double xS, double yS, const Params &par)
{
  double dx = xS - xP, dy = yS - yP;
  double a = dx * dx + dy * dy;
  if (a <= 0.0) return std::numeric_limits<double>::infinity();
  double ox = xP - par.xcirc, oy = yP - par.ycirc;
  double b = 2.0 * (dx * ox + dy * oy);
  double c = ox * ox + oy * oy - par.Rcirc * par.Rcirc;
  double disc = b * b - 4.0 * a * c;
  if (disc < 0.0) return std::numeric_limits<double>::infinity();
  double sdisc = std::sqrt(std::max(0.0, disc));
  double t1 = (-b - sdisc) / (2.0 * a);
  double t2 = (-b + sdisc) / (2.0 * a);
  double t = std::numeric_limits<double>::infinity();
  if (t1 >= 0.0 && t1 <= 1.0) t = t1;
  else if (t2 >= 0.0 && t2 <= 1.0) t = t2;
  if (!std::isfinite(t)) return std::numeric_limits<double>::infinity();
  return std::sqrt(a) * t;
}

