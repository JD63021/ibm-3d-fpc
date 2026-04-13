__device__ static inline double dval_u(const double* u, int i, int j, int k, int nx, int ny, int nz, double fb){ if(i<0||i>nx||j<0||j>=ny||k<0||k>=nz) return fb; return u[idxU3(i,j,k,nx,ny)]; }
__device__ static inline double dval_v(const double* v, int i, int j, int k, int nx, int ny, int nz, double fb){ if(i<0||i>=nx||j<0||j>ny||k<0||k>=nz) return fb; return v[idxV3(i,j,k,nx,ny)]; }
__device__ static inline double dval_w(const double* w, int i, int j, int k, int nx, int ny, int nz, double fb){ if(i<0||i>=nx||j<0||j>=ny||k<0||k>nz) return fb; return w[idxW3(i,j,k,nx,ny)]; }

__global__ static void k_apply_bc_cleanup(double* u,double* v,double* w,const unsigned char* active_u,const unsigned char* active_v,const unsigned char* active_w,const double* uin_step,int nx,int ny,int nz){
  int tid=blockIdx.x*blockDim.x+threadIdx.x; int Nu=(nx+1)*ny*nz, Nv=nx*(ny+1)*nz, Nw=nx*ny*(nz+1); int N=max(Nu,max(Nv,Nw)); if(tid>=N) return;
  if(tid<Nu){ int plane=(nx+1)*ny; int k=tid/plane; int rem=tid-k*plane; int j=rem/(nx+1); int i=rem-j*(nx+1); if(i==0) u[tid]=uin_step[j+ny*k]; else if(i==nx) u[tid]=u[idxU3(nx-1,j,k,nx,ny)]; else if(!active_u[tid]) u[tid]=0.0; }
  if(tid<Nv){ int plane=nx*(ny+1); int k=tid/plane; int rem=tid-k*plane; int j=rem/nx; int i=rem-j*nx; if(j==0||j==ny) v[tid]=0.0; else if(!active_v[tid]) v[tid]=0.0; }
  if(tid<Nw){ int plane=nx*ny; int k=tid/plane; int rem=tid-k*plane; int j=rem/nx; int i=rem-j*nx; if(k==0||k==nz) w[tid]=0.0; else if(k>=1 && k<=nz-1 && !active_w[idxW3(i,j,k-1,nx,ny)]) w[tid]=0.0; }
}

__global__ static void k_conv_u(double* conv,const double* u,const double* v,const double* w,const unsigned char* active_u,int nx,int ny,int nz,double dx,double dy,double dz){ int tid=blockIdx.x*blockDim.x+threadIdx.x; int N=(nx+1)*ny*nz; if(tid>=N) return; int plane=(nx+1)*ny; int k=tid/plane; int rem=tid-k*plane; int j=rem/(nx+1); int i=rem-j*(nx+1); conv[tid]=0.0; if(i<1||i>nx-1) return; if(!active_u[tid]) return; double uc=u[tid]; double vadv=0.25*(dval_v(v,i-1,j,k,nx,ny,nz,0.0)+dval_v(v,i-1,j+1,k,nx,ny,nz,0.0)+dval_v(v,i,j,k,nx,ny,nz,0.0)+dval_v(v,i,j+1,k,nx,ny,nz,0.0)); double wadv=0.25*(dval_w(w,i-1,j,k,nx,ny,nz,0.0)+dval_w(w,i-1,j,k+1,nx,ny,nz,0.0)+dval_w(w,i,j,k,nx,ny,nz,0.0)+dval_w(w,i,j,k+1,nx,ny,nz,0.0)); double ue=dval_u(u,i+1,j,k,nx,ny,nz,uc), uw=dval_u(u,i-1,j,k,nx,ny,nz,uc), un=dval_u(u,i,j+1,k,nx,ny,nz,uc), us=dval_u(u,i,j-1,k,nx,ny,nz,uc), ut=dval_u(u,i,j,k+1,nx,ny,nz,uc), ub=dval_u(u,i,j,k-1,nx,ny,nz,uc); conv[tid]=uc*(ue-uw)/(2*dx)+vadv*(un-us)/(2*dy)+wadv*(ut-ub)/(2*dz); }
__global__ static void k_conv_v(double* conv,const double* u,const double* v,const double* w,const unsigned char* active_v,int nx,int ny,int nz,double dx,double dy,double dz){ int tid=blockIdx.x*blockDim.x+threadIdx.x; int N=nx*(ny+1)*nz; if(tid>=N) return; int plane=nx*(ny+1); int k=tid/plane; int rem=tid-k*plane; int j=rem/nx; int i=rem-j*nx; conv[tid]=0.0; if(j<1||j>ny-1) return; if(!active_v[tid]) return; double vc=v[tid]; double uadv=0.25*(dval_u(u,i,j-1,k,nx,ny,nz,0.0)+dval_u(u,i+1,j-1,k,nx,ny,nz,0.0)+dval_u(u,i,j,k,nx,ny,nz,0.0)+dval_u(u,i+1,j,k,nx,ny,nz,0.0)); double wadv=0.25*(dval_w(w,i,j-1,k,nx,ny,nz,0.0)+dval_w(w,i,j,k,nx,ny,nz,0.0)+dval_w(w,i,j-1,k+1,nx,ny,nz,0.0)+dval_w(w,i,j,k+1,nx,ny,nz,0.0)); double ve=dval_v(v,i+1,j,k,nx,ny,nz,vc), vw=dval_v(v,i-1,j,k,nx,ny,nz,vc), vn=dval_v(v,i,j+1,k,nx,ny,nz,vc), vs=dval_v(v,i,j-1,k,nx,ny,nz,vc), vt=dval_v(v,i,j,k+1,nx,ny,nz,vc), vb=dval_v(v,i,j,k-1,nx,ny,nz,vc); conv[tid]=uadv*(ve-vw)/(2*dx)+vc*(vn-vs)/(2*dy)+wadv*(vt-vb)/(2*dz); }
__global__ static void k_conv_w(double* conv,const double* u,const double* v,const double* w,const unsigned char* active_w,int nx,int ny,int nz,double dx,double dy,double dz){ int tid=blockIdx.x*blockDim.x+threadIdx.x; int N=nx*ny*(nz+1); if(tid>=N) return; int plane=nx*ny; int k=tid/plane; int rem=tid-k*plane; int j=rem/nx; int i=rem-j*nx; conv[tid]=0.0; if(k<1||k>nz-1) return; int active_idx=idxW3(i,j,k-1,nx,ny); if(!active_w[active_idx]) return; double wc=w[tid]; double uadv=0.25*(dval_u(u,i,j,k-1,nx,ny,nz,0.0)+dval_u(u,i+1,j,k-1,nx,ny,nz,0.0)+dval_u(u,i,j,k,nx,ny,nz,0.0)+dval_u(u,i+1,j,k,nx,ny,nz,0.0)); double vadv=0.25*(dval_v(v,i,j,k-1,nx,ny,nz,0.0)+dval_v(v,i,j+1,k-1,nx,ny,nz,0.0)+dval_v(v,i,j,k,nx,ny,nz,0.0)+dval_v(v,i,j+1,k,nx,ny,nz,0.0)); double we=dval_w(w,i+1,j,k,nx,ny,nz,wc), ww=dval_w(w,i-1,j,k,nx,ny,nz,wc), wn=dval_w(w,i,j+1,k,nx,ny,nz,wc), ws=dval_w(w,i,j-1,k,nx,ny,nz,wc), wt=dval_w(w,i,j,k+1,nx,ny,nz,wc), wb=dval_w(w,i,j,k-1,nx,ny,nz,wc); conv[tid]=uadv*(we-ww)/(2*dx)+vadv*(wn-ws)/(2*dy)+wc*(wt-wb)/(2*dz); }

__global__ static void k_build_u_full(HYPRE_Complex* rhsf,HYPRE_Complex* x0,const double* u,const double* conv,const double* gpx,const unsigned char* active_u,const int* id_u,const double* rhs_bc,const double* scale_u,int nx,int ny,int nz,double rho,double dt,int zero){ int tid=blockIdx.x*blockDim.x+threadIdx.x; int N=(nx-1)*ny*nz; if(tid>=N) return; int plane=(nx-1)*ny; int k=tid/plane; int rem=tid-k*plane; int j=rem/(nx-1); int iu=rem-j*(nx-1); int i=iu+1; int idx=idxU3(i,j,k,nx,ny); int id=id_u[idx]; double rhs=0.0; if(id>0 && active_u[idx]) rhs=((rho/dt)*u[idx]-rho*conv[idx]-gpx[idx]+rhs_bc[id])*scale_u[id]; rhsf[tid]=rhs; x0[tid]=zero?0.0:u[idx]; }
__global__ static void k_build_v_full(HYPRE_Complex* rhsf,HYPRE_Complex* x0,const double* v,const double* conv,const double* gpy,const unsigned char* active_v,const int* id_v,const double* rhs_bc,const double* scale_v,int nx,int ny,int nz,double rho,double dt,int zero){ int tid=blockIdx.x*blockDim.x+threadIdx.x; int N=nx*(ny-1)*nz; if(tid>=N) return; int plane=nx*(ny-1); int k=tid/plane; int rem=tid-k*plane; int jv=rem/nx; int i=rem-jv*nx; int j=jv+1; int idx=idxV3(i,j,k,nx,ny); int id=id_v[idx]; double rhs=0.0; if(id>0 && active_v[idx]) rhs=((rho/dt)*v[idx]-rho*conv[idx]-gpy[idx]+rhs_bc[id])*scale_v[id]; rhsf[tid]=rhs; x0[tid]=zero?0.0:v[idx]; }
__global__ static void k_build_w_full(HYPRE_Complex* rhsf,HYPRE_Complex* x0,const double* w,const double* conv,const double* gpz,const unsigned char* active_w,const int* id_w,const double* rhs_bc,const double* scale_w,int nx,int ny,int nz,double rho,double dt,int zero){ int tid=blockIdx.x*blockDim.x+threadIdx.x; int N=nx*ny*(nz-1); if(tid>=N) return; int plane=nx*ny; int kw=tid/plane; int rem=tid-kw*plane; int j=rem/nx; int i=rem-j*nx; int k=kw+1; int idx=idxW3(i,j,k,nx,ny); int id=id_w[idx]; double rhs=0.0; if(id>0 && active_w[idx]) rhs=((rho/dt)*w[idx]-rho*conv[idx]-gpz[idx]+rhs_bc[id])*scale_w[id]; rhsf[tid]=rhs; x0[tid]=zero?0.0:w[idx]; }

__global__ static void k_scatter_u(double* uo,const HYPRE_Complex* x,const double* ubase,int nx,int ny,int nz){ int tid=blockIdx.x*blockDim.x+threadIdx.x; int N=(nx+1)*ny*nz; if(tid>=N) return; if(ubase) uo[tid]=ubase[tid]; int plane=(nx+1)*ny; int k=tid/plane; int rem=tid-k*plane; int j=rem/(nx+1); int i=rem-j*(nx+1); if(i>=1 && i<=nx-1) { int row=(k*ny + j)*(nx-1) + (i-1); uo[tid]=x[row]; } }
__global__ static void k_scatter_v(double* vo,const HYPRE_Complex* x,const double* vbase,int nx,int ny,int nz){ int tid=blockIdx.x*blockDim.x+threadIdx.x; int N=nx*(ny+1)*nz; if(tid>=N) return; if(vbase) vo[tid]=vbase[tid]; int plane=nx*(ny+1); int k=tid/plane; int rem=tid-k*plane; int j=rem/nx; int i=rem-j*nx; if(j>=1 && j<=ny-1){ int row=(k*(ny-1)+ (j-1))*nx + i; vo[tid]=x[row]; } }
__global__ static void k_scatter_w(double* wo,const HYPRE_Complex* x,const double* wbase,int nx,int ny,int nz){ int tid=blockIdx.x*blockDim.x+threadIdx.x; int N=nx*ny*(nz+1); if(tid>=N) return; if(wbase) wo[tid]=wbase[tid]; int plane=nx*ny; int k=tid/plane; int rem=tid-k*plane; int j=rem/nx; int i=rem-j*nx; if(k>=1 && k<=nz-1){ int row=((k-1)*ny + j)*nx + i; wo[tid]=x[row]; } }

struct ManagedCSR {
  int nrows = 0;
  int nnz = 0;
  int *rowptr = nullptr;
  int *colind = nullptr; // 1-based active column ids
  double *vals = nullptr;
};

static ManagedCSR build_managed_csr_1based(const SparseMatrix &A)
{
  ManagedCSR M; M.nrows = A.nrows; M.nnz = nnz(A);
  MallocManagedI(&M.rowptr, (size_t)M.nrows + 1);
  MallocManagedI(&M.colind, (size_t)std::max(1, M.nnz));
  MallocManagedD(&M.vals, (size_t)std::max(1, M.nnz));
  int pos = 0; M.rowptr[0] = 0;
  for (int i = 0; i < A.nrows; ++i) {
    std::vector<std::pair<int,double>> row; row.reserve(A.rows[(size_t)i].size());
    for (const auto &kv : A.rows[(size_t)i]) row.push_back(kv);
    std::sort(row.begin(), row.end(), [](const auto &a, const auto &b){ return a.first < b.first; });
    for (const auto &kv : row) { M.colind[pos] = kv.first; M.vals[pos] = kv.second; ++pos; }
    M.rowptr[i+1] = pos;
  }
  CUDA_CALL(cudaDeviceSynchronize());
  return M;
}

static void destroy_managed_csr(ManagedCSR &M)
{
  if (M.rowptr) CUDA_CALL(cudaFree(M.rowptr));
  if (M.colind) CUDA_CALL(cudaFree(M.colind));
  if (M.vals) CUDA_CALL(cudaFree(M.vals));
  M = ManagedCSR{};
}

__global__ static void k_zero_d(double *x, int n){ int tid=blockIdx.x*blockDim.x+threadIdx.x; if(tid<n) x[tid]=0.0; }
__global__ static void k_zero_c(HYPRE_Complex *x, int n){ int tid=blockIdx.x*blockDim.x+threadIdx.x; if(tid<n) x[tid]=0.0; }

__global__ static void k_spmv_csr_1based(int nrows, const int *rowptr, const int *colind, const double *vals, const double *x, double *y)
{
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= nrows) return;
  double s = 0.0;
  for (int jj = rowptr[row]; jj < rowptr[row+1]; ++jj) s += vals[jj] * x[colind[jj]];
  y[row + 1] = s;
}

__global__ static void k_pack_pressure_active(double *vec, const double *p, const unsigned char *fluid_p, const int *id_p, int N)
{ int tid=blockIdx.x*blockDim.x+threadIdx.x; if(tid>=N) return; if(fluid_p[tid]) { int id=id_p[tid]; if(id>0) vec[id]=p[tid]; } }

__global__ static void k_unpack_gp_u(const double *gp, double *gpx, const int *id_qu, int nx, int ny, int nz)
{ int tid=blockIdx.x*blockDim.x+threadIdx.x; int N=(nx+1)*ny*nz; if(tid>=N) return; int id=id_qu[tid]; gpx[tid]=(id>0?gp[id]:0.0); }
__global__ static void k_unpack_gp_v(const double *gp, double *gpy, const int *id_qv, int nx, int ny, int nz, int nQu)
{ int tid=blockIdx.x*blockDim.x+threadIdx.x; int N=nx*(ny+1)*nz; if(tid>=N) return; int id=id_qv[tid]; gpy[tid]=(id>0?gp[nQu+id]:0.0); }
__global__ static void k_unpack_gp_w(const double *gp, double *gpz, const int *id_qw, int nx, int ny, int nz, int nQu, int nQv)
{ int tid=blockIdx.x*blockDim.x+threadIdx.x; int N=nx*ny*(nz+1); if(tid>=N) return; int id=id_qw[tid]; gpz[tid]=(id>0?gp[nQu+nQv+id]:0.0); }

__global__ static void k_pack_projection_q_u(double *q, const double *u, const int *id_qu, int nx, int ny, int nz)
{ int tid=blockIdx.x*blockDim.x+threadIdx.x; int N=(nx+1)*ny*nz; if(tid>=N) return; int id=id_qu[tid]; if(id>0) q[id]=u[tid]; }
__global__ static void k_pack_projection_q_v(double *q, const double *v, const int *id_qv, int nx, int ny, int nz, int nQu)
{ int tid=blockIdx.x*blockDim.x+threadIdx.x; int N=nx*(ny+1)*nz; if(tid>=N) return; int id=id_qv[tid]; if(id>0) q[nQu+id]=v[tid]; }
__global__ static void k_pack_projection_q_w(double *q, const double *w, const int *id_qw, int nx, int ny, int nz, int nQu, int nQv)
{ int tid=blockIdx.x*blockDim.x+threadIdx.x; int N=nx*ny*(nz+1); if(tid>=N) return; int id=id_qw[tid]; if(id>0) q[nQu+nQv+id]=w[tid]; }

__global__ static void k_build_pressure_rhs_full(HYPRE_Complex *rhsf, HYPRE_Complex *x0, const double *Dqt, const double *div_bc_unit, const unsigned char *fluid_p, const int *id_p, int N, double rho, double dt, double ramp)
{
  int tid=blockIdx.x*blockDim.x+threadIdx.x; if(tid>=N) return; double rhs=0.0; if(fluid_p[tid]) { int id=id_p[tid]; if(id>0) rhs = -(rho/dt) * (Dqt[id] + ramp * div_bc_unit[id]); } rhsf[tid]=rhs; x0[tid]=0.0;
}

__global__ static void k_pack_phi_active(double *phi, const HYPRE_Complex *xp_full, const unsigned char *fluid_p, const int *id_p, int N)
{ int tid=blockIdx.x*blockDim.x+threadIdx.x; if(tid>=N) return; if(fluid_p[tid]) { int id=id_p[tid]; if(id>0) phi[id]=xp_full[tid]; } }

__global__ static void k_apply_q_correction(double *q, const double *gphi, double factor, int nplus1)
{ int tid=blockIdx.x*blockDim.x+threadIdx.x; if(tid>=nplus1) return; if(tid>0) q[tid] -= factor * gphi[tid]; }

__global__ static void k_unpack_projection_q_u(const double *q, double *u, const int *id_qu, int nx, int ny, int nz)
{ int tid=blockIdx.x*blockDim.x+threadIdx.x; int N=(nx+1)*ny*nz; if(tid>=N) return; int id=id_qu[tid]; if(id>0) u[tid]=q[id]; }
__global__ static void k_unpack_projection_q_v(const double *q, double *v, const int *id_qv, int nx, int ny, int nz, int nQu)
{ int tid=blockIdx.x*blockDim.x+threadIdx.x; int N=nx*(ny+1)*nz; if(tid>=N) return; int id=id_qv[tid]; if(id>0) v[tid]=q[nQu+id]; }
__global__ static void k_unpack_projection_q_w(const double *q, double *w, const int *id_qw, int nx, int ny, int nz, int nQu, int nQv)
{ int tid=blockIdx.x*blockDim.x+threadIdx.x; int N=nx*ny*(nz+1); if(tid>=N) return; int id=id_qw[tid]; if(id>0) w[tid]=q[nQu+nQv+id]; }

__global__ static void k_add_pressure_phi(double *p, const double *phi, const unsigned char *fluid_p, const int *id_p, int N)
{ int tid=blockIdx.x*blockDim.x+threadIdx.x; if(tid>=N) return; if(fluid_p[tid]) { int id=id_p[tid]; if(id>0) p[tid] += phi[id]; } }
static void get_u_stencil_from_active_row(const SparseMatrix &Aactive, int aid, const std::vector<int> &flat_of_id_u,
                                          int nx, int ny, double c[7], int &unexpected)
{
