static void build_u_diffusion_3d_immersed(
    SparseMatrix &A, std::vector<double> &rhs_u_bc_unit,
    const std::vector<char> &active_u, const std::vector<int> &id_u,
    const Params &par)
{
  int nx=par.nx, ny=par.ny, nz=par.nz;
  double dx=par.Lx/nx, dy=par.Ly/ny, dz=par.Lz/nz;
  double cx=par.nu/(dx*dx), cy=par.nu/(dy*dy), cz=par.nu/(dz*dz), at=par.rho/par.dt;
  int nU = *std::max_element(id_u.begin(), id_u.end());
  A = SparseMatrix(nU,nU);
  rhs_u_bc_unit.assign((size_t)nU+1, 0.0);
  for (int i=2;i<=nx;++i) {
    double xP = (i-1)*dx;
    for (int j=1;j<=ny;++j) {
      double yP = (j-0.5)*dy;
      for (int k=1;k<=nz;++k) {
        size_t idx = idxU3(i-1,j-1,k-1,nx,ny);
        if (!active_u[idx]) continue;
        int P = id_u[idx];
        double zP = (k-0.5)*dz;
        double diagP = at;
        // west
        if (i > 2 && active_u[idxU3(i-2,j-1,k-1,nx,ny)]) {
          int W = id_u[idxU3(i-2,j-1,k-1,nx,ny)]; add_entry(A,P,W,-cx); diagP += cx;
        } else {
          if (i == 2) { diagP += cx; rhs_u_bc_unit[(size_t)P] += cx * inletProfile(yP,zP,par.Ly,1.0); }
          else {
            double dP = segmentCircleDistance(xP,yP,xP-dx,yP,par); diagP += cx * (dx / dP);
          }
        }
        // east
        if (i < nx && active_u[idxU3(i,j-1,k-1,nx,ny)]) {
          int E = id_u[idxU3(i,j-1,k-1,nx,ny)]; add_entry(A,P,E,-cx); diagP += cx;
        } else {
          if (i != nx) { double dP = segmentCircleDistance(xP,yP,xP+dx,yP,par); diagP += cx * (dx / dP); }
        }
        // south
        if (j > 1 && active_u[idxU3(i-1,j-2,k-1,nx,ny)]) {
          int S = id_u[idxU3(i-1,j-2,k-1,nx,ny)]; add_entry(A,P,S,-cy); diagP += cy;
        } else {
          double alpha = (j == 1) ? dy / (0.5 * dy) : dy / segmentCircleDistance(xP,yP,xP,yP-dy,par); diagP += cy * alpha;
        }
        // north
        if (j < ny && active_u[idxU3(i-1,j,k-1,nx,ny)]) {
          int N = id_u[idxU3(i-1,j,k-1,nx,ny)]; add_entry(A,P,N,-cy); diagP += cy;
        } else {
          double alpha = (j == ny) ? dy / (0.5 * dy) : dy / segmentCircleDistance(xP,yP,xP,yP+dy,par); diagP += cy * alpha;
        }
        // bottom/top in z
        if (k > 1 && active_u[idxU3(i-1,j-1,k-2,nx,ny)]) {
          int B = id_u[idxU3(i-1,j-1,k-2,nx,ny)]; add_entry(A,P,B,-cz); diagP += cz;
        } else diagP += 2.0*cz;
        if (k < nz && active_u[idxU3(i-1,j-1,k,nx,ny)]) {
          int T = id_u[idxU3(i-1,j-1,k,nx,ny)]; add_entry(A,P,T,-cz); diagP += cz;
        } else diagP += 2.0*cz;

        add_entry(A,P,P,diagP);
      }
    }
  }
  compress_zeros(A);
}

static void build_v_diffusion_3d_immersed(
    SparseMatrix &A, std::vector<double> &rhs_v_bc,
    const std::vector<char> &active_v, const std::vector<int> &id_v,
    const Params &par)
{
  int nx=par.nx, ny=par.ny, nz=par.nz;
  double dx=par.Lx/nx, dy=par.Ly/ny, dz=par.Lz/nz;
  double cx=par.nu/(dx*dx), cy=par.nu/(dy*dy), cz=par.nu/(dz*dz), at=par.rho/par.dt;
  int nV = *std::max_element(id_v.begin(), id_v.end());
  A = SparseMatrix(nV,nV);
  rhs_v_bc.assign((size_t)nV+1, 0.0);
  for (int i=1;i<=nx;++i) {
    double xP = (i-0.5)*dx;
    for (int j=2;j<=ny;++j) {
      double yP = (j-1)*dy;
      for (int k=1;k<=nz;++k) {
        size_t idx = idxV3(i-1,j-1,k-1,nx,ny);
        if (!active_v[idx]) continue;
        int P = id_v[idx];
        double diagP = at;
        // west
        if (i > 1 && active_v[idxV3(i-2,j-1,k-1,nx,ny)]) {
          int W = id_v[idxV3(i-2,j-1,k-1,nx,ny)]; add_entry(A,P,W,-cx); diagP += cx;
        } else {
          double alpha = (i == 1) ? dx / (0.5 * dx) : dx / segmentCircleDistance(xP,yP,xP-dx,yP,par); diagP += cx * alpha;
        }
        // east
        if (i < nx && active_v[idxV3(i,j-1,k-1,nx,ny)]) {
          int E = id_v[idxV3(i,j-1,k-1,nx,ny)]; add_entry(A,P,E,-cx); diagP += cx;
        } else {
          if (i != nx) { double alpha = dx / segmentCircleDistance(xP,yP,xP+dx,yP,par); diagP += cx * alpha; }
        }
        // south
        if (j > 2 && active_v[idxV3(i-1,j-2,k-1,nx,ny)]) {
          int S = id_v[idxV3(i-1,j-2,k-1,nx,ny)]; add_entry(A,P,S,-cy); diagP += cy;
        } else {
          double alpha = (j == 2) ? dy / dy : dy / segmentCircleDistance(xP,yP,xP,yP-dy,par); diagP += cy * alpha;
        }
        // north
        if (j < ny && active_v[idxV3(i-1,j,k-1,nx,ny)]) {
          int N = id_v[idxV3(i-1,j,k-1,nx,ny)]; add_entry(A,P,N,-cy); diagP += cy;
        } else {
          double alpha = (j == ny) ? dy / dy : dy / segmentCircleDistance(xP,yP,xP,yP+dy,par); diagP += cy * alpha;
        }
        // z bottom/top walls
        if (k > 1 && active_v[idxV3(i-1,j-1,k-2,nx,ny)]) {
          int B = id_v[idxV3(i-1,j-1,k-2,nx,ny)]; add_entry(A,P,B,-cz); diagP += cz;
        } else diagP += 2.0*cz;
        if (k < nz && active_v[idxV3(i-1,j-1,k,nx,ny)]) {
          int T = id_v[idxV3(i-1,j-1,k,nx,ny)]; add_entry(A,P,T,-cz); diagP += cz;
        } else diagP += 2.0*cz;

        add_entry(A,P,P,diagP);
      }
    }
  }
  compress_zeros(A);
}

static void build_w_diffusion_3d_immersed(
    SparseMatrix &A, std::vector<double> &rhs_w_bc,
    const std::vector<char> &active_w, const std::vector<int> &id_w,
    const Params &par)
{
  int nx=par.nx, ny=par.ny, nz=par.nz;
  double dx=par.Lx/nx, dy=par.Ly/ny, dz=par.Lz/nz;
  double cx=par.nu/(dx*dx), cy=par.nu/(dy*dy), cz=par.nu/(dz*dz), at=par.rho/par.dt;
  int nW = *std::max_element(id_w.begin(), id_w.end());
  A = SparseMatrix(nW,nW);
  rhs_w_bc.assign((size_t)nW+1, 0.0);
  for (int i=1;i<=nx;++i) {
    double xP = (i-0.5)*dx;
    for (int j=1;j<=ny;++j) {
      double yP = (j-0.5)*dy;
      for (int k=2;k<=nz;++k) {
        size_t idx = idxW3(i-1,j-1,k-1,nx,ny);
        if (!active_w[idx]) continue;
        int P = id_w[idx];
        double diagP = at;
        // west/east with x-y ghosts
        if (i > 1 && active_w[idxW3(i-2,j-1,k-1,nx,ny)]) {
          int W = id_w[idxW3(i-2,j-1,k-1,nx,ny)]; add_entry(A,P,W,-cx); diagP += cx;
        } else {
          double alpha = (i == 1) ? dx / (0.5 * dx) : dx / segmentCircleDistance(xP,yP,xP-dx,yP,par); diagP += cx * alpha;
        }
        if (i < nx && active_w[idxW3(i,j-1,k-1,nx,ny)]) {
          int E = id_w[idxW3(i,j-1,k-1,nx,ny)]; add_entry(A,P,E,-cx); diagP += cx;
        } else {
          if (i != nx) { double alpha = dx / segmentCircleDistance(xP,yP,xP+dx,yP,par); diagP += cx * alpha; }
        }
        if (j > 1 && active_w[idxW3(i-1,j-2,k-1,nx,ny)]) {
          int S = id_w[idxW3(i-1,j-2,k-1,nx,ny)]; add_entry(A,P,S,-cy); diagP += cy;
        } else {
          double alpha = (j == 1) ? dy / (0.5 * dy) : dy / segmentCircleDistance(xP,yP,xP,yP-dy,par); diagP += cy * alpha;
        }
        if (j < ny && active_w[idxW3(i-1,j,k-1,nx,ny)]) {
          int N = id_w[idxW3(i-1,j,k-1,nx,ny)]; add_entry(A,P,N,-cy); diagP += cy;
        } else {
          double alpha = (j == ny) ? dy / (0.5 * dy) : dy / segmentCircleDistance(xP,yP,xP,yP+dy,par); diagP += cy * alpha;
        }
        // z boundaries, w lives on z-faces so k=2..nz with wall Dirichlet at k=1,nz+1
        if (k > 2 && active_w[idxW3(i-1,j-1,k-2,nx,ny)]) {
          int B = id_w[idxW3(i-1,j-1,k-2,nx,ny)]; add_entry(A,P,B,-cz); diagP += cz;
        } else diagP += cz;
        if (k < nz && active_w[idxW3(i-1,j-1,k,nx,ny)]) {
          int T = id_w[idxW3(i-1,j-1,k,nx,ny)]; add_entry(A,P,T,-cz); diagP += cz;
        } else diagP += cz;

        add_entry(A,P,P,diagP);
      }
    }
  }
  compress_zeros(A);
}

static void build_projection_operators_3d_channel_cylinder(
    SparseMatrix &Dproj, SparseMatrix &Gproj, SparseMatrix &Ap, std::vector<double> &div_bc_unit,
    const std::vector<double> &beta_p,
    const std::vector<double> &alpha_u,
    const std::vector<double> &alpha_v,
    const std::vector<double> &alpha_w,
    const std::vector<char> &fluid_p,
    const std::vector<char> &active_qu,
    const std::vector<char> &active_qv,
    const std::vector<char> &active_qw,
    const std::vector<int> &id_p,
    const std::vector<int> &id_qu,
    const std::vector<int> &id_qv,
    const std::vector<int> &id_qw,
    const Params &par)
{
  int nx=par.nx, ny=par.ny, nz=par.nz;
  double dx=par.Lx/nx, dy=par.Ly/ny, dz=par.Lz/nz;
  int nP = *std::max_element(id_p.begin(), id_p.end());
  int nQu = *std::max_element(id_qu.begin(), id_qu.end());
  int nQv = *std::max_element(id_qv.begin(), id_qv.end());
  int nQw = *std::max_element(id_qw.begin(), id_qw.end());
  int nF = nQu + nQv + nQw;
  Dproj = SparseMatrix(nP,nF);
  Gproj = SparseMatrix(nF,nP);
  div_bc_unit.assign((size_t)nP+1, 0.0);

  // D row per fluid cell
  for (int i=1;i<=nx;++i) {
    for (int j=1;j<=ny;++j) {
      for (int k=1;k<=nz;++k) {
        if (!fluid3_at(fluid_p,i,j,k,nx,ny)) continue;
        int row = id_p[idxP3(i-1,j-1,k-1,nx,ny)];
        double beta = beta_p[idxP3(i-1,j-1,k-1,nx,ny)];
        // east x-face
        if (i < nx) {
          size_t qe = idxU3(i,j-1,k-1,nx,ny);
          if (active_qu[qe]) add_entry(Dproj,row,id_qu[qe], alpha_u[qe] / (beta * dx));
        } else {
          size_t qe = idxU3(nx,j-1,k-1,nx,ny);
          if (active_qu[qe]) add_entry(Dproj,row,id_qu[qe], alpha_u[qe] / (beta * dx));
        }
        // west x-face or inlet BC
        if (i > 1) {
          size_t qw = idxU3(i-1,j-1,k-1,nx,ny);
          if (active_qu[qw]) add_entry(Dproj,row,id_qu[qw], -alpha_u[qw] / (beta * dx));
        } else {
          double y = (j - 0.5) * dy, z = (k - 0.5) * dz;
          div_bc_unit[(size_t)row] -= inletProfile(y,z,par.Ly,par.Um) / (beta * dx);
        }
        // north/south y-faces
        if (j < ny) {
          size_t qn = idxV3(i-1,j,k-1,nx,ny);
          if (active_qv[qn]) add_entry(Dproj,row,nQu + id_qv[qn], alpha_v[qn] / (beta * dy));
        }
        if (j > 1) {
          size_t qs = idxV3(i-1,j-1,k-1,nx,ny);
          if (active_qv[qs]) add_entry(Dproj,row,nQu + id_qv[qs], -alpha_v[qs] / (beta * dy));
        }
        // top/bottom z-faces
        if (k < nz) {
          size_t qt = idxW3(i-1,j-1,k,nx,ny);
          if (active_qw[qt]) add_entry(Dproj,row,nQu + nQv + id_qw[qt], alpha_w[qt] / (beta * dz));
        }
        if (k > 1) {
          size_t qb = idxW3(i-1,j-1,k-1,nx,ny);
          if (active_qw[qb]) add_entry(Dproj,row,nQu + nQv + id_qw[qb], -alpha_w[qb] / (beta * dz));
        }
      }
    }
  }

  // G rows
  for (int i=2;i<=nx;++i) for (int j=1;j<=ny;++j) for (int k=1;k<=nz;++k) {
    size_t q = idxU3(i-1,j-1,k-1,nx,ny);
    if (!active_qu[q]) continue;
    int row = id_qu[q];
    int pL = id_p[idxP3(i-2,j-1,k-1,nx,ny)];
    int pR = id_p[idxP3(i-1,j-1,k-1,nx,ny)];
    add_entry(Gproj,row,pL,-1.0/dx); add_entry(Gproj,row,pR,+1.0/dx);
  }
  for (int j=1;j<=ny;++j) for (int k=1;k<=nz;++k) {
    size_t q = idxU3(nx,j-1,k-1,nx,ny);
    if (!active_qu[q]) continue;
    int row = id_qu[q];
    int pL = id_p[idxP3(nx-1,j-1,k-1,nx,ny)];
    add_entry(Gproj,row,pL,-2.0/dx);
  }
  for (int i=1;i<=nx;++i) for (int j=2;j<=ny;++j) for (int k=1;k<=nz;++k) {
    size_t q = idxV3(i-1,j-1,k-1,nx,ny);
    if (!active_qv[q]) continue;
    int row = nQu + id_qv[q];
    int pS = id_p[idxP3(i-1,j-2,k-1,nx,ny)];
    int pN = id_p[idxP3(i-1,j-1,k-1,nx,ny)];
    add_entry(Gproj,row,pS,-1.0/dy); add_entry(Gproj,row,pN,+1.0/dy);
  }
  for (int i=1;i<=nx;++i) for (int j=1;j<=ny;++j) for (int k=2;k<=nz;++k) {
    size_t q = idxW3(i-1,j-1,k-1,nx,ny);
    if (!active_qw[q]) continue;
    int row = nQu + nQv + id_qw[q];
    int pB = id_p[idxP3(i-1,j-1,k-2,nx,ny)];
    int pT = id_p[idxP3(i-1,j-1,k-1,nx,ny)];
    add_entry(Gproj,row,pB,-1.0/dz); add_entry(Gproj,row,pT,+1.0/dz);
  }

  compress_zeros(Dproj);
  compress_zeros(Gproj);
  Ap = multiply(Dproj,Gproj);
}

static void invert_id_map(const std::vector<int> &idmap, std::vector<int> &coords, int tuple_size)
{
  int n = *std::max_element(idmap.begin(), idmap.end());
  coords.assign((size_t)(n+1) * (size_t)tuple_size, 0);
  for (size_t idx=0; idx<idmap.size(); ++idx) {
    int id = idmap[idx];
    if (!id) continue;
    coords[(size_t)id*tuple_size + 0] = (int)idx; // caller decodes flat idx
  }
}

static void print_generic_row(const SparseMatrix &A, int row)
{
  std::vector<std::pair<int,double>> kvs;
  for (const auto &kv : A.rows[(size_t)(row-1)]) kvs.push_back(kv);
  std::sort(kvs.begin(), kvs.end(), [](auto &a, auto &b){ return a.first < b.first; });
  std::printf("  row id = %d  nnz = %d\n", row, (int)kvs.size());
  for (const auto &kv : kvs)
    std::printf("    col %6d   val = %+.16e\n", kv.first, kv.second);
}

static void decode_p(const std::vector<int> &id_p, int target, int nx, int ny, int &i, int &j, int &k)
{
  for (k=1;k<=1000000;++k) { if (k>ny*nx*1000000) break; }
  int nz_guess = (int)(id_p.size() / ((size_t)nx * (size_t)ny));
  for (int kk=1; kk<=nz_guess; ++kk) for (int jj=1; jj<=ny; ++jj) for (int ii=1; ii<=nx; ++ii) {
    if (id_p[idxP3(ii-1,jj-1,kk-1,nx,ny)] == target) { i=ii; j=jj; k=kk; return; }
  }
  i=j=k=-1;
}

static void decode_u(const std::vector<int> &id_u, int target, int nx, int ny, int nz, int &i, int &j, int &k)
{
  for (int kk=1; kk<=nz; ++kk) for (int jj=1; jj<=ny; ++jj) for (int ii=1; ii<=nx+1; ++ii)
    if (id_u[idxU3(ii-1,jj-1,kk-1,nx,ny)] == target) { i=ii;j=jj;k=kk; return; }
  i=j=k=-1;
}
static void decode_v(const std::vector<int> &id_v, int target, int nx, int ny, int nz, int &i, int &j, int &k)
{
  for (int kk=1; kk<=nz; ++kk) for (int jj=1; jj<=ny+1; ++jj) for (int ii=1; ii<=nx; ++ii)
    if (id_v[idxV3(ii-1,jj-1,kk-1,nx,ny)] == target) { i=ii;j=jj;k=kk; return; }
  i=j=k=-1;
}
static void decode_w(const std::vector<int> &id_w, int target, int nx, int ny, int nz, int &i, int &j, int &k)
{
  for (int kk=1; kk<=nz+1; ++kk) for (int jj=1; jj<=ny; ++jj) for (int ii=1; ii<=nx; ++ii)
    if (id_w[idxW3(ii-1,jj-1,kk-1,nx,ny)] == target) { i=ii;j=jj;k=kk; return; }
  i=j=k=-1;
}

static void print_pressure_row(const SparseMatrix &A, int row, const std::vector<int> &id_p, int nx, int ny, int nz, int ip, int jp, int kp)
{
  std::vector<std::pair<int,double>> kvs;
  for (const auto &kv : A.rows[(size_t)(row-1)]) kvs.push_back(kv);
  std::sort(kvs.begin(), kvs.end(), [](auto &a, auto &b){ return a.first < b.first; });
  std::printf("  row id = %d -> p(%d,%d,%d)  nnz = %d\n", row, ip, jp, kp, (int)kvs.size());
  for (const auto &kv : kvs) {
    int ic,jc,kc; decode_p(id_p, kv.first, nx, ny, ic,jc,kc);
    std::printf("    col %6d -> p(%d,%d,%d)   delta=(%+d,%+d,%+d)   val = %+.16e\n",
                kv.first, ic,jc,kc, ic-ip, jc-jp, kc-kp, kv.second);
  }
}


static std::vector<double> spmv(const SparseMatrix &A, const std::vector<double> &x)
{
  std::vector<double> y((size_t)A.nrows + 1, 0.0);
  for (int i = 1; i <= A.nrows; ++i) {
    double sum = 0.0;
    for (const auto &kv : A.rows[(size_t)(i - 1)]) sum += kv.second * x[(size_t)kv.first];
    y[(size_t)i] = sum;
  }
  return y;
}

static void build_inverse_coords_u(const std::vector<int>& id_u, int nx, int ny, int nz, std::vector<int>& iu, std::vector<int>& ju, std::vector<int>& ku)
{
  int nU = *std::max_element(id_u.begin(), id_u.end());
  iu.assign((size_t)nU+1,0); ju.assign((size_t)nU+1,0); ku.assign((size_t)nU+1,0);
  for (int k=1;k<=nz;++k) for (int j=1;j<=ny;++j) for (int i=2;i<=nx;++i) {
    int id=id_u[idxU3(i-1,j-1,k-1,nx,ny)]; if(id){ iu[id]=i-1; ju[id]=j-1; ku[id]=k-1; }
  }
}
static void build_inverse_coords_v(const std::vector<int>& id_v, int nx, int ny, int nz, std::vector<int>& iv, std::vector<int>& jv, std::vector<int>& kv)
{
  int nV = *std::max_element(id_v.begin(), id_v.end());
  iv.assign((size_t)nV+1,0); jv.assign((size_t)nV+1,0); kv.assign((size_t)nV+1,0);
  for (int k=1;k<=nz;++k) for (int j=2;j<=ny;++j) for (int i=1;i<=nx;++i) {
    int id=id_v[idxV3(i-1,j-1,k-1,nx,ny)]; if(id){ iv[id]=i-1; jv[id]=j-2; kv[id]=k-1; }
  }
}
static void build_inverse_coords_w(const std::vector<int>& id_w, int nx, int ny, int nz, std::vector<int>& iw, std::vector<int>& jw, std::vector<int>& kw)
{
  int nW = *std::max_element(id_w.begin(), id_w.end());
  iw.assign((size_t)nW+1,0); jw.assign((size_t)nW+1,0); kw.assign((size_t)nW+1,0);
  for (int k=2;k<=nz;++k) for (int j=1;j<=ny;++j) for (int i=1;i<=nx;++i) {
    int id=id_w[idxW3(i-1,j-1,k-1,nx,ny)]; if(id){ iw[id]=i-1; jw[id]=j-1; kw[id]=k-2; }
  }
}
static void build_inverse_coords_p(const std::vector<int>& id_p, int nx, int ny, int nz, std::vector<int>& ip, std::vector<int>& jp, std::vector<int>& kp)
{
  int nP = *std::max_element(id_p.begin(), id_p.end());
  ip.assign((size_t)nP+1,0); jp.assign((size_t)nP+1,0); kp.assign((size_t)nP+1,0);
  for (int k=1;k<=nz;++k) for (int j=1;j<=ny;++j) for (int i=1;i<=nx;++i) {
    int id=id_p[idxP3(i-1,j-1,k-1,nx,ny)]; if(id){ ip[id]=i-1; jp[id]=j-1; kp[id]=k-1; }
  }
}

static void build_flat_of_id_u(const std::vector<int> &id_u, int nx, int ny, int nz, std::vector<int> &flat_of_id)
{
  int nU = *std::max_element(id_u.begin(), id_u.end());
  flat_of_id.assign((size_t)nU + 1, -1);
  for (int k = 0; k < nz; ++k)
    for (int j = 0; j < ny; ++j)
      for (int i = 1; i <= nx - 1; ++i) {
        int flat = idxU3(i, j, k, nx, ny);
        int id = id_u[(size_t)flat];
        if (id > 0) flat_of_id[(size_t)id] = flat;
      }
}

static void build_flat_of_id_v(const std::vector<int> &id_v, int nx, int ny, int nz, std::vector<int> &flat_of_id)
{
  int nV = *std::max_element(id_v.begin(), id_v.end());
  flat_of_id.assign((size_t)nV + 1, -1);
  for (int k = 0; k < nz; ++k)
    for (int j = 1; j <= ny - 1; ++j)
      for (int i = 0; i < nx; ++i) {
        int flat = idxV3(i, j, k, nx, ny);
        int id = id_v[(size_t)flat];
        if (id > 0) flat_of_id[(size_t)id] = flat;
      }
}

static void build_flat_of_id_w(const std::vector<int> &id_w, int nx, int ny, int nz, std::vector<int> &flat_of_id)
{
  int nW = *std::max_element(id_w.begin(), id_w.end());
  flat_of_id.assign((size_t)nW + 1, -1);
  for (int k = 1; k <= nz - 1; ++k)
    for (int j = 0; j < ny; ++j)
      for (int i = 0; i < nx; ++i) {
        int flat = idxW3(i, j, k, nx, ny);
        int id = id_w[(size_t)flat];
        if (id > 0) flat_of_id[(size_t)id] = flat;
      }
}

static void build_flat_of_id_p(const std::vector<int> &id_p, int nx, int ny, int nz, std::vector<int> &flat_of_id)
{
  int nP = *std::max_element(id_p.begin(), id_p.end());
  flat_of_id.assign((size_t)nP + 1, -1);
  for (int k = 0; k < nz; ++k)
    for (int j = 0; j < ny; ++j)
      for (int i = 0; i < nx; ++i) {
        int flat = idxP3(i, j, k, nx, ny);
        int id = id_p[(size_t)flat];
        if (id > 0) flat_of_id[(size_t)id] = flat;
      }
}

static void fill_rhs_full_from_active_u(HYPRE_Complex *rhs_full, const std::vector<double> &rhs_active,
                                        const std::vector<int> &id_u, int nx, int ny, int nz)
{
  int nxu = nx - 1;
  for (int k = 0; k < nz; ++k)
    for (int j = 0; j < ny; ++j)
      for (int iu0 = 0; iu0 < nxu; ++iu0) {
        int row = idxUu(iu0, j, k, nxu, ny);
        int aid = id_u[idxU3(iu0 + 1, j, k, nx, ny)];
        rhs_full[(size_t)row] = (aid > 0) ? rhs_active[(size_t)aid] : 0.0;
      }
}

static void fill_rhs_full_from_active_v(HYPRE_Complex *rhs_full, const std::vector<double> &rhs_active,
                                        const std::vector<int> &id_v, int nx, int ny, int nz)
{
  int nyv = ny - 1;
  for (int k = 0; k < nz; ++k)
    for (int jv = 0; jv < nyv; ++jv)
      for (int i = 0; i < nx; ++i) {
        int row = idxVv(i, jv, k, nx, nyv);
        int aid = id_v[idxV3(i, jv + 1, k, nx, ny)];
        rhs_full[(size_t)row] = (aid > 0) ? rhs_active[(size_t)aid] : 0.0;
      }
}

static void fill_rhs_full_from_active_w(HYPRE_Complex *rhs_full, const std::vector<double> &rhs_active,
                                        const std::vector<int> &id_w, int nx, int ny, int nz)
{
  int nzw = nz - 1;
  for (int kw0 = 0; kw0 < nzw; ++kw0)
    for (int j = 0; j < ny; ++j)
      for (int i = 0; i < nx; ++i) {
        int row = idxWw(i, j, kw0, nx, ny);
        int aid = id_w[idxW3(i, j, kw0 + 1, nx, ny)];
        rhs_full[(size_t)row] = (aid > 0) ? rhs_active[(size_t)aid] : 0.0;
      }
}

static void fill_rhs_full_from_active_p(HYPRE_Complex *rhs_full, const std::vector<double> &rhs_active,
                                        const std::vector<int> &id_p, int nx, int ny, int nz)
{
  for (int k = 0; k < nz; ++k)
    for (int j = 0; j < ny; ++j)
      for (int i = 0; i < nx; ++i) {
        int row = idxPfull(i, j, k, nx, ny);
        int aid = id_p[idxP3(i, j, k, nx, ny)];
        rhs_full[(size_t)row] = (aid > 0) ? rhs_active[(size_t)aid] : 0.0;
      }
}

static void fill_x0_u(HYPRE_Complex *x0, const std::vector<double> &u, int nx, int ny, int nz, bool zero_initial_guess)
{
  int nxu = nx - 1;
  for (int k = 0; k < nz; ++k)
    for (int j = 0; j < ny; ++j)
      for (int iu0 = 0; iu0 < nxu; ++iu0) {
        int row = idxUu(iu0, j, k, nxu, ny);
        x0[(size_t)row] = zero_initial_guess ? 0.0 : u[idxU3(iu0 + 1, j, k, nx, ny)];
      }
}

static void fill_x0_v(HYPRE_Complex *x0, const std::vector<double> &v, int nx, int ny, int nz, bool zero_initial_guess)
{
  int nyv = ny - 1;
  for (int k = 0; k < nz; ++k)
    for (int jv = 0; jv < nyv; ++jv)
      for (int i = 0; i < nx; ++i) {
        int row = idxVv(i, jv, k, nx, nyv);
        x0[(size_t)row] = zero_initial_guess ? 0.0 : v[idxV3(i, jv + 1, k, nx, ny)];
      }
}

static void fill_x0_w(HYPRE_Complex *x0, const std::vector<double> &w, int nx, int ny, int nz, bool zero_initial_guess)
{
  int nzw = nz - 1;
  for (int kw0 = 0; kw0 < nzw; ++kw0)
    for (int j = 0; j < ny; ++j)
      for (int i = 0; i < nx; ++i) {
        int row = idxWw(i, j, kw0, nx, ny);
        x0[(size_t)row] = zero_initial_guess ? 0.0 : w[idxW3(i, j, kw0 + 1, nx, ny)];
      }
}

static void scatter_solution_u(std::vector<double> &u_tilde, const HYPRE_Complex *x, int nx, int ny, int nz)
{
  int nxu = nx - 1;
  for (int k = 0; k < nz; ++k)
    for (int j = 0; j < ny; ++j)
      for (int iu0 = 0; iu0 < nxu; ++iu0)
        u_tilde[idxU3(iu0 + 1, j, k, nx, ny)] = x[(size_t)idxUu(iu0, j, k, nxu, ny)];
}

static void scatter_solution_v(std::vector<double> &v_tilde, const HYPRE_Complex *x, int nx, int ny, int nz)
{
  int nyv = ny - 1;
  for (int k = 0; k < nz; ++k)
    for (int jv = 0; jv < nyv; ++jv)
      for (int i = 0; i < nx; ++i)
        v_tilde[idxV3(i, jv + 1, k, nx, ny)] = x[(size_t)idxVv(i, jv, k, nx, nyv)];
}

static void scatter_solution_w(std::vector<double> &w_tilde, const HYPRE_Complex *x, int nx, int ny, int nz)
{
  int nzw = nz - 1;
  for (int kw0 = 0; kw0 < nzw; ++kw0)
    for (int j = 0; j < ny; ++j)
      for (int i = 0; i < nx; ++i)
        w_tilde[idxW3(i, j, kw0 + 1, nx, ny)] = x[(size_t)idxWw(i, j, kw0, nx, ny)];
}

static void extract_active_from_full_u(std::vector<double> &u_vec, const HYPRE_Complex *x, const std::vector<int> &id_u,
                                       int nx, int ny, int nz)
{
  int nU = *std::max_element(id_u.begin(), id_u.end());
  u_vec.assign((size_t)nU + 1, 0.0);
  int nxu = nx - 1;
  for (int k = 0; k < nz; ++k)
    for (int j = 0; j < ny; ++j)
      for (int iu0 = 0; iu0 < nxu; ++iu0) {
        int id = id_u[idxU3(iu0 + 1, j, k, nx, ny)];
        if (id > 0) u_vec[(size_t)id] = x[(size_t)idxUu(iu0, j, k, nxu, ny)];
      }
}

static void extract_active_from_full_v(std::vector<double> &v_vec, const HYPRE_Complex *x, const std::vector<int> &id_v,
                                       int nx, int ny, int nz)
{
  int nV = *std::max_element(id_v.begin(), id_v.end());
  v_vec.assign((size_t)nV + 1, 0.0);
  int nyv = ny - 1;
  for (int k = 0; k < nz; ++k)
    for (int jv = 0; jv < nyv; ++jv)
      for (int i = 0; i < nx; ++i) {
        int id = id_v[idxV3(i, jv + 1, k, nx, ny)];
        if (id > 0) v_vec[(size_t)id] = x[(size_t)idxVv(i, jv, k, nx, nyv)];
      }
}

static void extract_active_from_full_w(std::vector<double> &w_vec, const HYPRE_Complex *x, const std::vector<int> &id_w,
                                       int nx, int ny, int nz)
{
  int nW = *std::max_element(id_w.begin(), id_w.end());
  w_vec.assign((size_t)nW + 1, 0.0);
  int nzw = nz - 1;
  for (int kw0 = 0; kw0 < nzw; ++kw0)
    for (int j = 0; j < ny; ++j)
      for (int i = 0; i < nx; ++i) {
        int id = id_w[idxW3(i, j, kw0 + 1, nx, ny)];
        if (id > 0) w_vec[(size_t)id] = x[(size_t)idxWw(i, j, kw0, nx, ny)];
      }
}

static void embed_u_matrix_to_stencil_3d(const SparseMatrix &Aactive, const std::vector<int> &flat_of_id_u,
                                         const std::vector<int> &id_u, int nx, int ny, int nz,
                                         std::vector<HYPRE_Complex> &vals)
{
  int nxu = nx - 1;
  size_t nrows = (size_t)nxu * (size_t)ny * (size_t)nz;
  vals.assign((size_t)7 * nrows, 0.0);
  int sx = nx + 1, sy = ny;
  int plane = sx * sy;
  for (int k = 0; k < nz; ++k)
    for (int j = 0; j < ny; ++j)
      for (int iu0 = 0; iu0 < nxu; ++iu0) {
        int row = idxUu(iu0, j, k, nxu, ny);
        int aid = id_u[idxU3(iu0 + 1, j, k, nx, ny)];
        if (aid <= 0) { vals[0 * nrows + (size_t)row] = 1.0; continue; }
        for (const auto &kv : Aactive.rows[(size_t)(aid - 1)]) {
          int cid = kv.first;
          int flat = flat_of_id_u[(size_t)cid];
          int kc = flat / plane;
          int rem = flat - kc * plane;
          int jc = rem / sx;
          int ic = rem - jc * sx;
          int di = ic - (iu0 + 1), dj = jc - j, dk = kc - k;
          if      (di == 0 && dj == 0 && dk == 0) vals[0 * nrows + (size_t)row] += kv.second;
          else if (di == -1 && dj == 0 && dk == 0) vals[1 * nrows + (size_t)row] += kv.second;
          else if (di == +1 && dj == 0 && dk == 0) vals[2 * nrows + (size_t)row] += kv.second;
          else if (di == 0 && dj == -1 && dk == 0) vals[3 * nrows + (size_t)row] += kv.second;
          else if (di == 0 && dj == +1 && dk == 0) vals[4 * nrows + (size_t)row] += kv.second;
          else if (di == 0 && dj == 0 && dk == -1) vals[5 * nrows + (size_t)row] += kv.second;
          else if (di == 0 && dj == 0 && dk == +1) vals[6 * nrows + (size_t)row] += kv.second;
        }
      }
}

static void embed_v_matrix_to_stencil_3d(const SparseMatrix &Aactive, const std::vector<int> &flat_of_id_v,
                                         const std::vector<int> &id_v, int nx, int ny, int nz,
                                         std::vector<HYPRE_Complex> &vals)
{
  int nyv = ny - 1;
  size_t nrows = (size_t)nx * (size_t)nyv * (size_t)nz;
  vals.assign((size_t)7 * nrows, 0.0);
  int sx = nx, sy = ny + 1;
  int plane = sx * sy;
  for (int k = 0; k < nz; ++k)
    for (int jv = 0; jv < nyv; ++jv)
      for (int i = 0; i < nx; ++i) {
        int row = idxVv(i, jv, k, nx, nyv);
        int aid = id_v[idxV3(i, jv + 1, k, nx, ny)];
        if (aid <= 0) { vals[0 * nrows + (size_t)row] = 1.0; continue; }
        for (const auto &kv : Aactive.rows[(size_t)(aid - 1)]) {
          int cid = kv.first;
          int flat = flat_of_id_v[(size_t)cid];
          int kc = flat / plane;
          int rem = flat - kc * plane;
          int jc = rem / sx;
          int ic = rem - jc * sx;
          int di = ic - i, dj = jc - (jv + 1), dk = kc - k;
          if      (di == 0 && dj == 0 && dk == 0) vals[0 * nrows + (size_t)row] += kv.second;
          else if (di == -1 && dj == 0 && dk == 0) vals[1 * nrows + (size_t)row] += kv.second;
          else if (di == +1 && dj == 0 && dk == 0) vals[2 * nrows + (size_t)row] += kv.second;
          else if (di == 0 && dj == -1 && dk == 0) vals[3 * nrows + (size_t)row] += kv.second;
          else if (di == 0 && dj == +1 && dk == 0) vals[4 * nrows + (size_t)row] += kv.second;
          else if (di == 0 && dj == 0 && dk == -1) vals[5 * nrows + (size_t)row] += kv.second;
          else if (di == 0 && dj == 0 && dk == +1) vals[6 * nrows + (size_t)row] += kv.second;
        }
      }
}

static void embed_w_matrix_to_stencil_3d(const SparseMatrix &Aactive, const std::vector<int> &flat_of_id_w,
                                         const std::vector<int> &id_w, int nx, int ny, int nz,
                                         std::vector<HYPRE_Complex> &vals)
{
  int nzw = nz - 1;
  size_t nrows = (size_t)nx * (size_t)ny * (size_t)nzw;
  vals.assign((size_t)7 * nrows, 0.0);
  int sx = nx, sy = ny;
  int plane = sx * sy;
  for (int kw0 = 0; kw0 < nzw; ++kw0)
    for (int j = 0; j < ny; ++j)
      for (int i = 0; i < nx; ++i) {
        int row = idxWw(i, j, kw0, nx, ny);
        int aid = id_w[idxW3(i, j, kw0 + 1, nx, ny)];
        if (aid <= 0) { vals[0 * nrows + (size_t)row] = 1.0; continue; }
        for (const auto &kv : Aactive.rows[(size_t)(aid - 1)]) {
          int cid = kv.first;
          int flat = flat_of_id_w[(size_t)cid];
          int kc = flat / plane;
          int rem = flat - kc * plane;
          int jc = rem / sx;
          int ic = rem - jc * sx;
          int di = ic - i, dj = jc - j, dk = kc - (kw0 + 1);
          if      (di == 0 && dj == 0 && dk == 0) vals[0 * nrows + (size_t)row] += kv.second;
          else if (di == -1 && dj == 0 && dk == 0) vals[1 * nrows + (size_t)row] += kv.second;
          else if (di == +1 && dj == 0 && dk == 0) vals[2 * nrows + (size_t)row] += kv.second;
          else if (di == 0 && dj == -1 && dk == 0) vals[3 * nrows + (size_t)row] += kv.second;
          else if (di == 0 && dj == +1 && dk == 0) vals[4 * nrows + (size_t)row] += kv.second;
          else if (di == 0 && dj == 0 && dk == -1) vals[5 * nrows + (size_t)row] += kv.second;
          else if (di == 0 && dj == 0 && dk == +1) vals[6 * nrows + (size_t)row] += kv.second;
        }
      }
}

static void embed_p_matrix_to_stencil_3d(const SparseMatrix &Aactive, const std::vector<int> &flat_of_id_p,
                                         const std::vector<int> &id_p, int nx, int ny, int nz,
                                         std::vector<HYPRE_Complex> &vals)
{
  size_t nrows = (size_t)nx * (size_t)ny * (size_t)nz;
  vals.assign((size_t)7 * nrows, 0.0);
  int sx = nx, sy = ny;
  int plane = sx * sy;
  for (int k = 0; k < nz; ++k)
    for (int j = 0; j < ny; ++j)
      for (int i = 0; i < nx; ++i) {
        int row = idxPfull(i, j, k, nx, ny);
        int aid = id_p[idxP3(i, j, k, nx, ny)];
        if (aid <= 0) { vals[0 * nrows + (size_t)row] = 1.0; continue; }
        for (const auto &kv : Aactive.rows[(size_t)(aid - 1)]) {
          int cid = kv.first;
          int flat = flat_of_id_p[(size_t)cid];
          int kc = flat / plane;
          int rem = flat - kc * plane;
          int jc = rem / sx;
          int ic = rem - jc * sx;
          int di = ic - i, dj = jc - j, dk = kc - k;
          if      (di == 0 && dj == 0 && dk == 0) vals[0 * nrows + (size_t)row] += kv.second;
          else if (di == -1 && dj == 0 && dk == 0) vals[1 * nrows + (size_t)row] += kv.second;
          else if (di == +1 && dj == 0 && dk == 0) vals[2 * nrows + (size_t)row] += kv.second;
          else if (di == 0 && dj == -1 && dk == 0) vals[3 * nrows + (size_t)row] += kv.second;
          else if (di == 0 && dj == +1 && dk == 0) vals[4 * nrows + (size_t)row] += kv.second;
          else if (di == 0 && dj == 0 && dk == -1) vals[5 * nrows + (size_t)row] += kv.second;
          else if (di == 0 && dj == 0 && dk == +1) vals[6 * nrows + (size_t)row] += kv.second;
        }
      }
}

static void row_scale_sparse(const SparseMatrix &A, SparseMatrix &As, std::vector<double> &scale)
{
  scale.assign((size_t)A.nrows + 1, 1.0);
  As = SparseMatrix(A.nrows, A.ncols);
  for (int i = 1; i <= A.nrows; ++i) {
    double s = 0.0; for (const auto &kv : A.rows[(size_t)(i-1)]) s += std::abs(kv.second);
    if (s < 1.0e-14) s = 1.0;
    scale[(size_t)i] = 1.0 / s;
    for (const auto &kv : A.rows[(size_t)(i-1)]) add_entry(As, i, kv.first, scale[(size_t)i] * kv.second);
  }
  compress_zeros(As);
}

static void apply_row_scaling(const std::vector<double>& scale, std::vector<double>& rhs)
{ for (size_t i=1;i<rhs.size();++i) rhs[i] *= scale[i]; }


static std::vector<double> extract_diag_vec(const SparseMatrix &A)
{
  std::vector<double> d((size_t)A.nrows + 1, 0.0);
  for (int i = 1; i <= A.nrows; ++i) {
    for (const auto &kv : A.rows[(size_t)(i-1)]) if (kv.first == i) { d[(size_t)i] = kv.second; break; }
  }
  return d;
}

static double l2_norm1(const std::vector<double> &x)
{
  double s = 0.0; for (size_t i = 1; i < x.size(); ++i) s += x[i] * x[i]; return std::sqrt(s);
}

static size_t count_finite_vec(const std::vector<double> &x)
{
  size_t c = 0; for (double v : x) if (std::isfinite(v)) ++c; return c;
}

static void print_stats_vec1(const char *name, const std::vector<double> &x, int first_n = 8)
{
  size_t n = x.size() ? x.size() - 1 : 0;
  size_t nfin = 0, nnan = 0, ninf = 0;
  double xmin = 0.0, xmax = 0.0, n2 = 0.0, ninfv = 0.0; bool first = true;
  for (size_t i = 1; i < x.size(); ++i) {
    double v = x[i];
    if (std::isnan(v)) { ++nnan; continue; }
    if (!std::isfinite(v)) { ++ninf; continue; }
    ++nfin;
    if (first) { xmin = xmax = v; first = false; } else { xmin = std::min(xmin, v); xmax = std::max(xmax, v); }
    n2 += v * v; ninfv = std::max(ninfv, std::abs(v));
  }
  std::printf("%s\n", name);
  std::printf("  size            : %zu\n", n);
  std::printf("  finite / nan / inf : %zu / %zu / %zu\n", nfin, nnan, ninf);
  std::printf("  min / max       : %+ .16e / %+ .16e\n", xmin, xmax);
  std::printf("  norm2 / normInf : %+ .16e / %+ .16e\n", std::sqrt(n2), ninfv);
  std::printf("  first %d entries :\n", first_n);
  for (int k = 0; k < first_n && (size_t)(k + 1) < x.size(); ++k)
    std::printf("    (%6d)  %+ .16e\n", k + 1, x[(size_t)k + 1]);
}

static void print_stats_field(const char *name, const std::vector<double> &x, int nx, int ny, int nz, int first_n = 8)
{
  size_t nfin = 0, nnan = 0, ninf = 0;
  double xmin = 0.0, xmax = 0.0, n2 = 0.0, ninfv = 0.0; bool first = true;
  for (double v : x) {
    if (std::isnan(v)) { ++nnan; continue; }
    if (!std::isfinite(v)) { ++ninf; continue; }
    ++nfin;
    if (first) { xmin = xmax = v; first = false; } else { xmin = std::min(xmin, v); xmax = std::max(xmax, v); }
    n2 += v * v; ninfv = std::max(ninfv, std::abs(v));
  }
  std::printf("%s\n", name);
  std::printf("  shape           : [%d %d %d ]\n", nx, ny, nz);
  std::printf("  finite / nan / inf : %zu / %zu / %zu\n", nfin, nnan, ninf);
  std::printf("  min / max       : %+ .16e / %+ .16e\n", xmin, xmax);
  std::printf("  norm2 / normInf : %+ .16e / %+ .16e\n", std::sqrt(n2), ninfv);
  std::printf("  first %d flattened entries :\n", first_n);
  for (int k = 0; k < first_n && k < (int)x.size(); ++k)
    std::printf("    (%6d)  %+ .16e\n", k + 1, x[(size_t)k]);
}

static void print_stats_raw0(const char *name, const HYPRE_Complex *x, size_t n, int first_n = 8)
{
  size_t nfin = 0, nnan = 0, ninf = 0;
  double xmin = 0.0, xmax = 0.0, n2 = 0.0, ninfv = 0.0; bool first = true;
  for (size_t i = 0; i < n; ++i) {
    double v = x[i];
    if (std::isnan(v)) { ++nnan; continue; }
    if (!std::isfinite(v)) { ++ninf; continue; }
    ++nfin;
    if (first) { xmin = xmax = v; first = false; } else { xmin = std::min(xmin, v); xmax = std::max(xmax, v); }
    n2 += v * v; ninfv = std::max(ninfv, std::abs(v));
  }
  std::printf("%s\n", name);
  std::printf("  size            : %zu\n", n);
  std::printf("  finite / nan / inf : %zu / %zu / %zu\n", nfin, nnan, ninf);
  std::printf("  min / max       : %+ .16e / %+ .16e\n", xmin, xmax);
  std::printf("  norm2 / normInf : %+ .16e / %+ .16e\n", std::sqrt(n2), ninfv);
  std::printf("  first %d entries :\n", first_n);
  for (int k = 0; k < first_n && (size_t)k < n; ++k)
    std::printf("    (%6d)  %+ .16e\n", k + 1, x[(size_t)k]);
}


static void embed_u_matrix(const SparseMatrix &A, const std::vector<int> &id_u, const std::vector<int>& iu, const std::vector<int>& ju, const std::vector<int>& ku,
                           int nx, int ny, int nz, std::vector<HYPRE_Complex> &vals)
{
  int nxu = nx - 1; vals.assign((size_t)7*nxu*ny*nz, 0.0);
  for (int k=0;k<nz;++k) for (int j=0;j<ny;++j) for (int iu0=0; iu0<nxu; ++iu0) {
    int row_idx = idxUu(iu0,j,k,nxu,ny); int id = id_u[idxU3(iu0+1,j,k,nx,ny)];
    auto *a = &vals[(size_t)7*row_idx];
    if (!id) { a[0]=1.0; continue; }
    for (const auto &kv : A.rows[(size_t)(id-1)]) {
      int cid=kv.first; int di = iu[cid]-iu0; int dj = ju[cid]-j; int dk = ku[cid]-k;
      if (di==0 && dj==0 && dk==0) a[0]=kv.second;
      else if (di==-1 && dj==0 && dk==0) a[1]=kv.second;
      else if (di==+1 && dj==0 && dk==0) a[2]=kv.second;
      else if (di==0 && dj==-1 && dk==0) a[3]=kv.second;
      else if (di==0 && dj==+1 && dk==0) a[4]=kv.second;
      else if (di==0 && dj==0 && dk==-1) a[5]=kv.second;
      else if (di==0 && dj==0 && dk==+1) a[6]=kv.second;
    }
  }
}
static void embed_v_matrix(const SparseMatrix &A, const std::vector<int> &id_v, const std::vector<int>& iv, const std::vector<int>& jv, const std::vector<int>& kv,
                           int nx, int ny, int nz, std::vector<HYPRE_Complex> &vals)
{
  int nyv = ny - 1; vals.assign((size_t)7*nx*nyv*nz, 0.0);
  for (int k=0;k<nz;++k) for (int j=0;j<nyv;++j) for (int i=0; i<nx; ++i) {
    int row_idx = idxVv(i,j,k,nx,nyv); int id = id_v[idxV3(i,j+1,k,nx,ny)];
    auto *a = &vals[(size_t)7*row_idx];
    if (!id) { a[0]=1.0; continue; }
    for (const auto &kvp : A.rows[(size_t)(id-1)]) {
      int cid=kvp.first; int di = iv[cid]-i; int dj = jv[cid]-j; int dk = kv[cid]-k;
      if (di==0 && dj==0 && dk==0) a[0]=kvp.second;
      else if (di==-1 && dj==0 && dk==0) a[1]=kvp.second;
      else if (di==+1 && dj==0 && dk==0) a[2]=kvp.second;
      else if (di==0 && dj==-1 && dk==0) a[3]=kvp.second;
      else if (di==0 && dj==+1 && dk==0) a[4]=kvp.second;
      else if (di==0 && dj==0 && dk==-1) a[5]=kvp.second;
      else if (di==0 && dj==0 && dk==+1) a[6]=kvp.second;
    }
  }
}
static void embed_w_matrix(const SparseMatrix &A, const std::vector<int> &id_w, const std::vector<int>& iw, const std::vector<int>& jw, const std::vector<int>& kw,
                           int nx, int ny, int nz, std::vector<HYPRE_Complex> &vals)
{
  int nzw = nz - 1; vals.assign((size_t)7*nx*ny*nzw, 0.0);
  for (int kw0=0;kw0<nzw;++kw0) for (int j=0;j<ny;++j) for (int i=0; i<nx; ++i) {
    int row_idx = idxWw(i,j,kw0,nx,ny); int id = id_w[idxW3(i,j,kw0+1,nx,ny)];
    auto *a = &vals[(size_t)7*row_idx];
    if (!id) { a[0]=1.0; continue; }
    for (const auto &kvp : A.rows[(size_t)(id-1)]) {
      int cid=kvp.first; int di = iw[cid]-i; int dj = jw[cid]-j; int dk = kw[cid]-kw0;
      if (di==0 && dj==0 && dk==0) a[0]=kvp.second;
      else if (di==-1 && dj==0 && dk==0) a[1]=kvp.second;
      else if (di==+1 && dj==0 && dk==0) a[2]=kvp.second;
      else if (di==0 && dj==-1 && dk==0) a[3]=kvp.second;
      else if (di==0 && dj==+1 && dk==0) a[4]=kvp.second;
      else if (di==0 && dj==0 && dk==-1) a[5]=kvp.second;
      else if (di==0 && dj==0 && dk==+1) a[6]=kvp.second;
    }
  }
}
static void embed_p_matrix(const SparseMatrix &A, const std::vector<int> &id_p, const std::vector<int>& ip, const std::vector<int>& jp, const std::vector<int>& kp,
                           int nx, int ny, int nz, std::vector<HYPRE_Complex> &vals)
{
  vals.assign((size_t)7*nx*ny*nz, 0.0);
  for (int k=0;k<nz;++k) for (int j=0;j<ny;++j) for (int i=0; i<nx; ++i) {
    int row_idx = idxPfull(i,j,k,nx,ny); int id = id_p[idxP3(i,j,k,nx,ny)];
    auto *a = &vals[(size_t)7*row_idx];
    if (!id) { a[0]=1.0; continue; }
    for (const auto &kvp : A.rows[(size_t)(id-1)]) {
      int cid=kvp.first; int di = ip[cid]-i; int dj = jp[cid]-j; int dk = kp[cid]-k;
      if (di==0 && dj==0 && dk==0) a[0]=kvp.second;
      else if (di==-1 && dj==0 && dk==0) a[1]=kvp.second;
      else if (di==+1 && dj==0 && dk==0) a[2]=kvp.second;
      else if (di==0 && dj==-1 && dk==0) a[3]=kvp.second;
      else if (di==0 && dj==+1 && dk==0) a[4]=kvp.second;
      else if (di==0 && dj==0 && dk==-1) a[5]=kvp.second;
      else if (di==0 && dj==0 && dk==+1) a[6]=kvp.second;
    }
  }
}

static void build_uin(std::vector<double>& uin, int ny, int nz, double dy, double dz, double H, double Um)
{
  uin.assign((size_t)ny*(size_t)nz, 0.0);
  for (int k=0;k<nz;++k) for (int j=0;j<ny;++j) uin[(size_t)j + (size_t)ny*(size_t)k] = inletProfile((j+0.5)*dy,(k+0.5)*dz,H,Um);
}

static void pack_pressure(const std::vector<double>& p, const std::vector<char>& fluid_p, const std::vector<int>& id_p, int nx, int ny, int nz, std::vector<double>& vec)
{
  int nP = *std::max_element(id_p.begin(), id_p.end()); vec.assign((size_t)nP+1,0.0);
  for (int k=1;k<=nz;++k) for (int j=1;j<=ny;++j) for (int i=1;i<=nx;++i) {
    if (!fluid3_at(fluid_p,i,j,k,nx,ny)) continue; int id=id_p[idxP3(i-1,j-1,k-1,nx,ny)]; vec[(size_t)id]=p[idxP3(i-1,j-1,k-1,nx,ny)];
  }
}
static void unpack_pressure_add(std::vector<double>& p, const std::vector<double>& vec, const std::vector<char>& fluid_p, const std::vector<int>& id_p, int nx, int ny, int nz)
{
  for (int k=1;k<=nz;++k) for (int j=1;j<=ny;++j) for (int i=1;i<=nx;++i) {
    if (!fluid3_at(fluid_p,i,j,k,nx,ny)) continue; int id=id_p[idxP3(i-1,j-1,k-1,nx,ny)]; p[idxP3(i-1,j-1,k-1,nx,ny)] += vec[(size_t)id];
  }
}
static void unpack_projection_gradient_arrays(const std::vector<double>& gp, std::vector<double>& gpx, std::vector<double>& gpy, std::vector<double>& gpz,
    const std::vector<int>& id_qu, const std::vector<int>& id_qv, const std::vector<int>& id_qw, int nx, int ny, int nz)
{
  int nQu = *std::max_element(id_qu.begin(), id_qu.end());
  int nQv = *std::max_element(id_qv.begin(), id_qv.end());
  gpx.assign((size_t)(nx+1)*ny*nz,0.0); gpy.assign((size_t)nx*(ny+1)*nz,0.0); gpz.assign((size_t)nx*ny*(nz+1),0.0);
  for (int k=1;k<=nz;++k) for (int j=1;j<=ny;++j) for (int i=2;i<=nx+1;++i) {
    int id=id_qu[idxU3(i-1,j-1,k-1,nx,ny)]; if(id) gpx[idxU3(i-1,j-1,k-1,nx,ny)] = gp[(size_t)id];
  }
  for (int k=1;k<=nz;++k) for (int j=2;j<=ny;++j) for (int i=1;i<=nx;++i) {
    int id=id_qv[idxV3(i-1,j-1,k-1,nx,ny)]; if(id) gpy[idxV3(i-1,j-1,k-1,nx,ny)] = gp[(size_t)(nQu+id)];
  }
  for (int k=2;k<=nz;++k) for (int j=1;j<=ny;++j) for (int i=1;i<=nx;++i) {
    int id=id_qw[idxW3(i-1,j-1,k-1,nx,ny)]; if(id) gpz[idxW3(i-1,j-1,k-1,nx,ny)] = gp[(size_t)(nQu+nQv+id)];
  }
}
static void pack_projection_faces(const std::vector<double>& u, const std::vector<double>& v, const std::vector<double>& w,
    const std::vector<int>& id_qu, const std::vector<int>& id_qv, const std::vector<int>& id_qw, int nx, int ny, int nz, std::vector<double>& q)
{
  int nQu=*std::max_element(id_qu.begin(),id_qu.end()); int nQv=*std::max_element(id_qv.begin(),id_qv.end()); int nQw=*std::max_element(id_qw.begin(),id_qw.end());
  q.assign((size_t)(nQu+nQv+nQw)+1,0.0);
  for (int k=1;k<=nz;++k) for (int j=1;j<=ny;++j) for (int i=2;i<=nx+1;++i) { int id=id_qu[idxU3(i-1,j-1,k-1,nx,ny)]; if(id) q[(size_t)id] = u[idxU3(i-1,j-1,k-1,nx,ny)]; }
  for (int k=1;k<=nz;++k) for (int j=2;j<=ny;++j) for (int i=1;i<=nx;++i) { int id=id_qv[idxV3(i-1,j-1,k-1,nx,ny)]; if(id) q[(size_t)(nQu+id)] = v[idxV3(i-1,j-1,k-1,nx,ny)]; }
  for (int k=2;k<=nz;++k) for (int j=1;j<=ny;++j) for (int i=1;i<=nx;++i) { int id=id_qw[idxW3(i-1,j-1,k-1,nx,ny)]; if(id) q[(size_t)(nQu+nQv+id)] = w[idxW3(i-1,j-1,k-1,nx,ny)]; }
}
static void unpack_projection_faces(const std::vector<double>& q, std::vector<double>& u, std::vector<double>& v, std::vector<double>& w,
    const std::vector<int>& id_qu, const std::vector<int>& id_qv, const std::vector<int>& id_qw, int nx, int ny, int nz)
{
  int nQu=*std::max_element(id_qu.begin(),id_qu.end()); int nQv=*std::max_element(id_qv.begin(),id_qv.end());
  for (int k=1;k<=nz;++k) for (int j=1;j<=ny;++j) for (int i=2;i<=nx+1;++i) { int id=id_qu[idxU3(i-1,j-1,k-1,nx,ny)]; if(id) u[idxU3(i-1,j-1,k-1,nx,ny)] = q[(size_t)id]; }
  for (int k=1;k<=nz;++k) for (int j=2;j<=ny;++j) for (int i=1;i<=nx;++i) { int id=id_qv[idxV3(i-1,j-1,k-1,nx,ny)]; if(id) v[idxV3(i-1,j-1,k-1,nx,ny)] = q[(size_t)(nQu+id)]; }
  for (int k=2;k<=nz;++k) for (int j=1;j<=ny;++j) for (int i=1;i<=nx;++i) { int id=id_qw[idxW3(i-1,j-1,k-1,nx,ny)]; if(id) w[idxW3(i-1,j-1,k-1,nx,ny)] = q[(size_t)(nQu+nQv+id)]; }
}

static inline double val_u(const std::vector<double>& u, int i, int j, int k, int nx, int ny, int nz, double fb)
{
  if (i < 0 || i > nx || j < 0 || j >= ny || k < 0 || k >= nz) return fb;
  return u[idxU3(i,j,k,nx,ny)];
}
static inline double val_v(const std::vector<double>& v, int i, int j, int k, int nx, int ny, int nz, double fb)
{
  if (i < 0 || i >= nx || j < 0 || j > ny || k < 0 || k >= nz) return fb;
  return v[idxV3(i,j,k,nx,ny)];
}
static inline double val_w(const std::vector<double>& w, int i, int j, int k, int nx, int ny, int nz, double fb)
{
  if (i < 0 || i >= nx || j < 0 || j >= ny || k < 0 || k > nz) return fb;
  return w[idxW3(i,j,k,nx,ny)];
}

static void compute_convective_u_centered_3d(std::vector<double>& conv_u, const std::vector<double>& u, const std::vector<double>& v, const std::vector<double>& w,
    const std::vector<char>& active_u, const std::vector<double>& uin_step, int nx, int ny, int nz, double dx, double dy, double dz)
{
  conv_u.assign((size_t)(nx+1)*ny*nz, 0.0);
  for (int i=2;i<=nx;++i) for (int j=1;j<=ny;++j) for (int k=1;k<=nz;++k) {
    if (!active_u[idxU3(i-1,j-1,k-1,nx,ny)]) continue;
    double uc = u[idxU3(i-1,j-1,k-1,nx,ny)];
    double vadv = 0.25*(val_v(v,i-2,j-1,k-1,nx,ny,nz,0.0)+val_v(v,i-2,j,k-1,nx,ny,nz,0.0)+val_v(v,i-1,j-1,k-1,nx,ny,nz,0.0)+val_v(v,i-1,j,k-1,nx,ny,nz,0.0));
    double wadv = 0.25*(val_w(w,i-2,j-1,k-1,nx,ny,nz,0.0)+val_w(w,i-2,j-1,k,nx,ny,nz,0.0)+val_w(w,i-1,j-1,k-1,nx,ny,nz,0.0)+val_w(w,i-1,j-1,k,nx,ny,nz,0.0));
    double ue = val_u(u,i,j-1,k-1,nx,ny,nz,uc), uw = val_u(u,i-2,j-1,k-1,nx,ny,nz,uc);
    double un = val_u(u,i-1,j,k-1,nx,ny,nz,uc), us = val_u(u,i-1,j-2,k-1,nx,ny,nz,uc);
    double ut = val_u(u,i-1,j-1,k,nx,ny,nz,uc), ub = val_u(u,i-1,j-1,k-2,nx,ny,nz,uc);
    conv_u[idxU3(i-1,j-1,k-1,nx,ny)] = uc*(ue-uw)/(2*dx) + vadv*(un-us)/(2*dy) + wadv*(ut-ub)/(2*dz);
  }
}
static void compute_convective_v_centered_3d(std::vector<double>& conv_v, const std::vector<double>& u, const std::vector<double>& v, const std::vector<double>& w,
    const std::vector<char>& active_v, int nx, int ny, int nz, double dx, double dy, double dz)
{
  conv_v.assign((size_t)nx*(ny+1)*nz, 0.0);
  for (int i=1;i<=nx;++i) for (int j=2;j<=ny;++j) for (int k=1;k<=nz;++k) {
    if (!active_v[idxV3(i-1,j-1,k-1,nx,ny)]) continue;
    double vc = v[idxV3(i-1,j-1,k-1,nx,ny)];
    double uadv = 0.25*(val_u(u,i-1,j-2,k-1,nx,ny,nz,0.0)+val_u(u,i,j-2,k-1,nx,ny,nz,0.0)+val_u(u,i-1,j-1,k-1,nx,ny,nz,0.0)+val_u(u,i,j-1,k-1,nx,ny,nz,0.0));
    double wadv = 0.25*(val_w(w,i-1,j-2,k-1,nx,ny,nz,0.0)+val_w(w,i-1,j-1,k-1,nx,ny,nz,0.0)+val_w(w,i-1,j-2,k,nx,ny,nz,0.0)+val_w(w,i-1,j-1,k,nx,ny,nz,0.0));
    double ve = val_v(v,i,j-1,k-1,nx,ny,nz,vc), vw = val_v(v,i-2,j-1,k-1,nx,ny,nz,vc);
    double vn = val_v(v,i-1,j,k-1,nx,ny,nz,vc), vs = val_v(v,i-1,j-2,k-1,nx,ny,nz,vc);
    double vt = val_v(v,i-1,j-1,k,nx,ny,nz,vc), vb = val_v(v,i-1,j-1,k-2,nx,ny,nz,vc);
    conv_v[idxV3(i-1,j-1,k-1,nx,ny)] = uadv*(ve-vw)/(2*dx) + vc*(vn-vs)/(2*dy) + wadv*(vt-vb)/(2*dz);
  }
}
static void compute_convective_w_centered_3d(std::vector<double>& conv_w, const std::vector<double>& u, const std::vector<double>& v, const std::vector<double>& w,
    const std::vector<char>& active_w, int nx, int ny, int nz, double dx, double dy, double dz)
{
  conv_w.assign((size_t)nx*ny*(nz+1), 0.0);
  for (int i=1;i<=nx;++i) for (int j=1;j<=ny;++j) for (int k=2;k<=nz;++k) {
    if (!active_w[idxW3(i-1,j-1,k-1,nx,ny)]) continue;
    double wc = w[idxW3(i-1,j-1,k-1,nx,ny)];
    double uadv = 0.25*(val_u(u,i-1,j-1,k-2,nx,ny,nz,0.0)+val_u(u,i,j-1,k-2,nx,ny,nz,0.0)+val_u(u,i-1,j-1,k-1,nx,ny,nz,0.0)+val_u(u,i,j-1,k-1,nx,ny,nz,0.0));
    double vadv = 0.25*(val_v(v,i-1,j-1,k-2,nx,ny,nz,0.0)+val_v(v,i-1,j,k-2,nx,ny,nz,0.0)+val_v(v,i-1,j-1,k-1,nx,ny,nz,0.0)+val_v(v,i-1,j,k-1,nx,ny,nz,0.0));
    double we = val_w(w,i,j-1,k-1,nx,ny,nz,wc), ww = val_w(w,i-2,j-1,k-1,nx,ny,nz,wc);
    double wn = val_w(w,i-1,j,k-1,nx,ny,nz,wc), ws = val_w(w,i-1,j-2,k-1,nx,ny,nz,wc);
    double wt = val_w(w,i-1,j-1,k,nx,ny,nz,wc), wb = val_w(w,i-1,j-1,k-2,nx,ny,nz,wc);
    conv_w[idxW3(i-1,j-1,k-1,nx,ny)] = uadv*(we-ww)/(2*dx) + vadv*(wn-ws)/(2*dy) + wc*(wt-wb)/(2*dz);
  }
}

static void enforce_bc_and_cleanup(std::vector<double>& u, std::vector<double>& v, std::vector<double>& w,
  const std::vector<char>& active_u, const std::vector<char>& active_v, const std::vector<char>& active_w,
  const std::vector<double>& uin_step, int nx, int ny, int nz)
{
  for (int k=0;k<nz;++k) for (int j=0;j<ny;++j) {
    u[idxU3(0,j,k,nx,ny)] = uin_step[(size_t)j + (size_t)ny*(size_t)k];
    u[idxU3(nx,j,k,nx,ny)] = u[idxU3(nx-1,j,k,nx,ny)];
  }
  for (int k=0;k<nz;++k) for (int i=0;i<nx;++i) {
    v[idxV3(i,0,k,nx,ny)] = 0.0; v[idxV3(i,ny,k,nx,ny)] = 0.0;
  }
  for (int j=0;j<ny;++j) for (int i=0;i<nx;++i) {
    w[idxW3(i,j,0,nx,ny)] = 0.0; w[idxW3(i,j,nz,nx,ny)] = 0.0;
  }
  for (int k=1;k<=nz;++k) for (int j=1;j<=ny;++j) for (int i=2;i<=nx;++i) if (!active_u[idxU3(i-1,j-1,k-1,nx,ny)]) u[idxU3(i-1,j-1,k-1,nx,ny)] = 0.0;
  for (int k=1;k<=nz;++k) for (int j=2;j<=ny;++j) for (int i=1;i<=nx;++i) if (!active_v[idxV3(i-1,j-1,k-1,nx,ny)]) v[idxV3(i-1,j-1,k-1,nx,ny)] = 0.0;
  for (int k=2;k<=nz;++k) for (int j=1;j<=ny;++j) for (int i=1;i<=nx;++i) if (!active_w[idxW3(i-1,j-1,k-1,nx,ny)]) w[idxW3(i-1,j-1,k-1,nx,ny)] = 0.0;
}

static void write_vtk_cellcentered(const std::string& filename, const std::vector<double>& u, const std::vector<double>& v, const std::vector<double>& w,
    const std::vector<double>& p, const std::vector<char>& fluid_p, int nx, int ny, int nz, double Lx, double Ly, double Lz)
{
  double dx=Lx/nx, dy=Ly/ny, dz=Lz/nz;
  std::FILE* fp = std::fopen(filename.c_str(), "w"); if(!fp) return;
  std::fprintf(fp, "# vtk DataFile Version 3.0\n");
  std::fprintf(fp, "3D immersed cylinder MAC cell-centered\nASCII\nDATASET STRUCTURED_POINTS\n");
  std::fprintf(fp, "DIMENSIONS %d %d %d\n", nx+1, ny+1, nz+1);
  std::fprintf(fp, "ORIGIN 0 0 0\nSPACING %.16e %.16e %.16e\n", dx,dy,dz);
  std::fprintf(fp, "CELL_DATA %d\n", nx*ny*nz);
  std::fprintf(fp, "SCALARS ux double 1\nLOOKUP_TABLE default\n");
  for (int k=0;k<nz;++k) for (int j=0;j<ny;++j) for (int i=0;i<nx;++i) std::fprintf(fp, "%.16e\n", 0.5*(u[idxU3(i,j,k,nx,ny)] + u[idxU3(i+1,j,k,nx,ny)]));
  std::fprintf(fp, "SCALARS uy double 1\nLOOKUP_TABLE default\n");
  for (int k=0;k<nz;++k) for (int j=0;j<ny;++j) for (int i=0;i<nx;++i) std::fprintf(fp, "%.16e\n", 0.5*(v[idxV3(i,j,k,nx,ny)] + v[idxV3(i,j+1,k,nx,ny)]));
  std::fprintf(fp, "SCALARS uz double 1\nLOOKUP_TABLE default\n");
  for (int k=0;k<nz;++k) for (int j=0;j<ny;++j) for (int i=0;i<nx;++i) std::fprintf(fp, "%.16e\n", 0.5*(w[idxW3(i,j,k,nx,ny)] + w[idxW3(i,j,k+1,nx,ny)]));
  std::fprintf(fp, "SCALARS pressure double 1\nLOOKUP_TABLE default\n");
  for (double x : p) std::fprintf(fp, "%.16e\n", x);
  std::fprintf(fp, "SCALARS fluid_mask int 1\nLOOKUP_TABLE default\n");
  for (char c : fluid_p) std::fprintf(fp, "%d\n", c?1:0);
  std::fclose(fp);
}

