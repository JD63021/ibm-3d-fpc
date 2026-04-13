#include "common.hpp"
#include "immersed_geometry.hpp"
#include "immersed_operators.hpp"
#include "hypre_struct_utils.hpp"
#include "gpu_dense_ops.hpp"
#include "solver_helpers.hpp"

int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);
  int rank = 0; MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  Params par; ParseArgs(argc, argv, par);
  CUDA_CALL(cudaSetDevice(par.device));
  if (rank == 0) print_device_info(par.device);

  HYPRE_CALL(HYPRE_Initialize());
  HYPRE_CALL(HYPRE_SetMemoryLocation(HYPRE_MEMORY_DEVICE));
  HYPRE_CALL(HYPRE_SetExecutionPolicy(HYPRE_EXEC_DEVICE));

  const int nx = par.nx, ny = par.ny, nz = par.nz;
  const double dx = par.Lx / nx, dy = par.Ly / ny, dz = par.Lz / nz;

  std::vector<double> beta2d, alpha_u2d, alpha_v2d;
  compute_pressure_area_fractions_2d(beta2d, par);
  compute_vertical_face_open_fractions_2d(alpha_u2d, par);
  compute_horizontal_face_open_fractions_2d(alpha_v2d, par);

  std::vector<char> fluid2d, active_u2d, active_v2d, active_qu2d, active_qv2d;
  std::vector<std::pair<int,int>> pruned_cols;
  int n_pruned = build_and_prune_2d_masks(fluid2d, active_u2d, active_v2d, active_qu2d, active_qv2d,
                                          beta2d, alpha_u2d, alpha_v2d, par, pruned_cols);

  std::vector<double> beta_p((size_t)nx*ny*nz, 0.0), alpha_u((size_t)(nx+1)*ny*nz, 0.0), alpha_v((size_t)nx*(ny+1)*nz, 0.0), alpha_w((size_t)nx*ny*(nz+1), 1.0);
  std::vector<char> fluid_p((size_t)nx*ny*nz, 0), active_u((size_t)(nx+1)*ny*nz, 0), active_v((size_t)nx*(ny+1)*nz, 0), active_w((size_t)nx*ny*(nz+1), 0),
                    active_qu((size_t)(nx+1)*ny*nz, 0), active_qv((size_t)nx*(ny+1)*nz, 0), active_qw((size_t)nx*ny*(nz+1), 0);

  for (int k = 1; k <= nz; ++k) {
    for (int j = 1; j <= ny; ++j) {
      for (int i = 1; i <= nx; ++i) {
        size_t id = idxP3(i-1,j-1,k-1,nx,ny);
        beta_p[id] = beta2d[(size_t)id2_flat(i,j,nx)-1];
        fluid_p[id] = fluid2d[(size_t)id2_flat(i,j,nx)-1];
      }
    }
    for (int j = 1; j <= ny; ++j) {
      for (int i = 1; i <= nx+1; ++i) {
        size_t id = idxU3(i-1,j-1,k-1,nx,ny);
        alpha_u[id] = alpha_u2d[(size_t)(i-1) + (size_t)(j-1)*(size_t)(nx+1)];
        active_u[id] = active_u2d[(size_t)(i-1) + (size_t)(j-1)*(size_t)(nx+1)];
        active_qu[id] = active_qu2d[(size_t)(i-1) + (size_t)(j-1)*(size_t)(nx+1)];
      }
    }
    for (int j = 1; j <= ny+1; ++j) {
      for (int i = 1; i <= nx; ++i) {
        size_t id = idxV3(i-1,j-1,k-1,nx,ny);
        alpha_v[id] = alpha_v2d[(size_t)(i-1) + (size_t)(j-1)*(size_t)nx];
        active_v[id] = active_v2d[(size_t)(i-1) + (size_t)(j-1)*(size_t)nx];
        active_qv[id] = active_qv2d[(size_t)(i-1) + (size_t)(j-1)*(size_t)nx];
      }
    }
  }
  for (int kw = 1; kw <= nz+1; ++kw) {
    for (int j = 1; j <= ny; ++j) {
      for (int i = 1; i <= nx; ++i) {
        bool adjacent = (kw > 1 && fluid3_at(fluid_p, i, j, kw-1, nx, ny)) || (kw <= nz && fluid3_at(fluid_p, i, j, kw, nx, ny));
        size_t id = idxW3(i-1,j-1,kw-1,nx,ny);
        active_w[id] = (char)adjacent;
        active_qw[id] = (char)adjacent;
      }
    }
  }

  std::vector<int> id_u(active_u.size(),0), id_v(active_v.size(),0), id_w(active_w.size(),0), id_p(fluid_p.size(),0),
                   id_qu(active_qu.size(),0), id_qv(active_qv.size(),0), id_qw(active_qw.size(),0);
  {int c=0; for(size_t t=0;t<active_u.size();++t) if(active_u[t]) id_u[t]=++c;}
  {int c=0; for(size_t t=0;t<active_v.size();++t) if(active_v[t]) id_v[t]=++c;}
  {int c=0; for(size_t t=0;t<active_w.size();++t) if(active_w[t]) id_w[t]=++c;}
  {int c=0; for(size_t t=0;t<fluid_p.size();++t) if(fluid_p[t]) id_p[t]=++c;}
  {int c=0; for(size_t t=0;t<active_qu.size();++t) if(active_qu[t]) id_qu[t]=++c;}
  {int c=0; for(size_t t=0;t<active_qv.size();++t) if(active_qv[t]) id_qv[t]=++c;}
  {int c=0; for(size_t t=0;t<active_qw.size();++t) if(active_qw[t]) id_qw[t]=++c;}

  int Nu=*std::max_element(id_u.begin(),id_u.end()), Nv=*std::max_element(id_v.begin(),id_v.end()),
      Nw=*std::max_element(id_w.begin(),id_w.end()), Np=*std::max_element(id_p.begin(),id_p.end());
  int Nqu=*std::max_element(id_qu.begin(),id_qu.end()), Nqv=*std::max_element(id_qv.begin(),id_qv.end()), Nqw=*std::max_element(id_qw.begin(),id_qw.end());

  SparseMatrix Au0, Av0, Aw0, Dproj, Gproj, Ap0;
  std::vector<double> rhs_u_bc_unit, rhs_v_bc, rhs_w_bc, div_bc_unit;
  build_u_diffusion_3d_immersed(Au0, rhs_u_bc_unit, active_u, id_u, par);
  build_v_diffusion_3d_immersed(Av0, rhs_v_bc, active_v, id_v, par);
  build_w_diffusion_3d_immersed(Aw0, rhs_w_bc, active_w, id_w, par);
  build_projection_operators_3d_channel_cylinder(Dproj, Gproj, Ap0, div_bc_unit,
      beta_p, alpha_u, alpha_v, alpha_w, fluid_p, active_qu, active_qv, active_qw,
      id_p, id_qu, id_qv, id_qw, par);

  SparseMatrix Au, Av, Aw, Ap;
  std::vector<double> scale_u, scale_v, scale_w;
  row_scale_sparse(Au0, Au, scale_u);
  row_scale_sparse(Av0, Av, scale_v);
  row_scale_sparse(Aw0, Aw, scale_w);
  Ap = Ap0;
  for (int i = 1; i <= Ap.nrows; ++i)
    for (auto &kv : Ap.rows[(size_t)(i-1)]) kv.second = -kv.second;

  std::vector<int> flat_of_id_u, flat_of_id_v, flat_of_id_w, flat_of_id_p;
  build_flat_of_id_u(id_u, nx, ny, nz, flat_of_id_u);
  build_flat_of_id_v(id_v, nx, ny, nz, flat_of_id_v);
  build_flat_of_id_w(id_w, nx, ny, nz, flat_of_id_w);
  build_flat_of_id_p(id_p, nx, ny, nz, flat_of_id_p);

  std::vector<HYPRE_Complex> Ahu_vals_rowmajor, Ahv_vals_rowmajor, Ahw_vals_rowmajor, Ahp_vals_rowmajor;
  embed_u_matrix_to_rowmajor_3d(Au, flat_of_id_u, id_u, nx, ny, nz, Ahu_vals_rowmajor);
  embed_v_matrix_to_rowmajor_3d(Av, flat_of_id_v, id_v, nx, ny, nz, Ahv_vals_rowmajor);
  embed_w_matrix_to_rowmajor_3d(Aw, flat_of_id_w, id_w, nx, ny, nz, Ahw_vals_rowmajor);
  embed_p_matrix_to_rowmajor_3d(Ap, flat_of_id_p, id_p, nx, ny, nz, Ahp_vals_rowmajor);

  HYPRE_Complex *Ahu_vals_rowmajor_d = nullptr, *Ahv_vals_rowmajor_d = nullptr, *Ahw_vals_rowmajor_d = nullptr, *Ahp_vals_rowmajor_d = nullptr;
  CopyVecToManagedC(&Ahu_vals_rowmajor_d, Ahu_vals_rowmajor);
  CopyVecToManagedC(&Ahv_vals_rowmajor_d, Ahv_vals_rowmajor);
  CopyVecToManagedC(&Ahw_vals_rowmajor_d, Ahw_vals_rowmajor);
  CopyVecToManagedC(&Ahp_vals_rowmajor_d, Ahp_vals_rowmajor);

  std::vector<double> u((size_t)(nx+1)*ny*nz,0.0), v((size_t)nx*(ny+1)*nz,0.0), w((size_t)nx*ny*(nz+1),0.0), p((size_t)nx*ny*nz,0.0);
  std::vector<double> uin; build_uin(uin, ny, nz, dy, dz, par.Ly, par.Um);
  std::vector<double> uin_step = uin;
  enforce_bc_and_cleanup(u, v, w, active_u, active_v, active_w, uin_step, nx, ny, nz);

  double *du=nullptr,*dv=nullptr,*dw=nullptr,*dp=nullptr,*du_tilde=nullptr,*dv_tilde=nullptr,*dw_tilde=nullptr,*duin_step=nullptr,*dconv_u=nullptr,*dconv_v=nullptr,*dconv_w=nullptr,*dgpx=nullptr,*dgpy=nullptr,*dgpz=nullptr;
  MallocManagedD(&du, (size_t)(nx+1)*ny*nz); MallocManagedD(&dv, (size_t)nx*(ny+1)*nz); MallocManagedD(&dw, (size_t)nx*ny*(nz+1)); MallocManagedD(&dp, (size_t)nx*ny*nz);
  MallocManagedD(&du_tilde, (size_t)(nx+1)*ny*nz); MallocManagedD(&dv_tilde, (size_t)nx*(ny+1)*nz); MallocManagedD(&dw_tilde, (size_t)nx*ny*(nz+1));
  MallocManagedD(&duin_step, (size_t)ny*nz); MallocManagedD(&dconv_u, (size_t)(nx+1)*ny*nz); MallocManagedD(&dconv_v, (size_t)nx*(ny+1)*nz); MallocManagedD(&dconv_w, (size_t)nx*ny*(nz+1));
  MallocManagedD(&dgpx, (size_t)(nx+1)*ny*nz); MallocManagedD(&dgpy, (size_t)nx*(ny+1)*nz); MallocManagedD(&dgpz, (size_t)nx*ny*(nz+1));
  for(size_t i=0;i<u.size();++i) du[i]=u[i]; for(size_t i=0;i<v.size();++i) dv[i]=v[i]; for(size_t i=0;i<w.size();++i) dw[i]=w[i]; for(size_t i=0;i<p.size();++i) dp[i]=p[i]; for(size_t i=0;i<uin_step.size();++i) duin_step[i]=uin_step[i];
  unsigned char *dactive_u=nullptr,*dactive_v=nullptr,*dactive_w=nullptr,*dfluid_p=nullptr; int *did_u=nullptr,*did_v=nullptr,*did_w=nullptr,*did_p=nullptr,*did_qu=nullptr,*did_qv=nullptr,*did_qw=nullptr; double *dscale_u=nullptr,*dscale_v=nullptr,*dscale_w=nullptr,*drhs_u_bc_unit=nullptr,*drhs_v_bc=nullptr,*drhs_w_bc=nullptr,*ddiv_bc_unit=nullptr;
  CopyVecToManagedB(&dactive_u, active_u); CopyVecToManagedB(&dactive_v, active_v); CopyVecToManagedB(&dactive_w, active_w); CopyVecToManagedB(&dfluid_p, fluid_p);
  CopyVecToManagedI(&did_u, id_u); CopyVecToManagedI(&did_v, id_v); CopyVecToManagedI(&did_w, id_w); CopyVecToManagedI(&did_p, id_p); CopyVecToManagedI(&did_qu, id_qu); CopyVecToManagedI(&did_qv, id_qv); CopyVecToManagedI(&did_qw, id_qw);
  CopyVecToManagedD(&dscale_u, scale_u); CopyVecToManagedD(&dscale_v, scale_v); CopyVecToManagedD(&dscale_w, scale_w); CopyVecToManagedD(&drhs_u_bc_unit, rhs_u_bc_unit); CopyVecToManagedD(&drhs_v_bc, rhs_v_bc); CopyVecToManagedD(&drhs_w_bc, rhs_w_bc); CopyVecToManagedD(&ddiv_bc_unit, div_bc_unit);
  ManagedCSR dDproj = build_managed_csr_1based(Dproj), dGproj = build_managed_csr_1based(Gproj);

  HYPRE_Complex *rhsu_full=nullptr,*xu_full=nullptr,*rhsv_full=nullptr,*xv_full=nullptr,*rhsw_full=nullptr,*xw_full=nullptr,*rhsp_full=nullptr,*xp_full=nullptr;
  MallocManagedC(&rhsu_full,(size_t)(nx-1)*ny*nz); MallocManagedC(&xu_full,(size_t)(nx-1)*ny*nz);
  MallocManagedC(&rhsv_full,(size_t)nx*(ny-1)*nz); MallocManagedC(&xv_full,(size_t)nx*(ny-1)*nz);
  MallocManagedC(&rhsw_full,(size_t)nx*ny*(nz-1)); MallocManagedC(&xw_full,(size_t)nx*ny*(nz-1));
  MallocManagedC(&rhsp_full,(size_t)nx*ny*nz); MallocManagedC(&xp_full,(size_t)nx*ny*nz);
  double *dpvec=nullptr,*dgp=nullptr,*dqtilde=nullptr,*dDqt=nullptr,*dphi_vec=nullptr,*dgphi=nullptr,*ddivv=nullptr;
  // Np already defined above
  int Nq = Nqu + Nqv + Nqw;
  MallocManagedD(&dpvec, (size_t)Np + 1); MallocManagedD(&dgp, (size_t)Nq + 1); MallocManagedD(&dqtilde, (size_t)Nq + 1);
  MallocManagedD(&dDqt, (size_t)Np + 1); MallocManagedD(&dphi_vec, (size_t)Np + 1); MallocManagedD(&dgphi, (size_t)Nq + 1); MallocManagedD(&ddivv, (size_t)Np + 1);

  HYPRE_StructGrid gu = CreateGrid3D(nx-1, ny, nz), gv = CreateGrid3D(nx, ny-1, nz), gw = CreateGrid3D(nx, ny, nz-1), gpgrid = CreateGrid3D(nx, ny, nz);
  HYPRE_StructStencil st = CreateStencil3D();

  HYPRE_StructMatrix Ahu = CreateMatrix3D(gu, st, nx-1, ny, nz, Ahu_vals_rowmajor_d);
  HYPRE_StructMatrix Ahv = CreateMatrix3D(gv, st, nx, ny-1, nz, Ahv_vals_rowmajor_d);
  HYPRE_StructMatrix Ahw = CreateMatrix3D(gw, st, nx, ny, nz-1, Ahw_vals_rowmajor_d);
  HYPRE_StructMatrix Ahp = CreateMatrix3D(gpgrid, st, nx, ny, nz, Ahp_vals_rowmajor_d);

  HYPRE_StructVector bhu = CreateVector3D(gu), xhu = CreateVector3D(gu);
  HYPRE_StructVector bhv = CreateVector3D(gv), xhv = CreateVector3D(gv);
  HYPRE_StructVector bhw = CreateVector3D(gw), xhw = CreateVector3D(gw);
  HYPRE_StructVector bhp = CreateVector3D(gpgrid), xhp = CreateVector3D(gpgrid);

  for (size_t i = 0; i < (size_t)(nx-1)*ny*nz; ++i) { rhsu_full[i] = 0.0; xu_full[i] = 0.0; }
  for (size_t i = 0; i < (size_t)nx*(ny-1)*nz; ++i) { rhsv_full[i] = 0.0; xv_full[i] = 0.0; }
  for (size_t i = 0; i < (size_t)nx*ny*(nz-1); ++i) { rhsw_full[i] = 0.0; xw_full[i] = 0.0; }
  for (size_t i = 0; i < (size_t)nx*ny*nz; ++i) { rhsp_full[i] = 0.0; xp_full[i] = 0.0; }
  SetVector3D(bhu, nx-1, ny, nz, rhsu_full); SetVector3D(xhu, nx-1, ny, nz, xu_full);
  SetVector3D(bhv, nx, ny-1, nz, rhsv_full); SetVector3D(xhv, nx, ny-1, nz, xv_full);
  SetVector3D(bhw, nx, ny, nz-1, rhsw_full); SetVector3D(xhw, nx, ny, nz-1, xw_full);
  SetVector3D(bhp, nx, ny, nz, rhsp_full); SetVector3D(xhp, nx, ny, nz, xp_full);

  HYPRE_StructSolver solver_u, solver_v, solver_w, solver_p, pc_u, pc_v, pc_w, pc_p;
  HYPRE_CALL(HYPRE_StructGMRESCreate(MPI_COMM_WORLD, &solver_u));
  HYPRE_CALL(HYPRE_StructGMRESSetTol(solver_u, par.vel_tol));
  HYPRE_CALL(HYPRE_StructGMRESSetMaxIter(solver_u, par.vel_maxit));
  HYPRE_CALL(HYPRE_StructGMRESSetKDim(solver_u, par.vel_kdim));
  HYPRE_CALL(HYPRE_StructGMRESSetPrintLevel(solver_u, par.vel_print));
  HYPRE_CALL(HYPRE_StructJacobiCreate(MPI_COMM_WORLD, &pc_u));
  HYPRE_CALL(HYPRE_StructJacobiSetMaxIter(pc_u, par.vel_jacobi_iters));
  HYPRE_CALL(HYPRE_StructJacobiSetZeroGuess(pc_u));
  HYPRE_CALL(HYPRE_StructGMRESSetPrecond(solver_u, HYPRE_PtrToStructSolverFcn(HYPRE_StructJacobiSolve), HYPRE_PtrToStructSolverFcn(HYPRE_StructJacobiSetup), pc_u));
  HYPRE_CALL(HYPRE_StructGMRESSetup(solver_u, Ahu, bhu, xhu));

  HYPRE_CALL(HYPRE_StructGMRESCreate(MPI_COMM_WORLD, &solver_v));
  HYPRE_CALL(HYPRE_StructGMRESSetTol(solver_v, par.vel_tol));
  HYPRE_CALL(HYPRE_StructGMRESSetMaxIter(solver_v, par.vel_maxit));
  HYPRE_CALL(HYPRE_StructGMRESSetKDim(solver_v, par.vel_kdim));
  HYPRE_CALL(HYPRE_StructGMRESSetPrintLevel(solver_v, par.vel_print));
  HYPRE_CALL(HYPRE_StructJacobiCreate(MPI_COMM_WORLD, &pc_v));
  HYPRE_CALL(HYPRE_StructJacobiSetMaxIter(pc_v, par.vel_jacobi_iters));
  HYPRE_CALL(HYPRE_StructJacobiSetZeroGuess(pc_v));
  HYPRE_CALL(HYPRE_StructGMRESSetPrecond(solver_v, HYPRE_PtrToStructSolverFcn(HYPRE_StructJacobiSolve), HYPRE_PtrToStructSolverFcn(HYPRE_StructJacobiSetup), pc_v));
  HYPRE_CALL(HYPRE_StructGMRESSetup(solver_v, Ahv, bhv, xhv));

  HYPRE_CALL(HYPRE_StructGMRESCreate(MPI_COMM_WORLD, &solver_w));
  HYPRE_CALL(HYPRE_StructGMRESSetTol(solver_w, par.vel_tol));
  HYPRE_CALL(HYPRE_StructGMRESSetMaxIter(solver_w, par.vel_maxit));
  HYPRE_CALL(HYPRE_StructGMRESSetKDim(solver_w, par.vel_kdim));
  HYPRE_CALL(HYPRE_StructGMRESSetPrintLevel(solver_w, par.vel_print));
  HYPRE_CALL(HYPRE_StructJacobiCreate(MPI_COMM_WORLD, &pc_w));
  HYPRE_CALL(HYPRE_StructJacobiSetMaxIter(pc_w, par.vel_jacobi_iters));
  HYPRE_CALL(HYPRE_StructJacobiSetZeroGuess(pc_w));
  HYPRE_CALL(HYPRE_StructGMRESSetPrecond(solver_w, HYPRE_PtrToStructSolverFcn(HYPRE_StructJacobiSolve), HYPRE_PtrToStructSolverFcn(HYPRE_StructJacobiSetup), pc_w));
  HYPRE_CALL(HYPRE_StructGMRESSetup(solver_w, Ahw, bhw, xhw));

  HYPRE_CALL(HYPRE_StructGMRESCreate(MPI_COMM_WORLD, &solver_p));
  HYPRE_CALL(HYPRE_StructGMRESSetTol(solver_p, par.p_tol));
  HYPRE_CALL(HYPRE_StructGMRESSetMaxIter(solver_p, par.p_maxit));
  HYPRE_CALL(HYPRE_StructGMRESSetKDim(solver_p, par.p_kdim));
  HYPRE_CALL(HYPRE_StructGMRESSetPrintLevel(solver_p, par.p_print));
  HYPRE_CALL(HYPRE_StructGMRESSetLogging(solver_p, 1));
  HYPRE_CALL(HYPRE_StructPFMGCreate(MPI_COMM_WORLD, &pc_p));
  HYPRE_CALL(HYPRE_StructPFMGSetMaxIter(pc_p, par.p_pc_maxit));
  HYPRE_CALL(HYPRE_StructPFMGSetTol(pc_p, 0.0));
  HYPRE_CALL(HYPRE_StructPFMGSetRelaxType(pc_p, par.p_pc_relax_type));
  HYPRE_CALL(HYPRE_StructPFMGSetRAPType(pc_p, par.p_pc_rap_type));
  HYPRE_CALL(HYPRE_StructPFMGSetPrintLevel(pc_p, 0));
  HYPRE_CALL(HYPRE_StructPFMGSetZeroGuess(pc_p));
  HYPRE_CALL(HYPRE_StructGMRESSetPrecond(solver_p, HYPRE_PtrToStructSolverFcn(HYPRE_StructPFMGSolve), HYPRE_PtrToStructSolverFcn(HYPRE_StructPFMGSetup), pc_p));
  HYPRE_CALL(HYPRE_StructGMRESSetup(solver_p, Ahp, bhp, xhp));

  if (rank == 0) {
    std::printf("============================================================\n");
    std::printf("3D Schaefer-Turek channel with immersed full-height cylinder\n");
    std::printf("Multi-step reused-object device-solve code (CPU assembly, GPU momentum+pressure dense ops)\n");
    std::printf("Grid                  : %d x %d x %d pressure cells\n", nx, ny, nz);
    std::printf("Domain                : [0,%g] x [0,%g] x [0,%g]\n", par.Lx, par.Ly, par.Lz);
    std::printf("Cylinder center/rad   : (%.6f, %.6f), %.6f\n", par.xcirc, par.ycirc, par.Rcirc);
    std::printf("nu                    : %.6e\n", par.nu);
    std::printf("dt                    : %.6e\n", par.dt);
    std::printf("nsteps                : %d\n", par.nsteps);
    std::printf("Pressure cells        : %d\n", Np);
    std::printf("Active u/v/w          : %d / %d / %d\n", Nu, Nv, Nw);
    std::printf("Projection q faces    : %d / %d / %d\n", Nqu, Nqv, Nqw);
    std::printf("Pruned isolated x-y columns: %d\n", n_pruned);
    std::printf("Momentum solver       : explicit row-major Struct GMRES + Struct Jacobi (DEVICE mode, CPU assembly, reused objects, GPU momentum dense ops)\n");
    std::printf("Pressure solver       : explicit row-major Struct GMRES + Struct PFMG on sign-flipped exact A_p (DEVICE mode, CPU assembly, reused objects)\n");
    std::printf("Pressure matrix build : once\n");
    std::printf("VTK writing           : %s", par.write_vtk ? "enabled" : "disabled");
    if (par.write_vtk) std::printf(" (every %d step(s), base=%s)", std::max(1, par.vtk_every), par.vtk_filename.c_str());
    std::printf("\n");
    std::printf("============================================================\n");
    std::printf("step  ramp         max|u|         max|v|         max|w|         divL2          its(u,v,w,p)                relres(u,v,w,p)\n");
  }

  double t_build = 0.0, t_mom_solve = 0.0, t_p_solve = 0.0, t_total0 = MPI_Wtime();

  for (int step = 1; step <= par.nsteps; ++step) {
    double ramp = std::min(1.0, (double)step / (double)par.inlet_ramp_steps);
    for (size_t t = 0; t < uin.size(); ++t) { uin_step[t] = ramp * uin[t]; duin_step[t] = uin_step[t]; }

    double tb = MPI_Wtime();
    int bs=256; int NuFull=(nx+1)*ny*nz, NvFull=nx*(ny+1)*nz, NwFull=nx*ny*(nz+1);
    int Nmax=std::max(NuFull,std::max(NvFull,NwFull));
    k_apply_bc_cleanup<<<(Nmax+bs-1)/bs,bs>>>(du,dv,dw,dactive_u,dactive_v,dactive_w,duin_step,nx,ny,nz);
    k_conv_u<<<(NuFull+bs-1)/bs,bs>>>(dconv_u,du,dv,dw,dactive_u,nx,ny,nz,dx,dy,dz);
    k_conv_v<<<(NvFull+bs-1)/bs,bs>>>(dconv_v,du,dv,dw,dactive_v,nx,ny,nz,dx,dy,dz);
    k_conv_w<<<(NwFull+bs-1)/bs,bs>>>(dconv_w,du,dv,dw,dactive_w,nx,ny,nz,dx,dy,dz);
    CUDA_CALL(cudaDeviceSynchronize());

    CUDA_CALL(cudaMemset(dpvec, 0, ((size_t)Np + 1) * sizeof(double)));
    CUDA_CALL(cudaMemset(dgp, 0, ((size_t)Nq + 1) * sizeof(double)));
    CUDA_CALL(cudaMemset(dgpx, 0, ((size_t)(nx+1)*ny*nz) * sizeof(double)));
    CUDA_CALL(cudaMemset(dgpy, 0, ((size_t)nx*(ny+1)*nz) * sizeof(double)));
    CUDA_CALL(cudaMemset(dgpz, 0, ((size_t)nx*ny*(nz+1)) * sizeof(double)));
    k_pack_pressure_active<<<((nx*ny*nz)+bs-1)/bs,bs>>>(dpvec, dp, dfluid_p, did_p, nx*ny*nz);
    k_spmv_csr_1based<<<(dGproj.nrows+bs-1)/bs,bs>>>(dGproj.nrows, dGproj.rowptr, dGproj.colind, dGproj.vals, dpvec, dgp);
    k_unpack_gp_u<<<(NuFull+bs-1)/bs,bs>>>(dgp, dgpx, did_qu, nx, ny, nz);
    k_unpack_gp_v<<<(NvFull+bs-1)/bs,bs>>>(dgp, dgpy, did_qv, nx, ny, nz, Nqu);
    k_unpack_gp_w<<<(NwFull+bs-1)/bs,bs>>>(dgp, dgpz, did_qw, nx, ny, nz, Nqu, Nqv);
    k_build_u_full<<<(((nx-1)*ny*nz)+bs-1)/bs,bs>>>(rhsu_full,xu_full,du,dconv_u,dgpx,dactive_u,did_u,drhs_u_bc_unit,dscale_u,nx,ny,nz,par.rho,par.dt,par.use_zero_initial_guess!=0);
    k_build_v_full<<<((nx*(ny-1)*nz)+bs-1)/bs,bs>>>(rhsv_full,xv_full,dv,dconv_v,dgpy,dactive_v,did_v,drhs_v_bc,dscale_v,nx,ny,nz,par.rho,par.dt,par.use_zero_initial_guess!=0);
    k_build_w_full<<<((nx*ny*(nz-1))+bs-1)/bs,bs>>>(rhsw_full,xw_full,dw,dconv_w,dgpz,dactive_w,did_w,drhs_w_bc,dscale_w,nx,ny,nz,par.rho,par.dt,par.use_zero_initial_guess!=0);
    CUDA_CALL(cudaDeviceSynchronize());
    t_build += MPI_Wtime() - tb;

    SetVector3D(bhu, nx-1, ny, nz, rhsu_full); SetVector3D(xhu, nx-1, ny, nz, xu_full);
    SetVector3D(bhv, nx, ny-1, nz, rhsv_full); SetVector3D(xhv, nx, ny-1, nz, xv_full);
    SetVector3D(bhw, nx, ny, nz-1, rhsw_full); SetVector3D(xhw, nx, ny, nz-1, xw_full);
    CUDA_CALL(cudaDeviceSynchronize());

    double ts = MPI_Wtime();
    HYPRE_SOLVE_CALL(HYPRE_StructGMRESSolve(solver_u, Ahu, bhu, xhu), "u solve");
    HYPRE_SOLVE_CALL(HYPRE_StructGMRESSolve(solver_v, Ahv, bhv, xhv), "v solve");
    HYPRE_SOLVE_CALL(HYPRE_StructGMRESSolve(solver_w, Ahw, bhw, xhw), "w solve");
    CUDA_CALL(cudaDeviceSynchronize());
    t_mom_solve += MPI_Wtime() - ts;

    int its_u=0, its_v=0, its_w=0, its_p=0;
    double rel_u=0.0, rel_v=0.0, rel_w=0.0, rel_p=0.0;
    HYPRE_CALL(HYPRE_StructGMRESGetNumIterations(solver_u, &its_u));
    HYPRE_CALL(HYPRE_StructGMRESGetFinalRelativeResidualNorm(solver_u, &rel_u));
    HYPRE_CALL(HYPRE_StructGMRESGetNumIterations(solver_v, &its_v));
    HYPRE_CALL(HYPRE_StructGMRESGetFinalRelativeResidualNorm(solver_v, &rel_v));
    HYPRE_CALL(HYPRE_StructGMRESGetNumIterations(solver_w, &its_w));
    HYPRE_CALL(HYPRE_StructGMRESGetFinalRelativeResidualNorm(solver_w, &rel_w));
    GetVector3D(xhu, nx-1, ny, nz, xu_full);
    GetVector3D(xhv, nx, ny-1, nz, xv_full);
    GetVector3D(xhw, nx, ny, nz-1, xw_full);

    if (any_nonfinite_raw0(xu_full, (size_t)(nx-1)*ny*nz) || any_nonfinite_raw0(xv_full, (size_t)nx*(ny-1)*nz) || any_nonfinite_raw0(xw_full, (size_t)nx*ny*(nz-1))) {
      std::fprintf(stderr, "Non-finite momentum solve output at step %d\n", step);
      MPI_Abort(MPI_COMM_WORLD, 911);
    }

    k_scatter_u<<<(NuFull+bs-1)/bs,bs>>>(du_tilde, xu_full, du, nx, ny, nz);
    k_scatter_v<<<(NvFull+bs-1)/bs,bs>>>(dv_tilde, xv_full, dv, nx, ny, nz);
    k_scatter_w<<<(NwFull+bs-1)/bs,bs>>>(dw_tilde, xw_full, dw, nx, ny, nz);
    CUDA_CALL(cudaDeviceSynchronize());
    k_apply_bc_cleanup<<<(Nmax+bs-1)/bs,bs>>>(du_tilde,dv_tilde,dw_tilde,dactive_u,dactive_v,dactive_w,duin_step,nx,ny,nz);
    CUDA_CALL(cudaDeviceSynchronize());

    CUDA_CALL(cudaMemset(dqtilde, 0, ((size_t)Nq + 1) * sizeof(double)));
    k_pack_projection_q_u<<<(NuFull+bs-1)/bs,bs>>>(dqtilde, du_tilde, did_qu, nx, ny, nz);
    k_pack_projection_q_v<<<(NvFull+bs-1)/bs,bs>>>(dqtilde, dv_tilde, did_qv, nx, ny, nz, Nqu);
    k_pack_projection_q_w<<<(NwFull+bs-1)/bs,bs>>>(dqtilde, dw_tilde, did_qw, nx, ny, nz, Nqu, Nqv);
    CUDA_CALL(cudaMemset(dDqt, 0, ((size_t)Np + 1) * sizeof(double)));
    k_spmv_csr_1based<<<(dDproj.nrows+bs-1)/bs,bs>>>(dDproj.nrows, dDproj.rowptr, dDproj.colind, dDproj.vals, dqtilde, dDqt);
    k_build_pressure_rhs_full<<<((nx*ny*nz)+bs-1)/bs,bs>>>(rhsp_full, xp_full, dDqt, ddiv_bc_unit, dfluid_p, did_p, nx*ny*nz, par.rho, par.dt, ramp);
    CUDA_CALL(cudaDeviceSynchronize());
    SetVector3D(bhp, nx, ny, nz, rhsp_full);
    SetVector3D(xhp, nx, ny, nz, xp_full);
    CUDA_CALL(cudaDeviceSynchronize());

    ts = MPI_Wtime();
    HYPRE_SOLVE_CALL(HYPRE_StructGMRESSolve(solver_p, Ahp, bhp, xhp), "p solve");
    CUDA_CALL(cudaDeviceSynchronize());
    t_p_solve += MPI_Wtime() - ts;
    HYPRE_CALL(HYPRE_StructGMRESGetNumIterations(solver_p, &its_p));
    HYPRE_CALL(HYPRE_StructGMRESGetFinalRelativeResidualNorm(solver_p, &rel_p));
    GetVector3D(xhp, nx, ny, nz, xp_full);
    if (any_nonfinite_raw0(xp_full, (size_t)nx*ny*nz)) {
      std::fprintf(stderr, "Non-finite pressure solve output at step %d\n", step);
      MPI_Abort(MPI_COMM_WORLD, 912);
    }

    CUDA_CALL(cudaMemset(dphi_vec, 0, ((size_t)Np + 1) * sizeof(double)));
    k_pack_phi_active<<<((nx*ny*nz)+bs-1)/bs,bs>>>(dphi_vec, xp_full, dfluid_p, did_p, nx*ny*nz);
    CUDA_CALL(cudaMemset(dgphi, 0, ((size_t)Nq + 1) * sizeof(double)));
    k_spmv_csr_1based<<<(dGproj.nrows+bs-1)/bs,bs>>>(dGproj.nrows, dGproj.rowptr, dGproj.colind, dGproj.vals, dphi_vec, dgphi);
    k_apply_q_correction<<<((Nq+1)+bs-1)/bs,bs>>>(dqtilde, dgphi, par.dt / par.rho, Nq + 1);
    CUDA_CALL(cudaMemcpy(du, du_tilde, (size_t)NuFull * sizeof(double), cudaMemcpyDeviceToDevice));
    CUDA_CALL(cudaMemcpy(dv, dv_tilde, (size_t)NvFull * sizeof(double), cudaMemcpyDeviceToDevice));
    CUDA_CALL(cudaMemcpy(dw, dw_tilde, (size_t)NwFull * sizeof(double), cudaMemcpyDeviceToDevice));
    k_unpack_projection_q_u<<<(NuFull+bs-1)/bs,bs>>>(dqtilde, du, did_qu, nx, ny, nz);
    k_unpack_projection_q_v<<<(NvFull+bs-1)/bs,bs>>>(dqtilde, dv, did_qv, nx, ny, nz, Nqu);
    k_unpack_projection_q_w<<<(NwFull+bs-1)/bs,bs>>>(dqtilde, dw, did_qw, nx, ny, nz, Nqu, Nqv);
    k_apply_bc_cleanup<<<(Nmax+bs-1)/bs,bs>>>(du,dv,dw,dactive_u,dactive_v,dactive_w,duin_step,nx,ny,nz);
    k_add_pressure_phi<<<((nx*ny*nz)+bs-1)/bs,bs>>>(dp, dphi_vec, dfluid_p, did_p, nx*ny*nz);
    CUDA_CALL(cudaDeviceSynchronize());
    for(size_t i=0;i<u.size();++i) u[i]=du[i]; for(size_t i=0;i<v.size();++i) v[i]=dv[i]; for(size_t i=0;i<w.size();++i) w[i]=dw[i]; for(size_t i=0;i<p.size();++i) p[i]=dp[i];

    CUDA_CALL(cudaMemset(ddivv, 0, ((size_t)Np + 1) * sizeof(double)));
    k_spmv_csr_1based<<<(dDproj.nrows+bs-1)/bs,bs>>>(dDproj.nrows, dDproj.rowptr, dDproj.colind, dDproj.vals, dqtilde, ddivv);
    CUDA_CALL(cudaDeviceSynchronize());
    double div2 = 0.0; int ndiv = 0;
    for (int i = 1; i <= Np; ++i) { double d = ddivv[i] + ramp * div_bc_unit[(size_t)i]; div2 += d*d; ++ndiv; }
    double div_l2 = std::sqrt(div2 / std::max(1, ndiv));
    double umax = 0.0, vmax = 0.0, wmax = 0.0;
    for (double x : u) umax = std::max(umax, std::abs(x));
    for (double x : v) vmax = std::max(vmax, std::abs(x));
    for (double x : w) wmax = std::max(wmax, std::abs(x));

    if (rank == 0 && (step == 1 || step == par.nsteps || (par.print_every > 0 && step % par.print_every == 0))) {
      std::printf("%4d  %.6f  %.6e  %.6e  %.6e  %.6e  (%d,%d,%d,%d)  (%.3e,%.3e,%.3e,%.3e)\n",
                  step, ramp, umax, vmax, wmax, div_l2, its_u, its_v, its_w, its_p, rel_u, rel_v, rel_w, rel_p);
    }

    if (par.write_vtk && rank == 0) {
      int every = std::max(1, par.vtk_every);
      if (step == 1 || step == par.nsteps || (step % every == 0)) {
        write_vtk_cellcentered(vtk_filename_with_step(par.vtk_filename, step), u, v, w, p, fluid_p, nx, ny, nz, par.Lx, par.Ly, par.Lz);
      }
    }
  }

  if (par.write_vtk && rank == 0 && par.nsteps <= 0) write_vtk_cellcentered(par.vtk_filename, u, v, w, p, fluid_p, nx, ny, nz, par.Lx, par.Ly, par.Lz);

  if (rank == 0) {
    double total = MPI_Wtime() - t_total0;
    std::printf("============================================================\n");
    std::printf("Timing summary\n");
    std::printf("momentum build time    : %.6e s\n", t_build);
    std::printf("momentum solve time    : %.6e s\n", t_mom_solve);
    std::printf("pressure solve time    : %.6e s\n", t_p_solve);
    std::printf("total wall time        : %.6e s\n", total);
    std::printf("============================================================\n");
  }

  HYPRE_CALL(HYPRE_StructGMRESDestroy(solver_u)); HYPRE_CALL(HYPRE_StructJacobiDestroy(pc_u));
  HYPRE_CALL(HYPRE_StructGMRESDestroy(solver_v)); HYPRE_CALL(HYPRE_StructJacobiDestroy(pc_v));
  HYPRE_CALL(HYPRE_StructGMRESDestroy(solver_w)); HYPRE_CALL(HYPRE_StructJacobiDestroy(pc_w));
  HYPRE_CALL(HYPRE_StructGMRESDestroy(solver_p)); HYPRE_CALL(HYPRE_StructPFMGDestroy(pc_p));

  HYPRE_CALL(HYPRE_StructVectorDestroy(bhu)); HYPRE_CALL(HYPRE_StructVectorDestroy(xhu));
  HYPRE_CALL(HYPRE_StructVectorDestroy(bhv)); HYPRE_CALL(HYPRE_StructVectorDestroy(xhv));
  HYPRE_CALL(HYPRE_StructVectorDestroy(bhw)); HYPRE_CALL(HYPRE_StructVectorDestroy(xhw));
  HYPRE_CALL(HYPRE_StructVectorDestroy(bhp)); HYPRE_CALL(HYPRE_StructVectorDestroy(xhp));
  HYPRE_CALL(HYPRE_StructMatrixDestroy(Ahu)); HYPRE_CALL(HYPRE_StructMatrixDestroy(Ahv));
  HYPRE_CALL(HYPRE_StructMatrixDestroy(Ahw)); HYPRE_CALL(HYPRE_StructMatrixDestroy(Ahp));
  HYPRE_CALL(HYPRE_StructStencilDestroy(st));
  HYPRE_CALL(HYPRE_StructGridDestroy(gu)); HYPRE_CALL(HYPRE_StructGridDestroy(gv));
  HYPRE_CALL(HYPRE_StructGridDestroy(gw)); HYPRE_CALL(HYPRE_StructGridDestroy(gpgrid));

  CUDA_CALL(cudaFree(rhsu_full)); CUDA_CALL(cudaFree(xu_full));
  CUDA_CALL(cudaFree(rhsv_full)); CUDA_CALL(cudaFree(xv_full));
  CUDA_CALL(cudaFree(rhsw_full)); CUDA_CALL(cudaFree(xw_full));
  CUDA_CALL(cudaFree(rhsp_full)); CUDA_CALL(cudaFree(xp_full));
  CUDA_CALL(cudaFree(Ahu_vals_rowmajor_d)); CUDA_CALL(cudaFree(Ahv_vals_rowmajor_d));
  CUDA_CALL(cudaFree(Ahw_vals_rowmajor_d)); CUDA_CALL(cudaFree(Ahp_vals_rowmajor_d));
  CUDA_CALL(cudaFree(du)); CUDA_CALL(cudaFree(dv)); CUDA_CALL(cudaFree(dw)); CUDA_CALL(cudaFree(dp));
  CUDA_CALL(cudaFree(du_tilde)); CUDA_CALL(cudaFree(dv_tilde)); CUDA_CALL(cudaFree(dw_tilde)); CUDA_CALL(cudaFree(duin_step));
  CUDA_CALL(cudaFree(dconv_u)); CUDA_CALL(cudaFree(dconv_v)); CUDA_CALL(cudaFree(dconv_w)); CUDA_CALL(cudaFree(dgpx)); CUDA_CALL(cudaFree(dgpy)); CUDA_CALL(cudaFree(dgpz));
  CUDA_CALL(cudaFree(dactive_u)); CUDA_CALL(cudaFree(dactive_v)); CUDA_CALL(cudaFree(dactive_w)); CUDA_CALL(cudaFree(dfluid_p));
  CUDA_CALL(cudaFree(did_u)); CUDA_CALL(cudaFree(did_v)); CUDA_CALL(cudaFree(did_w)); CUDA_CALL(cudaFree(did_p)); CUDA_CALL(cudaFree(did_qu)); CUDA_CALL(cudaFree(did_qv)); CUDA_CALL(cudaFree(did_qw));
  CUDA_CALL(cudaFree(dscale_u)); CUDA_CALL(cudaFree(dscale_v)); CUDA_CALL(cudaFree(dscale_w)); CUDA_CALL(cudaFree(drhs_u_bc_unit)); CUDA_CALL(cudaFree(drhs_v_bc)); CUDA_CALL(cudaFree(drhs_w_bc)); CUDA_CALL(cudaFree(ddiv_bc_unit));
  CUDA_CALL(cudaFree(dpvec)); CUDA_CALL(cudaFree(dgp)); CUDA_CALL(cudaFree(dqtilde)); CUDA_CALL(cudaFree(dDqt)); CUDA_CALL(cudaFree(dphi_vec)); CUDA_CALL(cudaFree(dgphi)); CUDA_CALL(cudaFree(ddivv));
  destroy_managed_csr(dDproj); destroy_managed_csr(dGproj);

  HYPRE_CALL(HYPRE_Finalize());
  MPI_Finalize();
  return 0;
}
