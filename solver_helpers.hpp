  for (int s = 0; s < 7; ++s) c[s] = 0.0;
  unexpected = 0;
  int sx = nx + 1, plane = sx * ny;
  int flat_row = flat_of_id_u[(size_t)aid];
  int kr = flat_row / plane;
  int remr = flat_row - kr * plane;
  int jr = remr / sx;
  int ir = remr - jr * sx; // face index in full (1..nx-1)
  for (const auto &kv : Aactive.rows[(size_t)(aid - 1)]) {
    int cid = kv.first;
    int flat_col = flat_of_id_u[(size_t)cid];
    int kc = flat_col / plane;
    int remc = flat_col - kc * plane;
    int jc = remc / sx;
    int ic = remc - jc * sx;
    int di = ic - ir, dj = jc - jr, dk = kc - kr;
    if      (di == 0 && dj == 0 && dk == 0) c[0] += kv.second;
    else if (di == -1 && dj == 0 && dk == 0) c[1] += kv.second;
    else if (di == +1 && dj == 0 && dk == 0) c[2] += kv.second;
    else if (di == 0 && dj == -1 && dk == 0) c[3] += kv.second;
    else if (di == 0 && dj == +1 && dk == 0) c[4] += kv.second;
    else if (di == 0 && dj == 0 && dk == -1) c[5] += kv.second;
    else if (di == 0 && dj == 0 && dk == +1) c[6] += kv.second;
    else ++unexpected;
  }
}

static void get_u_stencil_from_entrymajor_intended(const std::vector<HYPRE_Complex> &vals, int row, size_t nrows, double c[7])
{
  for (int s = 0; s < 7; ++s) c[s] = vals[(size_t)s * nrows + (size_t)row];
}

static void get_u_stencil_from_rowmajor_buffer(const std::vector<HYPRE_Complex> &vals, int row, double c[7])
{
  for (int s = 0; s < 7; ++s) c[s] = vals[(size_t)7 * (size_t)row + (size_t)s];
}

static double max_abs_diff7(const double a[7], const double b[7])
{
  double m = 0.0;
  for (int s = 0; s < 7; ++s) m = std::max(m, std::abs(a[s] - b[s]));
  return m;
}

static void print_stencil7(const char *label, const double c[7])
{
  std::printf("  %-28s diag=%+.16e west=%+.16e east=%+.16e south=%+.16e north=%+.16e bot=%+.16e top=%+.16e\n",
              label, c[0], c[1], c[2], c[3], c[4], c[5], c[6]);
}

static void embed_u_matrix_to_rowmajor_3d(const SparseMatrix &Aactive, const std::vector<int> &flat_of_id_u,
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
        HYPRE_Complex *a = &vals[(size_t)7 * (size_t)row];
        if (aid <= 0) { a[0] = 1.0; continue; }
        for (const auto &kv : Aactive.rows[(size_t)(aid - 1)]) {
          int cid = kv.first;
          int flat = flat_of_id_u[(size_t)cid];
          int kc = flat / plane;
          int rem = flat - kc * plane;
          int jc = rem / sx;
          int ic = rem - jc * sx;
          int di = ic - (iu0 + 1), dj = jc - j, dk = kc - k;
          if      (di == 0 && dj == 0 && dk == 0) a[0] += kv.second;
          else if (di == -1 && dj == 0 && dk == 0) a[1] += kv.second;
          else if (di == +1 && dj == 0 && dk == 0) a[2] += kv.second;
          else if (di == 0 && dj == -1 && dk == 0) a[3] += kv.second;
          else if (di == 0 && dj == +1 && dk == 0) a[4] += kv.second;
          else if (di == 0 && dj == 0 && dk == -1) a[5] += kv.second;
          else if (di == 0 && dj == 0 && dk == +1) a[6] += kv.second;
        }
      }
}

static double residual_l2_from_active_u(const SparseMatrix &A, const std::vector<double> &x, const std::vector<double> &b)
{
  std::vector<double> Ax = spmv(A, x);
  double s = 0.0;
  for (size_t i = 1; i < Ax.size(); ++i) {
    double r = Ax[i] - b[i];
    s += r * r;
  }
  return std::sqrt(s);
}

static bool any_nonfinite_active_u(const std::vector<double> &x)
{
  for (size_t i = 1; i < x.size(); ++i) if (!std::isfinite(x[i])) return true;
  return false;
}

static void audit_u_embedding(const SparseMatrix &Au, const std::vector<int> &flat_of_id_u, const std::vector<int> &id_u,
                              const std::vector<HYPRE_Complex> &entrymajor_vals,
                              const std::vector<HYPRE_Complex> &rowmajor_vals,
                              const Params &par)
{
  const int nx = par.nx, ny = par.ny, nz = par.nz;
  const int nxu = nx - 1;
  const size_t nrows = (size_t)nxu * (size_t)ny * (size_t)nz;
  const double dx = par.Lx / nx, dy = par.Ly / ny;
  int bad_hypre_view = 0, bad_rowmajor = 0, bad_entry_intended = 0, unexpected_rows = 0;
  double worst_hypre = 0.0, worst_row = 0.0, worst_entry = 0.0;
  struct RowRec { int aid,row,ifull,j,k,unexpected; double dist,diff_hypre,diff_row,diff_entry; double want[7], hypre[7], rowp[7], intended[7]; };
  std::vector<RowRec> picks;
  for (int k = 0; k < nz; ++k)
    for (int j = 0; j < ny; ++j)
      for (int iu0 = 0; iu0 < nxu; ++iu0) {
        int aid = id_u[idxU3(iu0 + 1, j, k, nx, ny)];
        if (aid <= 0) continue;
        double want[7], intended[7], hypre[7], rowp[7];
        int unexpected = 0;
        get_u_stencil_from_active_row(Au, aid, flat_of_id_u, nx, ny, want, unexpected);
        int row = idxUu(iu0, j, k, nxu, ny);
        get_u_stencil_from_entrymajor_intended(entrymajor_vals, row, nrows, intended);
        get_u_stencil_from_rowmajor_buffer(entrymajor_vals, row, hypre); // how HYPRE would read the helper buffer if it expects point-major
        get_u_stencil_from_rowmajor_buffer(rowmajor_vals, row, rowp);
        double diff_entry = max_abs_diff7(want, intended);
        double diff_hypre = max_abs_diff7(want, hypre);
        double diff_row = max_abs_diff7(want, rowp);
        worst_entry = std::max(worst_entry, diff_entry);
        worst_hypre = std::max(worst_hypre, diff_hypre);
        worst_row = std::max(worst_row, diff_row);
        if (diff_entry > 1.0e-12) ++bad_entry_intended;
        if (diff_hypre > 1.0e-12) ++bad_hypre_view;
        if (diff_row > 1.0e-12) ++bad_rowmajor;
        if (unexpected) ++unexpected_rows;
        double x = (iu0 + 1) * dx;
        double y = (j + 0.5) * dy;
        double dist = std::abs(std::sqrt((x - par.xcirc) * (x - par.xcirc) + (y - par.ycirc) * (y - par.ycirc)) - par.Rcirc);
        bool near_cyl = dist <= 1.5 * std::max(dx, dy);
        bool suspicious = near_cyl || unexpected || diff_hypre > 1.0e-12 || diff_row > 1.0e-12 || diff_entry > 1.0e-12 || (int)Au.rows[(size_t)(aid - 1)].size() < 7;
        if (suspicious && (int)picks.size() < 24) {
          RowRec rec{};
          rec.aid = aid; rec.row = row; rec.ifull = iu0 + 1; rec.j = j; rec.k = k; rec.unexpected = unexpected;
          rec.dist = dist; rec.diff_hypre = diff_hypre; rec.diff_row = diff_row; rec.diff_entry = diff_entry;
          for (int s = 0; s < 7; ++s) { rec.want[s] = want[s]; rec.hypre[s] = hypre[s]; rec.rowp[s] = rowp[s]; rec.intended[s] = intended[s]; }
          picks.push_back(rec);
        }
      }

  std::printf("============================================================\n");
  std::printf("U-EMBEDDING AUDIT\n");
  std::printf("============================================================\n");
  std::printf("Rows audited                         : %d\n", *std::max_element(id_u.begin(), id_u.end()));
  std::printf("Rows with unexpected active offsets  : %d\n", unexpected_rows);
  std::printf("Rows where helper buffer (intended) mismatches active row : %d\n", bad_entry_intended);
  std::printf("Rows where helper buffer as HYPRE row-major view mismatches active row : %d\n", bad_hypre_view);
  std::printf("Rows where row-major buffer mismatches active row : %d\n", bad_rowmajor);
  std::printf("Worst diff helper intended          : %.16e\n", worst_entry);
  std::printf("Worst diff helper as HYPRE sees it  : %.16e\n", worst_hypre);
  std::printf("Worst diff row-major                : %.16e\n", worst_row);
  for (const auto &rec : picks) {
    std::printf("------------------------------------------------------------\n");
    std::printf("aid=%d row=%d face(i=%d,j=%d,k=%d) dist_to_cyl=%.6e active_nnz=%d unexpected=%d\n",
                rec.aid, rec.row, rec.ifull, rec.j + 1, rec.k + 1, rec.dist,
                (int)Au.rows[(size_t)(rec.aid - 1)].size(), rec.unexpected);
    print_stencil7("active-row target", rec.want);
    print_stencil7("helper entry-major intended", rec.intended);
    print_stencil7("helper buffer as HYPRE sees", rec.hypre);
    print_stencil7("row-major buffer", rec.rowp);
    std::printf("  maxdiff(helper intended) = %.16e\n", rec.diff_entry);
    std::printf("  maxdiff(helper as HYPRE) = %.16e\n", rec.diff_hypre);
    std::printf("  maxdiff(row-major)       = %.16e\n", rec.diff_row);
  }
  std::printf("============================================================\n");
}

static void run_u_solver_case(const char *label, HYPRE_StructGrid gu, HYPRE_StructStencil st,
                              int nx, int ny, int nz,
                              HYPRE_Complex *Avals,
                              HYPRE_Complex *rhsu_full, HYPRE_Complex *xu_full,
                              const SparseMatrix &Au, const std::vector<int> &id_u,
                              const Params &par)
{
  HYPRE_StructMatrix Ahu = CreateMatrix3D(gu, st, nx - 1, ny, nz, Avals);
  HYPRE_StructVector bhu = CreateVector3D(gu), xhu = CreateVector3D(gu);
  SetVector3D(bhu, nx - 1, ny, nz, rhsu_full);
  SetVector3D(xhu, nx - 1, ny, nz, xu_full);

  HYPRE_StructSolver solver_u, pc_u;
  HYPRE_CALL(HYPRE_StructGMRESCreate(MPI_COMM_WORLD, &solver_u));
  HYPRE_CALL(HYPRE_StructGMRESSetTol(solver_u, par.vel_tol));
  HYPRE_CALL(HYPRE_StructGMRESSetMaxIter(solver_u, par.vel_maxit));
  HYPRE_CALL(HYPRE_StructGMRESSetKDim(solver_u, par.vel_kdim));
  HYPRE_CALL(HYPRE_StructGMRESSetPrintLevel(solver_u, par.vel_print));
  HYPRE_CALL(HYPRE_StructJacobiCreate(MPI_COMM_WORLD, &pc_u));
  HYPRE_CALL(HYPRE_StructJacobiSetMaxIter(pc_u, par.vel_jacobi_iters));
  HYPRE_CALL(HYPRE_StructJacobiSetZeroGuess(pc_u));
  HYPRE_CALL(HYPRE_StructGMRESSetPrecond(solver_u,
      HYPRE_PtrToStructSolverFcn(HYPRE_StructJacobiSolve),
      HYPRE_PtrToStructSolverFcn(HYPRE_StructJacobiSetup), pc_u));
  HYPRE_CALL(HYPRE_StructGMRESSetup(solver_u, Ahu, bhu, xhu));
  HYPRE_SOLVE_CALL(HYPRE_StructGMRESSolve(solver_u, Ahu, bhu, xhu), label);
  int its_u = 0; double rel_u = 0.0;
  HYPRE_CALL(HYPRE_StructGMRESGetNumIterations(solver_u, &its_u));
  HYPRE_CALL(HYPRE_StructGMRESGetFinalRelativeResidualNorm(solver_u, &rel_u));
  GetVector3D(xhu, nx - 1, ny, nz, xu_full);
  std::vector<double> u_vec_act;
  extract_active_from_full_u(u_vec_act, xu_full, id_u, nx, ny, nz);
  std::printf("============================================================\n");
  std::printf("%s\n", label);
  std::printf("============================================================\n");
  std::printf("GMRES its / relres : %d / %.16e\n", its_u, rel_u);
  print_stats_raw0("xhu full-box", xu_full, (size_t)(nx - 1) * (size_t)ny * (size_t)nz);
  print_stats_vec1("u_vec active", u_vec_act);
  HYPRE_CALL(HYPRE_StructGMRESDestroy(solver_u));
  HYPRE_CALL(HYPRE_StructJacobiDestroy(pc_u));
  HYPRE_CALL(HYPRE_StructVectorDestroy(bhu));
  HYPRE_CALL(HYPRE_StructVectorDestroy(xhu));
  HYPRE_CALL(HYPRE_StructMatrixDestroy(Ahu));
}




static void embed_v_matrix_to_rowmajor_3d(const SparseMatrix &Aactive, const std::vector<int> &flat_of_id_v,
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
        HYPRE_Complex *a = &vals[(size_t)7 * (size_t)row];
        if (aid <= 0) { a[0] = 1.0; continue; }
        for (const auto &kv : Aactive.rows[(size_t)(aid - 1)]) {
          int cid = kv.first;
          int flat = flat_of_id_v[(size_t)cid];
          int kc = flat / plane;
          int rem = flat - kc * plane;
          int jc = rem / sx;
          int ic = rem - jc * sx;
          int di = ic - i, dj = jc - (jv + 1), dk = kc - k;
          if      (di == 0 && dj == 0 && dk == 0) a[0] += kv.second;
          else if (di == -1 && dj == 0 && dk == 0) a[1] += kv.second;
          else if (di == +1 && dj == 0 && dk == 0) a[2] += kv.second;
          else if (di == 0 && dj == -1 && dk == 0) a[3] += kv.second;
          else if (di == 0 && dj == +1 && dk == 0) a[4] += kv.second;
          else if (di == 0 && dj == 0 && dk == -1) a[5] += kv.second;
          else if (di == 0 && dj == 0 && dk == +1) a[6] += kv.second;
        }
      }
}

static void embed_w_matrix_to_rowmajor_3d(const SparseMatrix &Aactive, const std::vector<int> &flat_of_id_w,
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
        HYPRE_Complex *a = &vals[(size_t)7 * (size_t)row];
        if (aid <= 0) { a[0] = 1.0; continue; }
        for (const auto &kv : Aactive.rows[(size_t)(aid - 1)]) {
          int cid = kv.first;
          int flat = flat_of_id_w[(size_t)cid];
          int kc = flat / plane;
          int rem = flat - kc * plane;
          int jc = rem / sx;
          int ic = rem - jc * sx;
          int di = ic - i, dj = jc - j, dk = kc - (kw0 + 1);
          if      (di == 0 && dj == 0 && dk == 0) a[0] += kv.second;
          else if (di == -1 && dj == 0 && dk == 0) a[1] += kv.second;
          else if (di == +1 && dj == 0 && dk == 0) a[2] += kv.second;
          else if (di == 0 && dj == -1 && dk == 0) a[3] += kv.second;
          else if (di == 0 && dj == +1 && dk == 0) a[4] += kv.second;
          else if (di == 0 && dj == 0 && dk == -1) a[5] += kv.second;
          else if (di == 0 && dj == 0 && dk == +1) a[6] += kv.second;
        }
      }
}

static void embed_p_matrix_to_rowmajor_3d(const SparseMatrix &Aactive, const std::vector<int> &flat_of_id_p,
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
        HYPRE_Complex *a = &vals[(size_t)7 * (size_t)row];
        if (aid <= 0) { a[0] = 1.0; continue; }
        for (const auto &kv : Aactive.rows[(size_t)(aid - 1)]) {
          int cid = kv.first;
          int flat = flat_of_id_p[(size_t)cid];
          int kc = flat / plane;
          int rem = flat - kc * plane;
          int jc = rem / sx;
          int ic = rem - jc * sx;
          int di = ic - i, dj = jc - j, dk = kc - k;
          if      (di == 0 && dj == 0 && dk == 0) a[0] += kv.second;
          else if (di == -1 && dj == 0 && dk == 0) a[1] += kv.second;
          else if (di == +1 && dj == 0 && dk == 0) a[2] += kv.second;
          else if (di == 0 && dj == -1 && dk == 0) a[3] += kv.second;
          else if (di == 0 && dj == +1 && dk == 0) a[4] += kv.second;
          else if (di == 0 && dj == 0 && dk == -1) a[5] += kv.second;
          else if (di == 0 && dj == 0 && dk == +1) a[6] += kv.second;
        }
      }
}

static bool any_nonfinite_raw0(const HYPRE_Complex *x, size_t n)
{
  for (size_t i = 0; i < n; ++i) if (!std::isfinite((double)x[i])) return true;
  return false;
}

static void run_v_solver_case(const char *label, HYPRE_StructGrid gv, HYPRE_StructStencil st,
                              int nx, int ny, int nz,
                              HYPRE_Complex *Avals,
                              HYPRE_Complex *rhsv_full, HYPRE_Complex *xv_full,
                              const std::vector<int> &id_v,
                              const Params &par,
                              int &its_v, double &rel_v)
{
  HYPRE_StructMatrix Ahv = CreateMatrix3D(gv, st, nx, ny - 1, nz, Avals);
  HYPRE_StructVector bhv = CreateVector3D(gv), xhv = CreateVector3D(gv);
  SetVector3D(bhv, nx, ny - 1, nz, rhsv_full);
  SetVector3D(xhv, nx, ny - 1, nz, xv_full);
  HYPRE_StructSolver solver_v, pc_v;
  HYPRE_CALL(HYPRE_StructGMRESCreate(MPI_COMM_WORLD, &solver_v));
  HYPRE_CALL(HYPRE_StructGMRESSetTol(solver_v, par.vel_tol));
  HYPRE_CALL(HYPRE_StructGMRESSetMaxIter(solver_v, par.vel_maxit));
  HYPRE_CALL(HYPRE_StructGMRESSetKDim(solver_v, par.vel_kdim));
  HYPRE_CALL(HYPRE_StructGMRESSetPrintLevel(solver_v, par.vel_print));
  HYPRE_CALL(HYPRE_StructJacobiCreate(MPI_COMM_WORLD, &pc_v));
  HYPRE_CALL(HYPRE_StructJacobiSetMaxIter(pc_v, par.vel_jacobi_iters));
  HYPRE_CALL(HYPRE_StructJacobiSetZeroGuess(pc_v));
  HYPRE_CALL(HYPRE_StructGMRESSetPrecond(solver_v,
      HYPRE_PtrToStructSolverFcn(HYPRE_StructJacobiSolve),
      HYPRE_PtrToStructSolverFcn(HYPRE_StructJacobiSetup), pc_v));
  HYPRE_CALL(HYPRE_StructGMRESSetup(solver_v, Ahv, bhv, xhv));
  HYPRE_SOLVE_CALL(HYPRE_StructGMRESSolve(solver_v, Ahv, bhv, xhv), label);
  HYPRE_CALL(HYPRE_StructGMRESGetNumIterations(solver_v, &its_v));
  HYPRE_CALL(HYPRE_StructGMRESGetFinalRelativeResidualNorm(solver_v, &rel_v));
  GetVector3D(xhv, nx, ny - 1, nz, xv_full);
  std::vector<double> v_vec_act; extract_active_from_full_v(v_vec_act, xv_full, id_v, nx, ny, nz);
  std::printf("============================================================\n");
  std::printf("%s\n", label);
  std::printf("============================================================\n");
  std::printf("GMRES its / relres : %d / %.16e\n", its_v, rel_v);
  print_stats_raw0("xhv full-box", xv_full, (size_t)nx * (size_t)(ny - 1) * (size_t)nz);
  print_stats_vec1("v_vec active", v_vec_act);
  HYPRE_CALL(HYPRE_StructGMRESDestroy(solver_v)); HYPRE_CALL(HYPRE_StructJacobiDestroy(pc_v));
  HYPRE_CALL(HYPRE_StructVectorDestroy(bhv)); HYPRE_CALL(HYPRE_StructVectorDestroy(xhv)); HYPRE_CALL(HYPRE_StructMatrixDestroy(Ahv));
}

static void run_w_solver_case(const char *label, HYPRE_StructGrid gw, HYPRE_StructStencil st,
                              int nx, int ny, int nz,
                              HYPRE_Complex *Avals,
                              HYPRE_Complex *rhsw_full, HYPRE_Complex *xw_full,
                              const std::vector<int> &id_w,
                              const Params &par,
                              int &its_w, double &rel_w)
{
  HYPRE_StructMatrix Ahw = CreateMatrix3D(gw, st, nx, ny, nz - 1, Avals);
  HYPRE_StructVector bhw = CreateVector3D(gw), xhw = CreateVector3D(gw);
  SetVector3D(bhw, nx, ny, nz - 1, rhsw_full);
  SetVector3D(xhw, nx, ny, nz - 1, xw_full);
  HYPRE_StructSolver solver_w, pc_w;
  HYPRE_CALL(HYPRE_StructGMRESCreate(MPI_COMM_WORLD, &solver_w));
  HYPRE_CALL(HYPRE_StructGMRESSetTol(solver_w, par.vel_tol));
  HYPRE_CALL(HYPRE_StructGMRESSetMaxIter(solver_w, par.vel_maxit));
  HYPRE_CALL(HYPRE_StructGMRESSetKDim(solver_w, par.vel_kdim));
  HYPRE_CALL(HYPRE_StructGMRESSetPrintLevel(solver_w, par.vel_print));
  HYPRE_CALL(HYPRE_StructJacobiCreate(MPI_COMM_WORLD, &pc_w));
  HYPRE_CALL(HYPRE_StructJacobiSetMaxIter(pc_w, par.vel_jacobi_iters));
  HYPRE_CALL(HYPRE_StructJacobiSetZeroGuess(pc_w));
  HYPRE_CALL(HYPRE_StructGMRESSetPrecond(solver_w,
      HYPRE_PtrToStructSolverFcn(HYPRE_StructJacobiSolve),
      HYPRE_PtrToStructSolverFcn(HYPRE_StructJacobiSetup), pc_w));
  HYPRE_CALL(HYPRE_StructGMRESSetup(solver_w, Ahw, bhw, xhw));
  HYPRE_SOLVE_CALL(HYPRE_StructGMRESSolve(solver_w, Ahw, bhw, xhw), label);
  HYPRE_CALL(HYPRE_StructGMRESGetNumIterations(solver_w, &its_w));
  HYPRE_CALL(HYPRE_StructGMRESGetFinalRelativeResidualNorm(solver_w, &rel_w));
  GetVector3D(xhw, nx, ny, nz - 1, xw_full);
  std::vector<double> w_vec_act; extract_active_from_full_w(w_vec_act, xw_full, id_w, nx, ny, nz);
  std::printf("============================================================\n");
  std::printf("%s\n", label);
  std::printf("============================================================\n");
  std::printf("GMRES its / relres : %d / %.16e\n", its_w, rel_w);
  print_stats_raw0("xhw full-box", xw_full, (size_t)nx * (size_t)ny * (size_t)(nz - 1));
  print_stats_vec1("w_vec active", w_vec_act);
  HYPRE_CALL(HYPRE_StructGMRESDestroy(solver_w)); HYPRE_CALL(HYPRE_StructJacobiDestroy(pc_w));
  HYPRE_CALL(HYPRE_StructVectorDestroy(bhw)); HYPRE_CALL(HYPRE_StructVectorDestroy(xhw)); HYPRE_CALL(HYPRE_StructMatrixDestroy(Ahw));
}

static void run_p_solver_case(const char *label, HYPRE_StructGrid gpgrid, HYPRE_StructStencil st,
                              int nx, int ny, int nz,
                              HYPRE_Complex *Avals,
                              HYPRE_Complex *rhsp_full, HYPRE_Complex *xp_full,
                              const std::vector<int> &id_p,
                              const Params &par,
                              int &its_p, double &rel_p, std::vector<double> &phi_vec)
{
  HYPRE_StructMatrix Ahp = CreateMatrix3D(gpgrid, st, nx, ny, nz, Avals);
  HYPRE_StructVector bhp = CreateVector3D(gpgrid), xhp = CreateVector3D(gpgrid);
  SetVector3D(bhp, nx, ny, nz, rhsp_full);
  SetVector3D(xhp, nx, ny, nz, xp_full);
  HYPRE_StructSolver solver_p, pc_p;
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
  HYPRE_CALL(HYPRE_StructGMRESSetPrecond(solver_p,
      HYPRE_PtrToStructSolverFcn(HYPRE_StructPFMGSolve),
      HYPRE_PtrToStructSolverFcn(HYPRE_StructPFMGSetup), pc_p));
  HYPRE_CALL(HYPRE_StructGMRESSetup(solver_p, Ahp, bhp, xhp));
  HYPRE_SOLVE_CALL(HYPRE_StructGMRESSolve(solver_p, Ahp, bhp, xhp), label);
  HYPRE_CALL(HYPRE_StructGMRESGetNumIterations(solver_p, &its_p));
  HYPRE_CALL(HYPRE_StructGMRESGetFinalRelativeResidualNorm(solver_p, &rel_p));
  GetVector3D(xhp, nx, ny, nz, xp_full);
  phi_vec.assign((size_t)(*std::max_element(id_p.begin(), id_p.end())) + 1, 0.0);
  for (int k = 0; k < nz; ++k)
    for (int j = 0; j < ny; ++j)
      for (int i = 0; i < nx; ++i) {
        int id = id_p[idxP3(i, j, k, nx, ny)];
        if (id) phi_vec[(size_t)id] = xp_full[idxPfull(i, j, k, nx, ny)];
      }
  std::printf("============================================================\n");
  std::printf("%s\n", label);
  std::printf("============================================================\n");
  std::printf("GMRES its / relres : %d / %.16e\n", its_p, rel_p);
  print_stats_raw0("xhp full-box", xp_full, (size_t)nx * (size_t)ny * (size_t)nz);
  print_stats_vec1("phi_vec active", phi_vec);
  HYPRE_CALL(HYPRE_StructGMRESDestroy(solver_p)); HYPRE_CALL(HYPRE_StructPFMGDestroy(pc_p));
  HYPRE_CALL(HYPRE_StructVectorDestroy(bhp)); HYPRE_CALL(HYPRE_StructVectorDestroy(xhp)); HYPRE_CALL(HYPRE_StructMatrixDestroy(Ahp));
}



static std::string vtk_filename_with_step(const std::string& base, int step)
{
  std::string stem = base;
  std::string ext;
  std::size_t pos = base.find_last_of('.');
  if (pos != std::string::npos) {
    stem = base.substr(0, pos);
    ext = base.substr(pos);
  }
  char buf[64];
  std::snprintf(buf, sizeof(buf), "_%06d", step);
  return stem + buf + ext;
}

