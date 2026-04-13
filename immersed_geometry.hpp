static void compute_pressure_area_fractions_2d(std::vector<double> &beta2d, const Params &par)
{
  int nx = par.nx, ny = par.ny, nsub = par.nsub_beta;
  double dx = par.Lx / nx, dy = par.Ly / ny;
  std::vector<double> subx((size_t)nsub), suby((size_t)nsub);
  for (int k = 0; k < nsub; ++k) {
    subx[(size_t)k] = ((double)k + 0.5) / (double)nsub;
    suby[(size_t)k] = ((double)k + 0.5) / (double)nsub;
  }
  beta2d.assign((size_t)nx * (size_t)ny, 0.0);
  for (int i = 1; i <= nx; ++i) {
    double x0 = (i - 1) * dx;
    for (int j = 1; j <= ny; ++j) {
      double y0 = (j - 1) * dy;
      int cnt = 0;
      for (int ii = 0; ii < nsub; ++ii) {
        for (int jj = 0; jj < nsub; ++jj) {
          double xs = x0 + subx[(size_t)ii] * dx;
          double ys = y0 + suby[(size_t)jj] * dy;
          double rsq = (xs - par.xcirc) * (xs - par.xcirc) + (ys - par.ycirc) * (ys - par.ycirc);
          if (rsq > par.Rcirc * par.Rcirc) cnt++;
        }
      }
      beta2d[(size_t)id2_flat(i, j, nx) - 1] = (double)cnt / (double)(nsub * nsub);
    }
  }
}

static void compute_vertical_face_open_fractions_2d(std::vector<double> &alpha_u2d, const Params &par)
{
  int nx = par.nx, ny = par.ny;
  double dx = par.Lx / nx, dy = par.Ly / ny;
  alpha_u2d.assign((size_t)(nx + 1) * (size_t)ny, 1.0);
  for (int i = 1; i <= nx + 1; ++i) {
    double x = (i - 1) * dx;
    for (int j = 1; j <= ny; ++j) {
      double y0 = (j - 1) * dy;
      double y1 = j * dy;
      double val = 1.0;
      if (std::abs(x - par.xcirc) < par.Rcirc) {
        double h = std::sqrt(std::max(0.0, par.Rcirc * par.Rcirc - (x - par.xcirc) * (x - par.xcirc)));
        double ys0 = par.ycirc - h;
        double ys1 = par.ycirc + h;
        double overlap = std::max(0.0, std::min(y1, ys1) - std::max(y0, ys0));
        val = std::max(0.0, std::min(1.0, 1.0 - overlap / dy));
      }
      alpha_u2d[(size_t)(i - 1) + (size_t)(j - 1) * (size_t)(nx + 1)] = val;
    }
  }
}

static void compute_horizontal_face_open_fractions_2d(std::vector<double> &alpha_v2d, const Params &par)
{
  int nx = par.nx, ny = par.ny;
  double dx = par.Lx / nx, dy = par.Ly / ny;
  alpha_v2d.assign((size_t)nx * (size_t)(ny + 1), 1.0);
  for (int j = 1; j <= ny + 1; ++j) {
    double y = (j - 1) * dy;
    for (int i = 1; i <= nx; ++i) {
      double x0 = (i - 1) * dx;
      double x1 = i * dx;
      double val = 1.0;
      if (std::abs(y - par.ycirc) < par.Rcirc) {
        double h = std::sqrt(std::max(0.0, par.Rcirc * par.Rcirc - (y - par.ycirc) * (y - par.ycirc)));
        double xs0 = par.xcirc - h;
        double xs1 = par.xcirc + h;
        double overlap = std::max(0.0, std::min(x1, xs1) - std::max(x0, xs0));
        val = std::max(0.0, std::min(1.0, 1.0 - overlap / dx));
      }
      alpha_v2d[(size_t)(i - 1) + (size_t)(j - 1) * (size_t)nx] = val;
    }
  }
}

static void build_2d_masks_from_fluid(
    std::vector<char> &active_u2d,
    std::vector<char> &active_v2d,
    std::vector<char> &active_qu2d,
    std::vector<char> &active_qv2d,
    const std::vector<char> &fluid2d,
    const std::vector<double> &alpha_u2d,
    const std::vector<double> &alpha_v2d,
    const Params &par)
{
  int nx = par.nx, ny = par.ny;
  double dx = par.Lx / nx, dy = par.Ly / ny;
  active_u2d.assign((size_t)(nx + 1) * (size_t)ny, 0);
  active_v2d.assign((size_t)nx * (size_t)(ny + 1), 0);
  active_qu2d.assign((size_t)(nx + 1) * (size_t)ny, 0);
  active_qv2d.assign((size_t)nx * (size_t)(ny + 1), 0);

  for (int i = 2; i <= nx; ++i) {
    double x = (i - 1) * dx;
    for (int j = 1; j <= ny; ++j) {
      double y = (j - 0.5) * dy;
      bool face_outside = ((x - par.xcirc) * (x - par.xcirc) + (y - par.ycirc) * (y - par.ycirc)) > par.Rcirc * par.Rcirc;
      bool adjacent_fluid = fluid2_at(fluid2d, i - 1, j, nx) || fluid2_at(fluid2d, i, j, nx);
      active_u2d[(size_t)(i - 1) + (size_t)(j - 1) * (size_t)(nx + 1)] = (char)(face_outside && adjacent_fluid);
      active_qu2d[(size_t)(i - 1) + (size_t)(j - 1) * (size_t)(nx + 1)] = (char)(face_outside && adjacent_fluid && (alpha_u2_at(alpha_u2d,i,j,nx) > par.alpha_tol));
    }
  }
  for (int j = 1; j <= ny; ++j)
    active_qu2d[(size_t)nx + (size_t)(j - 1) * (size_t)(nx + 1)] = (char)fluid2_at(fluid2d, nx, j, nx);

  for (int i = 1; i <= nx; ++i) {
    double x = (i - 0.5) * dx;
    for (int j = 2; j <= ny; ++j) {
      double y = (j - 1) * dy;
      bool face_outside = ((x - par.xcirc) * (x - par.xcirc) + (y - par.ycirc) * (y - par.ycirc)) > par.Rcirc * par.Rcirc;
      bool adjacent_fluid = fluid2_at(fluid2d, i, j - 1, nx) || fluid2_at(fluid2d, i, j, nx);
      active_v2d[(size_t)(i - 1) + (size_t)(j - 1) * (size_t)nx] = (char)(face_outside && adjacent_fluid);
      active_qv2d[(size_t)(i - 1) + (size_t)(j - 1) * (size_t)nx] = (char)(face_outside && adjacent_fluid && (alpha_v2_at(alpha_v2d,i,j,nx) > par.alpha_tol));
    }
  }
}

static int build_and_prune_2d_masks(
    std::vector<char> &fluid2d,
    std::vector<char> &active_u2d,
    std::vector<char> &active_v2d,
    std::vector<char> &active_qu2d,
    std::vector<char> &active_qv2d,
    const std::vector<double> &beta2d,
    const std::vector<double> &alpha_u2d,
    const std::vector<double> &alpha_v2d,
    const Params &par,
    std::vector<std::pair<int,int>> &pruned_cols)
{
  int nx = par.nx, ny = par.ny;
  fluid2d.assign((size_t)nx * (size_t)ny, 0);
  for (int i = 1; i <= nx; ++i)
    for (int j = 1; j <= ny; ++j)
      fluid2d[(size_t)id2_flat(i,j,nx)-1] = (char)(beta2d[(size_t)id2_flat(i,j,nx)-1] > par.beta_tol);

  int n_pruned = 0;
  for (;;) {
    build_2d_masks_from_fluid(active_u2d, active_v2d, active_qu2d, active_qv2d, fluid2d, alpha_u2d, alpha_v2d, par);
    std::vector<char> remove((size_t)nx * (size_t)ny, 0);
    bool any = false;
    for (int i = 1; i <= nx; ++i) {
      for (int j = 1; j <= ny; ++j) {
        if (!fluid2_at(fluid2d, i, j, nx)) continue;
        int nconn = 0;
        if (i < nx) { if (active_qu2d[(size_t)i + (size_t)(j - 1) * (size_t)(nx + 1)]) nconn++; }
        else        { if (active_qu2d[(size_t)nx + (size_t)(j - 1) * (size_t)(nx + 1)]) nconn++; }
        if (i > 1)  { if (active_qu2d[(size_t)(i - 1) + (size_t)(j - 1) * (size_t)(nx + 1)]) nconn++; }
        if (j < ny) { if (active_qv2d[(size_t)(i - 1) + (size_t)j * (size_t)nx]) nconn++; }
        if (j > 1)  { if (active_qv2d[(size_t)(i - 1) + (size_t)(j - 1) * (size_t)nx]) nconn++; }
        if (nconn == 0) { remove[(size_t)id2_flat(i,j,nx)-1] = 1; any = true; }
      }
    }
    if (!any) break;
    for (int i = 1; i <= nx; ++i) for (int j = 1; j <= ny; ++j) {
      size_t idx = (size_t)id2_flat(i,j,nx)-1;
      if (remove[idx]) { fluid2d[idx] = 0; n_pruned++; pruned_cols.emplace_back(i,j); }
    }
  }
  build_2d_masks_from_fluid(active_u2d, active_v2d, active_qu2d, active_qv2d, fluid2d, alpha_u2d, alpha_v2d, par);
  return n_pruned;
}

static void build_indices(const std::vector<char> &mask, std::vector<int> &id)
{
  id.assign(mask.size(), 0);
  int c = 0;
  for (size_t k = 0; k < mask.size(); ++k) if (mask[k]) id[k] = ++c;
}

