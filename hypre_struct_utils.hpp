static HYPRE_StructGrid CreateGrid3D(int nx, int ny, int nz)
{
  HYPRE_StructGrid grid; int il[3]={1,1,1}; int iu[3]={nx,ny,nz};
  HYPRE_CALL(HYPRE_StructGridCreate(MPI_COMM_WORLD,3,&grid));
  HYPRE_CALL(HYPRE_StructGridSetExtents(grid,il,iu)); HYPRE_CALL(HYPRE_StructGridAssemble(grid)); return grid;
}
static HYPRE_StructStencil CreateStencil3D()
{
  HYPRE_StructStencil st; int offs[7][3]={{0,0,0},{-1,0,0},{1,0,0},{0,-1,0},{0,1,0},{0,0,-1},{0,0,1}};
  HYPRE_CALL(HYPRE_StructStencilCreate(3,7,&st)); for(int s=0;s<7;++s) HYPRE_CALL(HYPRE_StructStencilSetElement(st,s,offs[s])); return st;
}
static void SetMatrix3D(HYPRE_StructMatrix A, int nx, int ny, int nz, HYPRE_Complex *vals)
{
  int il[3]={1,1,1}, iu[3]={nx,ny,nz}, entries[7]={0,1,2,3,4,5,6};
  HYPRE_CALL(HYPRE_StructMatrixInitialize(A));
  HYPRE_CALL(HYPRE_StructMatrixSetBoxValues(A,il,iu,7,entries,vals));
  HYPRE_CALL(HYPRE_StructMatrixAssemble(A));
}
static HYPRE_StructMatrix CreateMatrix3D(HYPRE_StructGrid grid, HYPRE_StructStencil st, int nx,int ny,int nz,HYPRE_Complex* vals)
{ HYPRE_StructMatrix A; HYPRE_CALL(HYPRE_StructMatrixCreate(MPI_COMM_WORLD,grid,st,&A)); SetMatrix3D(A,nx,ny,nz,vals); return A; }
static void SetVector3D(HYPRE_StructVector x, int nx,int ny,int nz, HYPRE_Complex* vals)
{ int il[3]={1,1,1}, iu[3]={nx,ny,nz}; HYPRE_CALL(HYPRE_StructVectorSetBoxValues(x,il,iu,vals)); HYPRE_CALL(HYPRE_StructVectorAssemble(x)); }
static void GetVector3D(HYPRE_StructVector x, int nx,int ny,int nz, HYPRE_Complex* vals)
{ int il[3]={1,1,1}, iu[3]={nx,ny,nz}; HYPRE_CALL(HYPRE_StructVectorGetBoxValues(x,il,iu,vals)); }
static HYPRE_StructVector CreateVector3D(HYPRE_StructGrid grid)
{ HYPRE_StructVector x; HYPRE_CALL(HYPRE_StructVectorCreate(MPI_COMM_WORLD,grid,&x)); HYPRE_CALL(HYPRE_StructVectorInitialize(x)); return x; }
static void MallocManagedC(HYPRE_Complex **p, size_t n) { CUDA_CALL(cudaMallocManaged((void**)p, n*sizeof(HYPRE_Complex), cudaMemAttachGlobal)); }

static void CopyVecToManagedC(HYPRE_Complex **p, const std::vector<HYPRE_Complex> &src)
{
  MallocManagedC(p, src.size());
  for (size_t i = 0; i < src.size(); ++i) (*p)[i] = src[i];
  CUDA_CALL(cudaDeviceSynchronize());
}



static void MallocManagedD(double **p, size_t n) { CUDA_CALL(cudaMallocManaged((void**)p, n*sizeof(double), cudaMemAttachGlobal)); }
static void MallocManagedI(int **p, size_t n) { CUDA_CALL(cudaMallocManaged((void**)p, n*sizeof(int), cudaMemAttachGlobal)); }
static void MallocManagedB(unsigned char **p, size_t n) { CUDA_CALL(cudaMallocManaged((void**)p, n*sizeof(unsigned char), cudaMemAttachGlobal)); }
static void CopyVecToManagedD(double **p, const std::vector<double> &src){ MallocManagedD(p, src.size()); for(size_t i=0;i<src.size();++i)(*p)[i]=src[i]; CUDA_CALL(cudaDeviceSynchronize()); }
static void CopyVecToManagedI(int **p, const std::vector<int> &src){ MallocManagedI(p, src.size()); for(size_t i=0;i<src.size();++i)(*p)[i]=src[i]; CUDA_CALL(cudaDeviceSynchronize()); }
static void CopyVecToManagedB(unsigned char **p, const std::vector<char> &src){ MallocManagedB(p, src.size()); for(size_t i=0;i<src.size();++i)(*p)[i]=(unsigned char)src[i]; CUDA_CALL(cudaDeviceSynchronize()); }

