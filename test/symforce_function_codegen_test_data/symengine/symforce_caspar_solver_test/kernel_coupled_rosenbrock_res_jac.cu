#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

#include "kernel_coupled_rosenbrock_res_jac.h"
#include "memops.cuh"

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1) CoupledRosenbrockResJacKernel(
    double* x, unsigned int x_num_alloc, SharedIndex* x_indices, double* y,
    unsigned int y_num_alloc, SharedIndex* y_indices, double* out_res,
    unsigned int out_res_num_alloc, double* out_x_jac, unsigned int out_x_jac_num_alloc,
    double* const out_x_njtr, unsigned int out_x_njtr_num_alloc, double* const out_x_precond_diag,
    unsigned int out_x_precond_diag_num_alloc, double* const out_x_precond_tril,
    unsigned int out_x_precond_tril_num_alloc, double* out_y_jac, unsigned int out_y_jac_num_alloc,
    double* const out_y_njtr, unsigned int out_y_njtr_num_alloc, double* const out_y_precond_diag,
    unsigned int out_y_precond_diag_num_alloc, double* const out_y_precond_tril,
    unsigned int out_y_precond_tril_num_alloc, size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex x_indices_loc[1024];
  x_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size ? x_indices[global_thread_idx]
                                        : SharedIndex{0xffffffff, 0xffff, 0xffff});
  __shared__ SharedIndex y_indices_loc[1024];
  y_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size ? y_indices[global_thread_idx]
                                        : SharedIndex{0xffffffff, 0xffff, 0xffff});

  double r0 = 0, r1 = 0, r2 = 0, r3 = 0, r4 = 0, r5 = 0, r6 = 0, r7 = 0;

  if (global_thread_idx < problem_size) {
    r0 = 1.00000000000000000e+01;
  };
  LoadShared<2, double, double>(y, 0 * y_num_alloc, y_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double*)inout_shared, y_indices_loc[threadIdx.x].target, r1, r2);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r1 = -1.00000000000000000e+00;
  };
  LoadShared<2, double, double>(x, 0 * x_num_alloc, x_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double*)inout_shared, x_indices_loc[threadIdx.x].target, r3, r4);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r4 = r3 * r3;
    r2 = fma(r1, r4, r2);
    r0 = r0 * r2;
    r5 = 1.00000000000000000e+00;
    r1 = fma(r3, r1, r5);
    WriteIdx2<1024, double, double, double2>(out_res, 0 * out_res_num_alloc, global_thread_idx, r0,
                                             r1);
    r0 = -2.00000000000000000e+01;
    r0 = r3 * r0;
    WriteIdx1<1024, double, double, double>(out_x_jac, 0 * out_x_jac_num_alloc, global_thread_idx,
                                            r0);
    r0 = 0.00000000000000000e+00;
    r6 = r3 * r2;
    r7 = 2.00000000000000000e+02;
    r6 = fma(r7, r6, r1);
    WriteSum2<double, double>((double*)inout_shared, r6, r0);
  };
  FlushSumShared<2, double>(out_x_njtr, 0 * out_x_njtr_num_alloc, x_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r6 = 4.00000000000000000e+02;
    r4 = fma(r6, r4, r5);
    WriteSum2<double, double>((double*)inout_shared, r4, r0);
  };
  FlushSumShared<2, double>(out_x_precond_diag, 0 * out_x_precond_diag_num_alloc, x_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r4 = -1.00000000000000000e+02;
    r4 = r2 * r4;
    WriteSum2<double, double>((double*)inout_shared, r0, r4);
  };
  FlushSumShared<2, double>(out_y_njtr, 0 * out_y_njtr_num_alloc, y_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r4 = 1.00000000000000000e+02;
    WriteSum2<double, double>((double*)inout_shared, r0, r4);
  };
  FlushSumShared<2, double>(out_y_precond_diag, 0 * out_y_precond_diag_num_alloc, y_indices_loc,
                            (double*)inout_shared);
}

void CoupledRosenbrockResJac(
    double* x, unsigned int x_num_alloc, SharedIndex* x_indices, double* y,
    unsigned int y_num_alloc, SharedIndex* y_indices, double* out_res,
    unsigned int out_res_num_alloc, double* out_x_jac, unsigned int out_x_jac_num_alloc,
    double* const out_x_njtr, unsigned int out_x_njtr_num_alloc, double* const out_x_precond_diag,
    unsigned int out_x_precond_diag_num_alloc, double* const out_x_precond_tril,
    unsigned int out_x_precond_tril_num_alloc, double* out_y_jac, unsigned int out_y_jac_num_alloc,
    double* const out_y_njtr, unsigned int out_y_njtr_num_alloc, double* const out_y_precond_diag,
    unsigned int out_y_precond_diag_num_alloc, double* const out_y_precond_tril,
    unsigned int out_y_precond_tril_num_alloc, size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  CoupledRosenbrockResJacKernel<<<n_blocks, 1024>>>(
      x, x_num_alloc, x_indices, y, y_num_alloc, y_indices, out_res, out_res_num_alloc, out_x_jac,
      out_x_jac_num_alloc, out_x_njtr, out_x_njtr_num_alloc, out_x_precond_diag,
      out_x_precond_diag_num_alloc, out_x_precond_tril, out_x_precond_tril_num_alloc, out_y_jac,
      out_y_jac_num_alloc, out_y_njtr, out_y_njtr_num_alloc, out_y_precond_diag,
      out_y_precond_diag_num_alloc, out_y_precond_tril, out_y_precond_tril_num_alloc, problem_size);
}

}  // namespace caspar