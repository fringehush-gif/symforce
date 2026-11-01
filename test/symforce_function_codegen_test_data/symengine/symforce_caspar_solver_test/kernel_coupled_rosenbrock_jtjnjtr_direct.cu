#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

#include "kernel_coupled_rosenbrock_jtjnjtr_direct.h"
#include "memops.cuh"

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1) CoupledRosenbrockJtjnjtrDirectKernel(
    double* x_njtr, unsigned int x_njtr_num_alloc, SharedIndex* x_njtr_indices, double* x_jac,
    unsigned int x_jac_num_alloc, double* y_njtr, unsigned int y_njtr_num_alloc,
    SharedIndex* y_njtr_indices, double* y_jac, unsigned int y_jac_num_alloc,
    double* const out_x_njtr, unsigned int out_x_njtr_num_alloc, double* const out_y_njtr,
    unsigned int out_y_njtr_num_alloc, size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex x_njtr_indices_loc[1024];
  x_njtr_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size ? x_njtr_indices[global_thread_idx]
                                        : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ SharedIndex y_njtr_indices_loc[1024];
  y_njtr_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size ? y_njtr_indices[global_thread_idx]
                                        : SharedIndex{0xffffffff, 0xffff, 0xffff});

  double r0 = 0, r1 = 0, r2 = 0, r3 = 0;

  if (global_thread_idx < problem_size) {
    r0 = 0.00000000000000000e+00;
  };
  LoadShared<2, double, double>(y_njtr, 0 * y_njtr_num_alloc, y_njtr_indices_loc,
                                (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double*)inout_shared, y_njtr_indices_loc[threadIdx.x].target, r1, r2);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx1<1024, double, double, double>(x_jac, 0 * x_jac_num_alloc, global_thread_idx, r1);
    r3 = 1.00000000000000000e+01;
    r3 = r1 * r3;
    r2 = r2 * r3;
    WriteSum2<double, double>((double*)inout_shared, r2, r0);
  };
  FlushSumShared<2, double>(out_x_njtr, 0 * out_x_njtr_num_alloc, x_njtr_indices_loc,
                            (double*)inout_shared);
  LoadShared<2, double, double>(x_njtr, 0 * x_njtr_num_alloc, x_njtr_indices_loc,
                                (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double*)inout_shared, x_njtr_indices_loc[threadIdx.x].target, r2, r1);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r3 = r2 * r3;
    WriteSum2<double, double>((double*)inout_shared, r0, r3);
  };
  FlushSumShared<2, double>(out_y_njtr, 0 * out_y_njtr_num_alloc, y_njtr_indices_loc,
                            (double*)inout_shared);
}

void CoupledRosenbrockJtjnjtrDirect(double* x_njtr, unsigned int x_njtr_num_alloc,
                                    SharedIndex* x_njtr_indices, double* x_jac,
                                    unsigned int x_jac_num_alloc, double* y_njtr,
                                    unsigned int y_njtr_num_alloc, SharedIndex* y_njtr_indices,
                                    double* y_jac, unsigned int y_jac_num_alloc,
                                    double* const out_x_njtr, unsigned int out_x_njtr_num_alloc,
                                    double* const out_y_njtr, unsigned int out_y_njtr_num_alloc,
                                    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  CoupledRosenbrockJtjnjtrDirectKernel<<<n_blocks, 1024>>>(
      x_njtr, x_njtr_num_alloc, x_njtr_indices, x_jac, x_jac_num_alloc, y_njtr, y_njtr_num_alloc,
      y_njtr_indices, y_jac, y_jac_num_alloc, out_x_njtr, out_x_njtr_num_alloc, out_y_njtr,
      out_y_njtr_num_alloc, problem_size);
}

}  // namespace caspar