#pragma once

#include <cuda_runtime.h>

#include "shared_indices.h"

namespace caspar {

void CoupledRosenbrockResJac(
    double* x, unsigned int x_num_alloc, SharedIndex* x_indices, double* y,
    unsigned int y_num_alloc, SharedIndex* y_indices, double* out_res,
    unsigned int out_res_num_alloc, double* out_x_jac, unsigned int out_x_jac_num_alloc,
    double* const out_x_njtr, unsigned int out_x_njtr_num_alloc, double* const out_x_precond_diag,
    unsigned int out_x_precond_diag_num_alloc, double* const out_x_precond_tril,
    unsigned int out_x_precond_tril_num_alloc, double* out_y_jac, unsigned int out_y_jac_num_alloc,
    double* const out_y_njtr, unsigned int out_y_njtr_num_alloc, double* const out_y_precond_diag,
    unsigned int out_y_precond_diag_num_alloc, double* const out_y_precond_tril,
    unsigned int out_y_precond_tril_num_alloc, size_t problem_size);

}  // namespace caspar