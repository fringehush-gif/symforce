#pragma once

#include <cuda_runtime.h>

#include "shared_indices.h"

namespace caspar {

void CoupledRosenbrockScore(double* x, unsigned int x_num_alloc, SharedIndex* x_indices, double* y,
                            unsigned int y_num_alloc, SharedIndex* y_indices, double* const out_rTr,
                            size_t problem_size);

}  // namespace caspar