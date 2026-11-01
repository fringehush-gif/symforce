/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2025, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cuda_runtime.h>

#include "solver.h"

namespace caspar {

// Only the seeded norm regression needs access beyond the public solver API.
struct GraphSolverTestAccess {
  static void CheckFirstNormReset(GraphSolver& solver, const double seed) {
    const double current_seed = 2 * seed;
    REQUIRE(cudaMemcpy(solver.solver__r_0_norm2_tot_, &seed, sizeof(seed),
                       cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cudaMemcpy(solver.solver__r_kp1_norm2_tot_, &current_seed, sizeof(current_seed),
                       cudaMemcpyHostToDevice) == cudaSuccess);
    solver.DoUpdateRFirst();
    CHECK(solver.pcg_r_0_norm2_ == 0);
    CHECK(solver.pcg_r_kp1_norm2_ == 0);
  }
};

}  // namespace caspar

namespace {

constexpr size_t kNumNodes = 4096;

bool HaveDevice() {
  int count = 0;
  return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

struct Solve {
  caspar::SolveResult result;
  std::vector<double> x;
};

// Independent Rosenbrock problems by default; the coupled mode adds a ring.
Solve Run(const caspar::SolverParams<double>& params, const bool coupled = false) {
  caspar::GraphSolver solver(params, kNumNodes, kNumNodes, kNumNodes);
  solver.SetCoupledRosenbrockNum(coupled ? kNumNodes : 0);

  std::vector<unsigned int> indices(kNumNodes);
  std::vector<double> x(2 * kNumNodes);
  for (size_t i = 0; i < kNumNodes; i++) {
    indices[i] = static_cast<unsigned int>(i);
    x[2 * i + 0] = -1.2;
    x[2 * i + 1] = 1.0;
    if (coupled) {
      // Break the uniform ring symmetry without random fixture state.
      x[2 * i + 0] += 0.1 * (i % 4);
      x[2 * i + 1] -= 0.1 * ((i + 1) % 4);
    }
  }

  solver.SetRosenbrockXIndicesFromHost(indices.data(), kNumNodes);
  if (coupled) {
    std::vector<unsigned int> next_indices(kNumNodes);
    for (size_t i = 0; i < kNumNodes; i++) {
      next_indices[i] = static_cast<unsigned int>((i + 1) % kNumNodes);
    }
    solver.SetCoupledRosenbrockXIndicesFromHost(indices.data(), kNumNodes);
    solver.SetCoupledRosenbrockYIndicesFromHost(next_indices.data(), kNumNodes);
  }
  solver.SetMatrix21NodesFromStackedHost(x.data(), 0, kNumNodes);
  solver.finish_indices();

  Solve solve;
  solve.result = solver.solve(/* print_progress */ false, /* verbose_logging */ true);
  solve.x.resize(2 * kNumNodes);
  solver.GetMatrix21NodesToStackedHost(solve.x.data(), 0, kNumNodes);
  return solve;
}

// Half the sum of squared residuals, which is what the solver reports.
double ScoreOf(const std::vector<double>& x) {
  double score = 0.0;
  for (size_t i = 0; i < kNumNodes; i++) {
    const double r0 = 10.0 * (x[2 * i + 1] - x[2 * i + 0] * x[2 * i + 0]);
    const double r1 = 1.0 - x[2 * i + 0];
    score += r0 * r0 + r1 * r1;
  }
  return 0.5 * score;
}

double CoupledScoreOf(const std::vector<double>& x) {
  std::vector<double> cross(2 * kNumNodes);
  for (size_t i = 0; i < kNumNodes; i++) {
    cross[2 * i] = x[2 * i];
    cross[2 * i + 1] = x[2 * ((i + 1) % kNumNodes) + 1];
  }
  return ScoreOf(x) + ScoreOf(cross);
}

caspar::SolverParams<double> ParamsWithIterations(const int solver_iter_max) {
  caspar::SolverParams<double> params;
  params.solver_iter_max = solver_iter_max;
  return params;
}

}  // namespace

TEST_CASE("Solver drives a bank of Rosenbrock problems toward the optimum",
          "[caspar_solver_cuda_test]") {
  REQUIRE(HaveDevice());

  const Solve solve = Run(ParamsWithIterations(50));

  CHECK(std::isfinite(solve.result.final_score));
  CHECK(solve.result.final_score < solve.result.initial_score);
  CHECK(solve.x[0] > -1.2);
  CHECK(ScoreOf(solve.x) == Catch::Approx(solve.result.final_score).epsilon(1e-9));
}

TEST_CASE("The last iteration makes progress", "[caspar_solver_cuda_test]") {
  REQUIRE(HaveDevice());

  const Solve one = Run(ParamsWithIterations(1));
  const Solve two = Run(ParamsWithIterations(2));

  CHECK(std::isfinite(two.result.final_score));
  CHECK(two.result.final_score < one.result.final_score);
}

TEST_CASE("step_accepted marks exactly the iterations that moved the score",
          "[caspar_solver_cuda_test]") {
  REQUIRE(HaveDevice());

  const Solve solve = Run(ParamsWithIterations(8));
  REQUIRE(solve.result.iterations.size() > 1);

  int accepted = 0;
  double previous_best = solve.result.initial_score;
  for (const caspar::IterationData& iteration : solve.result.iterations) {
    CAPTURE(iteration.solver_iter, iteration.score_best, previous_best);
    CHECK(iteration.step_accepted == (iteration.score_best < previous_best));
    accepted += iteration.step_accepted;
    previous_best = iteration.score_best;
  }
  CHECK(accepted > 0);
}

TEST_CASE("A solve that accepts non-monotonically still returns what it reports",
          "[caspar_solver_cuda_test]") {
  REQUIRE(HaveDevice());

  caspar::SolverParams<double> params = ParamsWithIterations(8);
  params.pcg_rel_decrease_min = 0.5;
  params.solver_rel_decrease_min = 1.5;

  const Solve solve = Run(params);

  CHECK(std::isfinite(solve.result.final_score));
  CHECK(ScoreOf(solve.x) == Catch::Approx(solve.result.final_score).epsilon(1e-9));
}

TEST_CASE("A solver with no inner iterations is rejected", "[caspar_solver_cuda_test]") {
  REQUIRE(HaveDevice());

  caspar::SolverParams<double> params = ParamsWithIterations(8);
  params.pcg_iter_max = 0;

  CHECK_THROWS(caspar::GraphSolver(params, kNumNodes, kNumNodes, kNumNodes));
}

TEST_CASE("Invalid post-construction solver parameters are rejected", "[caspar_solver_cuda_test]") {
  REQUIRE(HaveDevice());
  caspar::SolverParams<double> params = ParamsWithIterations(8);
  caspar::GraphSolver solver(params, kNumNodes, kNumNodes, kNumNodes);
  solver.SetCoupledRosenbrockNum(0);
  for (const int invalid : {0, -1}) {
    params.pcg_iter_max = invalid;
    CHECK_THROWS_AS(solver.set_params(params), std::runtime_error);
  }
  params = ParamsWithIterations(8);
  params.pcg_rel_error_exit = 0;
  CHECK_THROWS_AS(solver.set_params(params), std::runtime_error);
  params = ParamsWithIterations(8);
  params.pcg_rel_error_exit = -1;
  CHECK_THROWS_AS(solver.set_params(params), std::runtime_error);
  params = ParamsWithIterations(8);
  params.diag_init = -1;
  CHECK_THROWS_AS(solver.set_params(params), std::runtime_error);
  params = ParamsWithIterations(8);
  params.pcg_iter_max = 1;
  params.diag_init = 0;  // The existing constructor accepts zero.
  CHECK_NOTHROW(solver.set_params(params));
}

TEST_CASE("Scored PCG criteria operate independently", "[caspar_solver_cuda_test]") {
  if (!HaveDevice()) {
    SKIP("CUDA device unavailable");
  }

  auto reference_params = ParamsWithIterations(1);
  reference_params.pcg_iter_max = 1;
  const Solve reference = Run(reference_params);
  REQUIRE(reference.result.iterations.size() == 1);
  const auto& expected = reference.result.iterations.front();
  REQUIRE(expected.step_accepted);
  REQUIRE(std::isfinite(expected.step_quality));

  for (const bool decrease_enabled : {false, true}) {
    for (const bool score_exit_enabled : {false, true}) {
      CAPTURE(decrease_enabled, score_exit_enabled);
      auto params = reference_params;
      params.pcg_rel_decrease_min = decrease_enabled ? 1.0 : -1.0;
      params.pcg_rel_score_exit = score_exit_enabled ? 0.5 : -1.0;
      const Solve solve = Run(params);
      REQUIRE(solve.result.iterations.size() == 1);
      const auto& iteration = solve.result.iterations.front();
      CHECK(iteration.step_accepted);
      CHECK(iteration.pcg_iter == 0);
      CHECK(iteration.score_best < solve.result.initial_score);
      CHECK(iteration.score_best == Catch::Approx(expected.score_best).epsilon(1e-9));
      CHECK(iteration.step_quality == Catch::Approx(expected.step_quality).epsilon(1e-9));
      CHECK(ScoreOf(solve.x) == Catch::Approx(solve.result.final_score).epsilon(1e-9));
    }
  }
}

TEST_CASE("Retained PCG predictions match a truncated solve", "[caspar_solver_cuda_test]") {
  if (!HaveDevice()) {
    SKIP("CUDA device unavailable");
  }

  auto params = ParamsWithIterations(2);
  params.diag_init = 1;
  params.diag_scaling_down = 0;
  params.pcg_rel_decrease_min = 0.5;
  params.pcg_rel_score_exit = -1;
  params.pcg_rel_error_exit = 1e-12;
  params.pcg_iter_max = 1;
  const Solve truncated = Run(params, /* coupled */ true);
  params.pcg_iter_max = 2;
  const Solve retained = Run(params, /* coupled */ true);
  REQUIRE(truncated.result.iterations.size() == 2);
  REQUIRE(retained.result.iterations.size() == 2);
  const auto& a = truncated.result.iterations[0];
  const auto& b = retained.result.iterations[0];
  REQUIRE(a.step_accepted);
  REQUIRE(b.step_accepted);
  REQUIRE(a.pcg_iter == 0);
  REQUIRE(b.pcg_iter == 1);
  REQUIRE(a.score_best > 0);
  // The second proposal must halve the retained score to replace it. Matching
  // the one-step score therefore identifies a retained first proposal.
  REQUIRE(b.score_best == Catch::Approx(a.score_best).epsilon(1e-9));
  REQUIRE(std::isfinite(a.step_quality));
  REQUIRE(std::isfinite(b.step_quality));
  CHECK(b.step_quality == Catch::Approx(a.step_quality).epsilon(1e-9));
  CHECK(retained.result.iterations[1].diag ==
        Catch::Approx(truncated.result.iterations[1].diag).epsilon(1e-9).margin(1e-12));
  CHECK(CoupledScoreOf(truncated.x) == Catch::Approx(truncated.result.final_score).epsilon(1e-9));
  CHECK(CoupledScoreOf(retained.x) == Catch::Approx(retained.result.final_score).epsilon(1e-9));
}

TEST_CASE("First residual updates reset both seeded norm accumulators",
          "[caspar_solver_cuda_test]") {
  if (!HaveDevice()) {
    SKIP("CUDA device unavailable");
  }

  // Allocate valid storage, then disable all nodes. No residual contribution
  // can hide a stale scalar, and no uninitialized kernel input is consumed.
  caspar::GraphSolver solver(ParamsWithIterations(1), 1, 1, 1);
  solver.SetMatrix21Num(0);
  solver.SetRosenbrockNum(0);
  solver.SetCoupledRosenbrockNum(0);
  for (const double seed : {3.0, 17.0, 4096.0}) {
    CAPTURE(seed);
    caspar::GraphSolverTestAccess::CheckFirstNormReset(solver, seed);
  }
}
