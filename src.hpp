#pragma once
#include "simulator.hpp"
namespace sjtu {

void Calculate(std::vector<Matrix *> keys, std::vector<Matrix *> values,
               Rater &rater, GpuSimulator &gpu_sim,
               MatrixMemoryAllocator matrix_memory_allocator) {
  assert(keys.size() == values.size());
  for (size_t i = 0; i < keys.size(); ++i) {
    auto current_query = rater.GetNextQuery();

    // Build K_acc in HBM by concatenating rows, then transpose and move once
    Matrix *k_acc = matrix_memory_allocator.Allocate("k_acc_init");
    gpu_sim.Copy(keys[0], k_acc, kInGpuHbm);
    for (size_t j = 1; j <= i; ++j) {
      Matrix *k_new = matrix_memory_allocator.Allocate("k_acc_step");
      gpu_sim.Concat(k_acc, keys[j], k_new, /*axis=*/0, kInGpuHbm);
      gpu_sim.ReleaseMatrix(k_acc);
      k_acc = k_new;
    }
    gpu_sim.Transpose(k_acc, kInGpuHbm); // K^T (d, i+1)

    // Build V_acc in HBM by concatenating rows
    Matrix *v_acc = matrix_memory_allocator.Allocate("v_acc_init");
    gpu_sim.Copy(values[0], v_acc, kInGpuHbm);
    for (size_t j = 1; j <= i; ++j) {
      Matrix *v_new = matrix_memory_allocator.Allocate("v_acc_step");
      gpu_sim.Concat(v_acc, values[j], v_new, /*axis=*/0, kInGpuHbm);
      gpu_sim.ReleaseMatrix(v_acc);
      v_acc = v_new;
    }

    // Move operands to SRAM once
    gpu_sim.MoveMatrixToSharedMem(current_query);
    gpu_sim.MoveMatrixToSharedMem(k_acc);
    gpu_sim.MoveMatrixToSharedMem(v_acc);

    // scores = Q (i+1 x d) * K^T (d x i+1)
    Matrix *scores = matrix_memory_allocator.Allocate("scores");
    gpu_sim.MatMul(current_query, k_acc, scores);

    // For each row r: softmax(row) * V_acc => out_row (1, d)
    Matrix *ans_acc = nullptr;
    for (size_t r = 0; r <= i; ++r) {
      Matrix *row_scores = matrix_memory_allocator.Allocate("row_scores");
      gpu_sim.GetRow(scores, r, row_scores, kInSharedMemory);

      Matrix *exp_row = matrix_memory_allocator.Allocate("exp_row");
      gpu_sim.MatExp(row_scores, exp_row);

      Matrix *sum_scalar = matrix_memory_allocator.Allocate("sum_scalar");
      gpu_sim.Sum(exp_row, sum_scalar);

      Matrix *softmax_row = matrix_memory_allocator.Allocate("softmax_row");
      gpu_sim.MatDiv(exp_row, sum_scalar, softmax_row);

      Matrix *out_row = matrix_memory_allocator.Allocate("out_row");
      gpu_sim.MatMul(softmax_row, v_acc, out_row);

      if (r == 0) {
        ans_acc = matrix_memory_allocator.Allocate("ans_acc_init");
        gpu_sim.Copy(out_row, ans_acc, kInSharedMemory);
      } else {
        Matrix *ans_new = matrix_memory_allocator.Allocate("ans_acc_step");
        gpu_sim.Concat(ans_acc, out_row, ans_new, /*axis=*/0, kInSharedMemory);
        gpu_sim.ReleaseMatrix(ans_acc);
        ans_acc = ans_new;
      }

      gpu_sim.ReleaseMatrix(row_scores);
      gpu_sim.ReleaseMatrix(exp_row);
      gpu_sim.ReleaseMatrix(sum_scalar);
      gpu_sim.ReleaseMatrix(softmax_row);
      gpu_sim.ReleaseMatrix(out_row);
    }

    // Move final answer to HBM and cleanup
    gpu_sim.MoveMatrixToGpuHbm(ans_acc);
    gpu_sim.ReleaseMatrix(scores);
    gpu_sim.ReleaseMatrix(k_acc);
    gpu_sim.ReleaseMatrix(v_acc);

    gpu_sim.Run(false, &matrix_memory_allocator);
    rater.CommitAnswer(*ans_acc);
  }
}

void Test(Rater &rater, GpuSimulator &gpu_sim,
          MatrixMemoryAllocator &matrix_memory_allocator) {
  Calculate(rater.keys_, rater.values_, rater, gpu_sim,
            matrix_memory_allocator);
  rater.PrintResult(gpu_sim);
}

} // namespace sjtu
