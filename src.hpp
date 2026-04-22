#pragma once
#include "simulator.hpp"
namespace sjtu {

void Calculate(std::vector<Matrix *> keys, std::vector<Matrix *> values,
               Rater &rater, GpuSimulator &gpu_sim,
               MatrixMemoryAllocator matrix_memory_allocator) {
  assert(keys.size() == values.size());
  for (size_t i = 0; i < keys.size(); ++i) {
    auto current_query = rater.GetNextQuery();

    // Move current query to SRAM for computation
    gpu_sim.MoveMatrixToSharedMem(current_query);

    // Build K^T in SRAM by concatenating transposed copies of keys along columns
    Matrix *kT_acc = nullptr;
    for (size_t j = 0; j <= i; ++j) {
      gpu_sim.MoveMatrixToSharedMem(keys[j]);
      Matrix *k_col = matrix_memory_allocator.Allocate("k_col");
      gpu_sim.Copy(keys[j], k_col, kInSharedMemory);
      gpu_sim.Transpose(k_col, kInSharedMemory); // now (d x 1)
      if (j == 0) {
        kT_acc = k_col;
      } else {
        Matrix *kT_new = matrix_memory_allocator.Allocate("kT_acc");
        gpu_sim.Concat(kT_acc, k_col, kT_new, /*axis=*/1, kInSharedMemory);
        gpu_sim.ReleaseMatrix(kT_acc);
        gpu_sim.ReleaseMatrix(k_col);
        kT_acc = kT_new;
      }
    }

    // scores = Q (i+1 x d) * K^T (d x i+1) => (i+1 x i+1)
    Matrix *scores = matrix_memory_allocator.Allocate("scores");
    gpu_sim.MatMul(current_query, kT_acc, scores);

    // For each row r, compute softmax(row) and out_row = sum_j softmax_j * V_j
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

      // out_row = softmax_row[0] * values[0]
      gpu_sim.MoveMatrixToSharedMem(values[0]);
      Matrix *alpha0 = matrix_memory_allocator.Allocate("alpha0");
      gpu_sim.GetColumn(softmax_row, 0, alpha0, kInSharedMemory); // 1x1
      Matrix *out_row = matrix_memory_allocator.Allocate("out_row");
      gpu_sim.MatMulNum(values[0], alpha0, out_row);
      gpu_sim.ReleaseMatrix(alpha0);

      // accumulate for j=1..i: out_row += softmax_row[j] * values[j]
      for (size_t j = 1; j <= i; ++j) {
        gpu_sim.MoveMatrixToSharedMem(values[j]);
        Matrix *alpha = matrix_memory_allocator.Allocate("alpha");
        gpu_sim.GetColumn(softmax_row, j, alpha, kInSharedMemory); // 1x1
        Matrix *scaled_v = matrix_memory_allocator.Allocate("scaled_v");
        gpu_sim.MatMulNum(values[j], alpha, scaled_v);
        Matrix *new_out = matrix_memory_allocator.Allocate("new_out");
        gpu_sim.MatAdd(out_row, scaled_v, new_out);
        gpu_sim.ReleaseMatrix(out_row);
        gpu_sim.ReleaseMatrix(scaled_v);
        gpu_sim.ReleaseMatrix(alpha);
        out_row = new_out;
      }

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

    // Move final answer to HBM and clean up temporaries
    gpu_sim.MoveMatrixToGpuHbm(ans_acc);
    gpu_sim.ReleaseMatrix(scores);
    gpu_sim.ReleaseMatrix(kT_acc);

    // Execute queued instructions and commit
    gpu_sim.Run(false, &matrix_memory_allocator);
    rater.CommitAnswer(*ans_acc);

    // Move query back to HBM to free SRAM
    gpu_sim.MoveMatrixToGpuHbm(current_query);
  }
}

void Test(Rater &rater, GpuSimulator &gpu_sim,
          MatrixMemoryAllocator &matrix_memory_allocator) {
  Calculate(rater.keys_, rater.values_, rater, gpu_sim,
            matrix_memory_allocator);
  rater.PrintResult(gpu_sim);
}

} // namespace sjtu
