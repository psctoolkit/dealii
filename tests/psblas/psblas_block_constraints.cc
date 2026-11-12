// -----------------------------------------------------------------------------
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception OR LGPL-2.1-or-later
// Copyright (C) 2026 by the deal.II authors
//
// This file is part of the deal.II library.
//
// Detailed license information governing the source code and contributions
// can be found in LICENSE.md and CONTRIBUTING.md at the top level directory.
//
// -----------------------------------------------------------------------------

#include <deal.II/base/index_set.h>
#include <deal.II/base/mpi.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/psblas_block_sparse_matrix.h>
#include <deal.II/lac/psblas_block_vector.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/vector.h>

#include <cmath>
#include <vector>

#include "../tests.h"

using namespace dealii;

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  mpi_initlog();
  const MPI_Comm communicator = MPI_COMM_WORLD;
  const auto     rank         = Utilities::MPI::this_mpi_process(communicator);
  const auto     n_processes  = Utilities::MPI::n_mpi_processes(communicator);
  AssertThrow(n_processes >= 2, ExcInternalError());

  std::vector<IndexSet> owned = {IndexSet(2 * n_processes),
                                 IndexSet(n_processes)};
  owned[0].add_range(2 * rank, 2 * rank + 2);
  owned[1].add_index(rank);
  const types::global_dof_index size = 3 * n_processes;
  IndexSet                      combined_owned(size);
  combined_owned.add_range(2 * rank, 2 * rank + 2);
  combined_owned.add_index(2 * n_processes + rank);

  // The constrained entry is owned by rank zero; its two masters belong to
  // the last rank and to different blocks.
  AffineConstraints<double> constraints(combined_owned,
                                        complete_index_set(size));
  constraints.add_line(1);
  constraints.add_entry(1, 2 * n_processes - 1, 0.5);
  constraints.add_entry(1, 3 * n_processes - 1, 2.0);
  constraints.set_inhomogeneity(1, 0.75);
  constraints.close();

  PSCToolkitWrappers::BlockSparseMatrix matrix;
  PSCToolkitWrappers::BlockSparseMatrix matrix_only;
  matrix.reinit(owned, communicator);
  matrix_only.reinit(owned, communicator);
  PSCToolkitWrappers::BlockVector rhs(owned, communicator);

  FullMatrix<double> local_matrix(3, 3);
  local_matrix(0, 0) = 4.0;
  local_matrix(1, 1) = 3.0;
  local_matrix(2, 2) = 2.0;
  local_matrix(0, 1) = local_matrix(1, 0) = -1.0;
  local_matrix(0, 2) = local_matrix(2, 0) = 0.5;
  local_matrix(1, 2) = local_matrix(2, 1) = -0.25;
  Vector<double> local_rhs(3);
  local_rhs(0) = 1.0;
  local_rhs(1) = 2.0;
  local_rhs(2) = 3.0;

  const std::vector<types::global_dof_index> local_indices = {
    2 * rank, 2 * rank + 1, 2 * n_processes + rank};
  constraints.distribute_local_to_global(
    local_matrix, local_rhs, local_indices, matrix, rhs);
  constraints.distribute_local_to_global(local_matrix,
                                         local_indices,
                                         matrix_only);
  matrix.compress();
  matrix_only.compress();
  rhs.compress(VectorOperation::add);

  DynamicSparsityPattern dynamic_pattern(size, size);
  for (types::global_dof_index i = 0; i < size; ++i)
    for (types::global_dof_index j = 0; j < size; ++j)
      dynamic_pattern.add(i, j);
  SparsityPattern pattern;
  pattern.copy_from(dynamic_pattern);
  SparseMatrix<double> reference_matrix(pattern);
  Vector<double>       reference_rhs(size);
  for (unsigned int p = 0; p < n_processes; ++p)
    {
      const std::vector<types::global_dof_index> indices = {
        2 * p, 2 * p + 1, 2 * n_processes + p};
      constraints.distribute_local_to_global(
        local_matrix, local_rhs, indices, reference_matrix, reference_rhs);
    }

  PSCToolkitWrappers::BlockVector source(owned, communicator);
  PSCToolkitWrappers::BlockVector result(owned, communicator);
  Vector<double>                  reference_source(size);
  Vector<double>                  reference_result(size);
  for (types::global_dof_index i = 0; i < size; ++i)
    reference_source(i) = 1.0 + i;
  for (unsigned int b = 0; b < 2; ++b)
    for (const auto i : owned[b])
      {
        const auto global_i = i + (b == 0 ? 0 : 2 * n_processes);
        source.block(b)(i)  = reference_source(global_i);
        AssertThrow(std::abs(rhs.block(b)(i) - reference_rhs(global_i)) <
                      1.e-12,
                    ExcInternalError());
      }
  source.compress(VectorOperation::insert);
  reference_matrix.vmult(reference_result, reference_source);

  const auto check_result = [&]() {
    for (unsigned int b = 0; b < 2; ++b)
      for (const auto i : owned[b])
        {
          const auto global_i = i + (b == 0 ? 0 : 2 * n_processes);
          AssertThrow(std::abs(result.block(b)(i) -
                               reference_result(global_i)) < 1.e-11,
                      ExcInternalError());
        }
  };
  matrix.vmult(result, source);
  check_result();
  matrix_only.vmult(result, source);
  check_result();
  if (rank == 0)
    deallog << "Constrained native block assembly OK" << std::endl;

  constraints.distribute(source);
  constraints.distribute(reference_source);
  for (unsigned int b = 0; b < 2; ++b)
    for (const auto i : owned[b])
      {
        const auto global_i = i + (b == 0 ? 0 : 2 * n_processes);
        AssertThrow(std::abs(source.block(b)(i) - reference_source(global_i)) <
                      1.e-12,
                    ExcInternalError());
      }
  if (rank == 0)
    deallog << "Native block constraint distribution OK" << std::endl;
}
