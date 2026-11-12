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

#include <deal.II/lac/psblas_block_sparse_matrix.h>
#include <deal.II/lac/psblas_block_vector.h>

#include "../tests.h"

// A 2-by-3 block operator with rectangular blocks, different halo maps for
// the same source block, and ranks owning no entries in some blocks.
namespace
{
  double
  entry(const unsigned int r,
        const unsigned int c,
        const unsigned int i,
        const unsigned int j,
        const unsigned int n_columns)
  {
    if (j != (i + 2 * r + c) % n_columns &&
        j != (2 * i + r + c + 1) % n_columns)
      return 0.;
    return 0.5 * (1 + r + c) + 0.125 * i - 0.25 * j;
  }
} // namespace

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  mpi_initlog();
  const auto         comm = MPI_COMM_WORLD;
  const unsigned int rank = Utilities::MPI::this_mpi_process(comm);
  const unsigned int np   = Utilities::MPI::n_mpi_processes(comm);
  AssertThrow(np >= 2, ExcInternalError());
  std::vector<IndexSet> rows{IndexSet(2 * np + 1), IndexSet(np - 1)};
  std::vector<IndexSet> columns{IndexSet(np + 2),
                                IndexSet(2 * np),
                                IndexSet(1)};
  for (auto &partition : rows)
    for (unsigned int i = 0; i < partition.size(); ++i)
      if (i % np == rank)
        partition.add_index(i);
  for (auto &partition : columns)
    for (unsigned int i = 0; i < partition.size(); ++i)
      if (i % np == rank)
        partition.add_index(i);

  PSCToolkitWrappers::BlockSparseMatrix matrix;
  matrix.reinit(rows, columns, comm);
  unsigned int row_offset = 0;
  for (unsigned int r = 0; r < rows.size(); ++r)
    {
      unsigned int column_offset = 0;
      for (unsigned int c = 0; c < columns.size(); ++c)
        {
          for (unsigned int i = 0; i < rows[r].size(); ++i)
            if ((i + 1) % np == rank)
              for (unsigned int j = 0; j < columns[c].size(); ++j)
                {
                  const double value = entry(r, c, i, j, columns[c].size());
                  if (value != 0.)
                    matrix.add(row_offset + i, column_offset + j, value);
                }
          column_offset += columns[c].size();
        }
      row_offset += rows[r].size();
    }
  matrix.compress();
  PSCToolkitWrappers::BlockVector x(columns, comm), y(rows, comm);
  PSCToolkitWrappers::BlockVector u(rows, comm), z(columns, comm);
  for (unsigned int c = 0; c < columns.size(); ++c)
    for (const auto j : columns[c])
      x.block(c)(j) = 1. + c + 0.5 * j;
  for (unsigned int r = 0; r < rows.size(); ++r)
    for (const auto i : rows[r])
      u.block(r)(i) = 2. + r - 0.25 * i;
  x.compress(VectorOperation::insert);
  u.compress(VectorOperation::insert);
  const auto check_rows = [&](const double scale) {
    for (unsigned int r = 0; r < rows.size(); ++r)
      for (const auto i : rows[r])
        {
          double expected = 0.;
          for (unsigned int c = 0; c < columns.size(); ++c)
            for (unsigned int j = 0; j < columns[c].size(); ++j)
              expected +=
                entry(r, c, i, j, columns[c].size()) * (1. + c + 0.5 * j);
          AssertThrow(std::abs(y.block(r)(i) - scale * expected) < 1.e-10,
                      ExcInternalError());
        }
  };
  const auto check_columns = [&](const double scale) {
    for (unsigned int c = 0; c < columns.size(); ++c)
      for (const auto j : columns[c])
        {
          double expected = 0.;
          for (unsigned int r = 0; r < rows.size(); ++r)
            for (unsigned int i = 0; i < rows[r].size(); ++i)
              expected +=
                entry(r, c, i, j, columns[c].size()) * (2. + r - 0.25 * i);
          AssertThrow(std::abs(z.block(c)(j) - scale * expected) < 1.e-10,
                      ExcInternalError());
        }
  };
  // Reuse each native block across several different descriptor halo maps.
  for (unsigned int repeat = 0; repeat < 3; ++repeat)
    {
      matrix.vmult(y, x);
      check_rows(1.);
      matrix.vmult_add(y, x);
      check_rows(2.);
      matrix.Tvmult(z, u);
      check_columns(1.);
      matrix.Tvmult_add(z, u);
      check_columns(2.);
    }
  if (rank == 0)
    deallog
      << "Native 2-by-3 block products, transposes, and empty partitions OK"
      << std::endl;
}
