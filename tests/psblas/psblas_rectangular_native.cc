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

#include <deal.II/lac/psblas_sparse_matrix.h>
#include <deal.II/lac/psblas_vector.h>

#include "../tests.h"

// Independent, interleaved partitions; remote assembly; direct native handles.
namespace
{
  double
  entry(const unsigned int i, const unsigned int j)
  {
    return static_cast<int>((3 * i + 5 * j) % 7) - 3.;
  }

  void
  check(PSCToolkitWrappers::SparseMatrix &matrix,
        const IndexSet                   &rows,
        const IndexSet                   &columns,
        const MPI_Comm                    comm)
  {
    PSCToolkitWrappers::Vector x(columns, comm), y(rows, comm);
    PSCToolkitWrappers::Vector u(rows, comm), z(columns, comm);
    for (const auto j : columns)
      x(j) = 1. + j;
    for (const auto i : rows)
      u(i) = 2. - 0.25 * i;
    x.compress(VectorOperation::insert);
    u.compress(VectorOperation::insert);
    auto *const x_handle   = x.get_psblas_vector();
    auto *const y_handle   = y.get_psblas_vector();
    const auto  check_rows = [&](const double scale) {
      for (const auto i : rows)
        {
          double expected = 0.;
          for (unsigned int j = 0; j < columns.size(); ++j)
            expected += entry(i, j) * (1. + j);
          AssertThrow(std::abs(y(i) - scale * expected) < 1.e-10,
                      ExcInternalError());
        }
    };
    const auto check_columns = [&](const double scale) {
      for (const auto j : columns)
        {
          double expected = 0.;
          for (unsigned int i = 0; i < rows.size(); ++i)
            expected += entry(i, j) * (2. - 0.25 * i);
          AssertThrow(std::abs(z(j) - scale * expected) < 1.e-10,
                      ExcInternalError());
        }
    };
    matrix.vmult(y, x);
    check_rows(1.);
    matrix.vmult_add(y, x);
    check_rows(2.);
    matrix.Tvmult(z, u);
    check_columns(1.);
    matrix.Tvmult_add(z, u);
    check_columns(2.);
    matrix.vmult(y, x);
    check_rows(1.);
    AssertThrow(x_handle == x.get_psblas_vector() &&
                  y_handle == y.get_psblas_vector(),
                ExcInternalError());
    for (unsigned int i = 0; i < rows.size(); ++i)
      AssertThrow(matrix.in_local_range(i) == rows.is_element(i),
                  ExcInternalError());
    for (const auto i : rows)
      for (unsigned int j = 0; j < columns.size(); ++j)
        AssertThrow(std::abs(matrix.el(i, j) - entry(i, j)) < 1.e-12,
                    ExcInternalError());
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
  IndexSet           rows(5 * np + 1), columns(3 * np + 2);
  for (unsigned int i = 0; i < rows.size(); ++i)
    if (i % np == rank)
      rows.add_index(i);
  for (unsigned int j = 0; j < columns.size(); ++j)
    if ((j + 1) % np == rank)
      columns.add_index(j);

  DynamicSparsityPattern pattern(rows.size(), columns.size(), rows);
  for (const auto i : rows)
    for (unsigned int j = 0; j < columns.size(); ++j)
      pattern.add(i, j);
  PSCToolkitWrappers::SparseMatrix matrix;
  matrix.reinit(rows, columns, pattern, comm);
  for (unsigned int i = 0; i < rows.size(); ++i)
    for (unsigned int j = 0; j < columns.size(); ++j)
      {
        // Every process contributes, including to nonlocal rows.
        const double contribution = entry(i, j) / (np * (np + 1.) / 2.);
        matrix.add(i, j, (rank + 1.) * 0.25 * contribution);
        matrix.add(i, j, (rank + 1.) * 0.75 * contribution);
      }
  matrix.compress(VectorOperation::add);
  check(matrix, rows, columns, comm);
  if (rank == 0)
    deallog << "Rectangular native products and remote additions OK"
            << std::endl;

  matrix.reinit(rows, columns, pattern, comm);
  for (unsigned int i = 0; i < rows.size(); ++i)
    for (unsigned int j = 0; j < columns.size(); ++j)
      {
        matrix.set(i, j, 99. + rank);
        matrix.set(i, j, entry(i, j));
      }
  matrix.compress(VectorOperation::insert);
  check(matrix, rows, columns, comm);
  if (rank == 0)
    deallog << "Repeated and remote set, including zero overwrites, OK"
            << std::endl;
}
