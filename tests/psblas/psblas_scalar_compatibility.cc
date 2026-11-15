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

// Define the solver template before the native vector's swap overload.
#include <deal.II/lac/solver_minres.h>

#include <deal.II/lac/precondition.h>
#include <deal.II/lac/psblas_sparse_matrix.h>
#include <deal.II/lac/psblas_vector.h>

#include "../tests.h"

// Keep the original scalar entry points and assembly behavior usable.
int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  mpi_initlog();
  const auto         comm = MPI_COMM_WORLD;
  const unsigned int rank = Utilities::MPI::this_mpi_process(comm);
  const unsigned int np   = Utilities::MPI::n_mpi_processes(comm);
  IndexSet           owned(2 * np);
  owned.add_index(rank);
  owned.add_index(rank + np);
  PSCToolkitWrappers::SparseMatrix matrix;
  matrix.reinit(owned, comm);
  for (const auto i : owned)
    {
      // The established single-partition API permits this sequence.
      matrix.set(i, i, 2.);
      matrix.add(i, i, 1.);
    }
  matrix.compress();
  matrix.compress();
  PSCToolkitWrappers::Vector x(owned, comm), y(owned, comm);
  for (const auto i : owned)
    x(i) = i + 1.;
  x.compress(VectorOperation::insert);
  const auto check = [&](const double scale) {
    for (const auto i : owned)
      AssertThrow(std::abs(y(i) - scale * (i + 1.)) < 1.e-12,
                  ExcInternalError());
  };
  matrix.vmult(y, x);
  check(3.);
  matrix.vmult_add(y, x);
  check(6.);
  matrix.Tvmult(y, x);
  check(3.);
  matrix.Tvmult_add(y, x);
  check(6.);

  IndexSet contiguous(2 * np);
  contiguous.add_range(2 * rank, 2 * rank + 2);
  DynamicSparsityPattern pattern(2 * np, 2 * np, contiguous);
  for (const auto i : contiguous)
    pattern.add(i, i);
  matrix.reinit(contiguous, pattern, comm);
  for (const auto i : contiguous)
    matrix.set(i, i, 4.);
  matrix.compress();
  x.reinit(contiguous, comm);
  y.reinit(contiguous, comm);
  x = 2.;
  matrix.vmult(y, x);
  for (const auto i : contiguous)
    AssertThrow(std::abs(y(i) - 8.) < 1.e-12, ExcInternalError());
  if (rank == 0)
    deallog << "Original scalar assembly and native products OK" << std::endl;

  // Unqualified swap must exchange native storage without copying values.
  auto *const x_handle = x.get_psblas_vector();
  auto *const y_handle = y.get_psblas_vector();
  swap(x, y);
  AssertThrow(x.get_psblas_vector() == y_handle &&
                y.get_psblas_vector() == x_handle,
              ExcInternalError());
  for (const auto i : contiguous)
    {
      AssertThrow(std::abs(x(i) - 8.) < 1.e-12, ExcInternalError());
      AssertThrow(std::abs(y(i) - 2.) < 1.e-12, ExcInternalError());
    }

  // MINRES rotates scalar work vectors with the same unqualified swap.
  // Use a symmetric indefinite operator and an SPD identity preconditioner.
  matrix.reinit(contiguous, pattern, comm);
  for (const auto i : contiguous)
    {
      matrix.set(i, i, i % 2 == 0 ? 2. : -3.);
      x(i) = i + 1.;
    }
  matrix.compress();
  x.compress(VectorOperation::insert);

  PSCToolkitWrappers::Vector rhs(contiguous, comm), residual(contiguous, comm);
  matrix.vmult(rhs, x);
  y = 0.;
  SolverControl control(20, 1.e-12 * rhs.l2_norm(), false, false);
  SolverMinRes<PSCToolkitWrappers::Vector> solver(control);
  solver.solve(matrix, y, rhs, PreconditionIdentity());
  matrix.vmult(residual, y);
  residual -= rhs;
  AssertThrow(residual.l2_norm() <= 1.e-12 * rhs.l2_norm(), ExcInternalError());
  y -= x;
  AssertThrow(y.l2_norm() <= 1.e-12 * x.l2_norm(), ExcInternalError());
  if (rank == 0)
    deallog << "Scalar free swap and MINRES OK" << std::endl;
}
