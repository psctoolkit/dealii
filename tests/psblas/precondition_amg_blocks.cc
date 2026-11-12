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
#include <deal.II/base/init_finalize.h>
#include <deal.II/base/mpi.h>

#include <deal.II/lac/psblas_block_sparse_matrix.h>
#include <deal.II/lac/psblas_block_vector.h>
#include <deal.II/lac/psblas_precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>

#include <array>
#include <vector>

#include "../tests.h"

using namespace dealii;

namespace
{
  void
  fill_diagonal_block(PSCToolkitWrappers::SparseMatrix &matrix,
                      const IndexSet                   &owned)
  {
    for (const auto i : owned)
      {
        matrix.set(i, i, 4.0);
        if (i > 0)
          matrix.set(i, i - 1, -1.0);
        if (i + 1 < owned.size())
          matrix.set(i, i + 1, -1.0);
      }
  }


  // The scalar interface and its native vector application remain available.
  // Reusing the same object also exercises clear() before and after setup.
  void
  check_scalar_interface()
  {
    const MPI_Comm     communicator = MPI_COMM_WORLD;
    const unsigned int rank = Utilities::MPI::this_mpi_process(communicator);
    const unsigned int n_processes =
      Utilities::MPI::n_mpi_processes(communicator);
    IndexSet owned(12 * n_processes);
    owned.add_range(12 * rank, 12 * (rank + 1));

    PSCToolkitWrappers::SparseMatrix matrix;
    matrix.reinit(owned, communicator);
    fill_diagonal_block(matrix, owned);
    matrix.compress();

    PSCToolkitWrappers::Vector rhs(owned, communicator);
    PSCToolkitWrappers::Vector solution(owned, communicator);
    PSCToolkitWrappers::Vector residual(owned, communicator);
    rhs = 1.0;

    PSCToolkitWrappers::PreconditionAMG preconditioner;
    preconditioner.clear();
    preconditioner.clear();
    preconditioner.initialize(matrix);
    preconditioner.vmult(solution, rhs);
    preconditioner.clear();
    preconditioner.initialize(matrix);

    // Existing users may configure options not exposed by AdditionalData
    // through the native handle before calling initialize(). Preserve that
    // handle, including its native settings, during hierarchy construction.
    PSCToolkitWrappers::PreconditionAMG configured_preconditioner;
    auto *handle = configured_preconditioner.get_psblas_preconditioner();
    int   native_error =
      amg_c_dprecinit(*InitFinalize::get_psblas_context(), handle, "ML");
    AssertThrow(native_error == 0, ExcInternalError());
    native_error = amg_c_dprecseti(handle, "MIN_COARSE_SIZE", 4);
    AssertThrow(native_error == 0, ExcInternalError());
    const void *native_preconditioner = handle->dprec;
    configured_preconditioner.initialize(matrix);
    configured_preconditioner.vmult(solution, rhs);
    configured_preconditioner.initialize(matrix);
    AssertThrow(configured_preconditioner.get_psblas_preconditioner() ==
                    handle &&
                  handle->dprec == native_preconditioner,
                ExcInternalError());

    solution = 0.0;
    SolverControl                        control(100, 1.e-12);
    SolverCG<PSCToolkitWrappers::Vector> solver(control);
    solver.solve(matrix, solution, rhs, configured_preconditioner);
    matrix.vmult(residual, solution);
    residual.add(-1.0, rhs);
    AssertThrow(residual.l2_norm() < 1.e-10, ExcInternalError());
  }


  class BlockDiagonalAMG
  {
  public:
    void
    initialize(const PSCToolkitWrappers::BlockSparseMatrix &matrix)
    {
      for (unsigned int b = 0; b < preconditioners.size(); ++b)
        preconditioners[b].initialize(matrix.block(b, b));
    }

    void
    vmult(PSCToolkitWrappers::BlockVector       &dst,
          const PSCToolkitWrappers::BlockVector &src) const
    {
      for (unsigned int b = 0; b < preconditioners.size(); ++b)
        preconditioners[b].vmult(dst.block(b), src.block(b));
    }

  private:
    std::array<PSCToolkitWrappers::PreconditionAMG, 2> preconditioners;
  };


  void
  check_block_interface(const MPI_Comm communicator, const bool empty_rank)
  {
    const unsigned int rank = Utilities::MPI::this_mpi_process(communicator);
    const unsigned int n_processes =
      Utilities::MPI::n_mpi_processes(communicator);
    const unsigned int n_empty_ranks = empty_rank ? 1 : 0;
    AssertThrow(n_processes > n_empty_ranks, ExcInternalError());

    // Exceed AMG4PSBLAS's default coarse-size threshold, so these solves
    // exercise the multilevel hierarchy rather than only the coarse solver.
    std::vector<IndexSet> owned = {IndexSet(768 * n_processes),
                                   IndexSet(512 *
                                            (n_processes - n_empty_ranks))};
    owned[0].add_range(768 * rank, 768 * (rank + 1));
    if (rank >= n_empty_ranks)
      owned[1].add_range(512 * (rank - n_empty_ranks),
                         512 * (rank + 1 - n_empty_ranks));

    PSCToolkitWrappers::BlockSparseMatrix matrix;
    matrix.reinit(owned, communicator);
    for (unsigned int b = 0; b < 2; ++b)
      fill_diagonal_block(matrix.block(b, b), owned[b]);
    for (const auto i : owned[0])
      if (i < owned[1].size())
        matrix.block(0, 1).set(i, i, 0.25);
    for (const auto i : owned[1])
      matrix.block(1, 0).set(i, i, 0.25);
    matrix.compress();

    PSCToolkitWrappers::BlockVector exact(owned, communicator);
    PSCToolkitWrappers::BlockVector rhs(owned, communicator);
    PSCToolkitWrappers::BlockVector solution(owned, communicator);
    PSCToolkitWrappers::BlockVector residual(owned, communicator);
    for (unsigned int b = 0; b < 2; ++b)
      for (const auto i : owned[b])
        exact.block(b)(i) = 1.0 + 0.01 * i + 0.2 * b;
    exact.compress(VectorOperation::insert);
    matrix.vmult(rhs, exact);

    BlockDiagonalAMG preconditioner;
    preconditioner.initialize(matrix);
    const auto rhs_handle      = rhs.block(1).get_psblas_vector();
    const auto solution_handle = solution.block(1).get_psblas_vector();
    const auto rhs_native      = rhs_handle->dvector;
    const auto solution_native = solution_handle->dvector;
    preconditioner.vmult(solution, rhs);
    AssertThrow(rhs.block(1).get_psblas_vector() == rhs_handle &&
                  rhs_handle->dvector == rhs_native &&
                  solution.block(1).get_psblas_vector() == solution_handle &&
                  solution_handle->dvector == solution_native,
                ExcInternalError());

    solution = 0.0;
    SolverControl                             control(100, 1.e-12);
    SolverCG<PSCToolkitWrappers::BlockVector> solver(control);
    solver.solve(matrix, solution, rhs, preconditioner);
    matrix.vmult(residual, solution);
    residual.add(-1.0, rhs);
    AssertThrow(residual.l2_norm() < 1.e-10, ExcInternalError());
    solution.add(-1.0, exact);
    AssertThrow(solution.l2_norm() < 1.e-10, ExcInternalError());
  }
} // namespace


int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  mpi_initlog();
  const unsigned int rank = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);

  check_scalar_interface();
  if (rank == 0)
    deallog << "Scalar AMG reuse OK" << std::endl;
  check_block_interface(MPI_COMM_WORLD, false);
  if (rank == 0)
    deallog << "Native block AMG OK" << std::endl;
  check_block_interface(MPI_COMM_WORLD, true);
  if (rank == 0)
    deallog << "Empty rank AMG OK" << std::endl;

  MPI_Comm  subcommunicator;
  const int ierr =
    MPI_Comm_split(MPI_COMM_WORLD, rank % 2, rank, &subcommunicator);
  AssertThrowMPI(ierr);
  check_block_interface(subcommunicator, false);
  const int free_error = MPI_Comm_free(&subcommunicator);
  AssertThrowMPI(free_error);
  if (rank == 0)
    deallog << "Subcommunicator AMG OK" << std::endl;
}
