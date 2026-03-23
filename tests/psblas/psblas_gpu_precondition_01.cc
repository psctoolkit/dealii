// -----------------------------------------------------------------------------
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception OR LGPL-2.1-or-later
// Copyright (C) 2025 by the deal.II authors
//
// This file is part of the deal.II library.
//
// Detailed license information governing the source code and contributions
// can be found in LICENSE.md and CONTRIBUTING.md at the top level directory.
//
// -----------------------------------------------------------------------------

// Test AMG4PSBLAS preconditioner applied on the GPU (CUDA backend).

#include <deal.II/base/index_set.h>
#include <deal.II/base/init_finalize.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/types.h>

#include <deal.II/lac/psblas_common.h>
#include <deal.II/lac/psblas_precondition.h>
#include <deal.II/lac/psblas_sparse_matrix.h>
#include <deal.II/lac/psblas_vector.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>

#include "../tests.h"

using namespace dealii;


int
main(int argc, char **argv)
{
  InitFinalize init(argc,
                    argv,
                    InitializeLibrary::MPI | InitializeLibrary::PSBLAS |
                      InitializeLibrary::PSBLASCuda);

  const MPI_Comm comm  = MPI_COMM_WORLD;
  const int      rank  = Utilities::MPI::this_mpi_process(comm);
  const int      nproc = Utilities::MPI::n_mpi_processes(comm);

  AssertThrow(nproc == 2,
              ExcMessage(
                "This test must be run with exactly 2 MPI processes."));

  initlog();

  const types::global_dof_index N = 200;
  IndexSet                      local_rows(N);
  {
    const auto per = N / static_cast<types::global_dof_index>(nproc);
    const auto rem = N % static_cast<types::global_dof_index>(nproc);
    const auto lo  = static_cast<types::global_dof_index>(rank) * per +
                    std::min(static_cast<types::global_dof_index>(rank), rem);
    const auto hi =
      lo + per + (static_cast<types::global_dof_index>(rank) < rem ? 1u : 0u);
    local_rows.add_range(lo, hi);
  }

  const PSCToolkitWrappers::StorageFormat mat_fmt =
    PSCToolkitWrappers::StorageFormat::cuda("HLG");
  const PSCToolkitWrappers::StorageFormat vec_fmt =
    PSCToolkitWrappers::StorageFormat::cuda();

  //   1D Laplacian
  PSCToolkitWrappers::SparseMatrix A;
  A.reinit(local_rows, comm, mat_fmt);

  for (const auto i : local_rows)
    {
      A.add(i, i, 2.0);
      if (i > 0)
        A.add(i, i - 1, -1.0);
      if (i < N - 1)
        A.add(i, i + 1, -1.0);
    }
  A.compress();

  PSCToolkitWrappers::PreconditionAMG                 prec;
  PSCToolkitWrappers::PreconditionAMG::AdditionalData prec_data;
  prec_data.cycle_type  = "VCYCLE";
  prec_data.aggr_prol   = "SMOOTHED";
  prec_data.n_cycles    = 1;
  prec_data.coarse_type = "ILU";
  prec.initialize(A, prec_data);


  PSCToolkitWrappers::Vector b(local_rows, comm, vec_fmt);
  PSCToolkitWrappers::Vector x_sol(local_rows, comm, vec_fmt);
  for (const auto i : local_rows)
    b.set({i}, {1.0});
  b.compress(VectorOperation::insert);

  SolverControl solver_control(500, 1e-6 * b.l2_norm(), false, false);
  SolverCG<PSCToolkitWrappers::Vector> solver(solver_control);

  check_solver_within_range(solver.solve(A, x_sol, b, prec),
                            solver_control.last_step(),
                            3,
                            10);


  return 0;
}
