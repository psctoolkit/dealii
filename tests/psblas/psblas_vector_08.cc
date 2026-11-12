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

#include <deal.II/lac/psblas_vector.h>

#include "../tests.h"

// Repeated remote additions must preserve the existing entries and must not
// replay contributions from a previous assembly. Only one process sends in
// some rounds, so the receiving processes must also enter native update state.

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  mpi_initlog();
  const MPI_Comm     comm        = MPI_COMM_WORLD;
  const unsigned int rank        = Utilities::MPI::this_mpi_process(comm);
  const unsigned int n_processes = Utilities::MPI::n_mpi_processes(comm);
  AssertThrow(n_processes > 1, ExcInternalError());

  IndexSet owned(n_processes);
  owned.add_index(rank);
  PSCToolkitWrappers::Vector v(owned, comm);
  auto *const                native_handle = v.get_psblas_vector();
  v(rank)                                  = 10. + rank;
  v.compress(VectorOperation::insert);

  const auto check = [&](const double expected) {
    AssertThrow(std::abs(static_cast<double>(v(rank)) - expected) < 1.e-12,
                ExcInternalError());
    AssertThrow(v.get_psblas_vector() == native_handle, ExcInternalError());
  };

  v.add(std::vector<types::global_dof_index>{(rank + 1) % n_processes},
        std::vector<double>{1. + rank});
  v.compress(VectorOperation::add);
  double expected = 10. + rank + 1. + (rank + n_processes - 1) % n_processes;
  check(expected);

  if (rank == 0)
    v.add(std::vector<types::global_dof_index>{1}, std::vector<double>{5.});
  v.compress(VectorOperation::add);
  expected += rank == 1 ? 5. : 0.;
  check(expected);

  if (rank == 1)
    v.add(std::vector<types::global_dof_index>{0}, std::vector<double>{7.});
  v.compress(VectorOperation::add);
  expected += rank == 0 ? 7. : 0.;
  check(expected);
  v.compress(VectorOperation::add);
  check(expected);

  // Overwrite assembled storage, including with zero, without changing the
  // public insert/compress interface.
  v = 0.;
  v.compress(VectorOperation::insert);
  check(0.);
  v(rank) = 2. + rank;
  v.compress(VectorOperation::insert);
  check(2. + rank);
  v(rank) = 0.;
  v.compress(VectorOperation::insert);
  check(0.);

  v.add(std::vector<types::global_dof_index>{(rank + 1) % n_processes},
        std::vector<double>{3.});
  v.compress(VectorOperation::add);
  check(3.);

  if (rank == 0)
    deallog << "OK" << std::endl;
}
