// ------------------------------------------------------------------------
//
// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (C) 2017 - 2022 by the deal.II authors
//
// This file is part of the deal.II library.
//
// Part of the source code is dual licensed under Apache-2.0 WITH
// LLVM-exception OR LGPL-2.1-or-later. Detailed license information
// governing the source code and code contributions can be found in
// LICENSE.md and CONTRIBUTING.md at the top level directory of deal.II.
//
// ------------------------------------------------------------------------

#include <deal.II/base/logstream.h>
#include <deal.II/base/types.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/lac/psblas_vector.h>

#include <psb_c_dbase.h>

#include <iostream>

#include "../tests.h"

using namespace dealii;

template <typename VectorType>
void
test_vector_operations(const IndexSet &local_indices, VectorType &vector)
{
  for (const types::global_dof_index idx : local_indices)
    vector(idx) = 1.;
  vector.compress(VectorOperation::insert);

  // now we add 10 to all entries, and we check that the result is 11
  for (const types::global_dof_index idx : local_indices)
    vector(idx) += 10.;
  vector.compress(VectorOperation::add);

  double expected_value = 11.;

  AssertThrow(
    std::all_of(vector.begin(),
                vector.end(),
                [expected_value](const double val) {
                  return val == expected_value;
                }),
    ExcMessage(
      "Not all values are equal to the expected value after add operation."));

  // Reset the values to 0, and check that the result is 0
  vector = 0.;
  AssertThrow(vector.l2_norm() == 0., ExcMessage("Vector should be 0."));
}



template <typename VectorType>
void
test_vector_operations_with_remote_entries(
  const IndexSet &locally_owned_dofs,
  const IndexSet &locally_relevant_dofs,
  VectorType     &vector)
{
  // set all relevant entries to 1
  for (const types::global_dof_index idx : locally_relevant_dofs)
    vector(idx) = 1.;
  vector.compress(VectorOperation::insert);

  //  add 10 to all entries
  for (const types::global_dof_index idx : locally_relevant_dofs)
    vector(idx) += 10.;
  vector.compress(VectorOperation::add);

  // With relevant sets:
  // rank 0: [0,20), rank 1: [10,25)
  // indices [10,20) are touched by both ranks -> 1 + 10 + 10 = 21
  // all other indices are touched by one rank only -> 1 + 10 = 11
  for (const types::global_dof_index idx : locally_owned_dofs)
    {
      const double expected = (idx >= 10 && idx < 20) ? 21.0 : 11.0;
      AssertThrow(vector(idx) == expected,
                  ExcMessage("Unexpected value after remote set/add test."));
    }

  vector = 0.;
  AssertThrow(vector.l2_norm() == 0., ExcMessage("Vector should be 0."));
}



int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  MPI_Comm                         mpi_communicator = MPI_COMM_WORLD;

  AssertThrow(Utilities::MPI::n_mpi_processes(mpi_communicator) == 2,
              ExcMessage("This test needs to be run with 2 MPI processes."));

  initlog();

  int id;
  MPI_Comm_rank(mpi_communicator, &id);
  IndexSet locally_owned_dofs(25);
  if (id == 0)
    locally_owned_dofs.add_range(0, 15);
  else if (id == 1)
    locally_owned_dofs.add_range(15, 25);

  PSCToolkitWrappers::Vector psblas_vector(locally_owned_dofs,
                                           mpi_communicator);

  test_vector_operations(locally_owned_dofs, psblas_vector);
  deallog << "Test switching insertion mode (local): OK" << std::endl;

  IndexSet locally_relevant_dofs(25);
  if (id == 0)
    locally_relevant_dofs.add_range(0, 20);
  else if (id == 1)
    locally_relevant_dofs.add_range(10, 25);

  PSCToolkitWrappers::Vector psblas_vector_remote(locally_owned_dofs,
                                                  mpi_communicator);
  test_vector_operations_with_remote_entries(locally_owned_dofs,
                                             locally_relevant_dofs,
                                             psblas_vector_remote);
  deallog << "Test switching insertion mode (with remote relevant dofs): OK"
          << std::endl;

  return 0;
}