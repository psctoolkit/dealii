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

#include <deal.II/lac/psblas_block_vector.h>

#include "../tests.h"

// Native block-vector algebra, noncontiguous ownership, empty local blocks,
// ghost values, and construction from individually initialized scalar blocks.

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  mpi_initlog();
  const MPI_Comm     comm        = MPI_COMM_WORLD;
  const unsigned int rank        = Utilities::MPI::this_mpi_process(comm);
  const unsigned int n_processes = Utilities::MPI::n_mpi_processes(comm);
  AssertThrow(n_processes > 1, ExcInternalError());

  std::vector<IndexSet> owned{IndexSet(2 * n_processes), IndexSet(n_processes)};
  owned[0].add_index(rank);
  owned[0].add_index(rank + n_processes);
  if (rank + 1 < n_processes)
    owned[1].add_index(rank);
  if (rank == 0)
    owned[1].add_index(n_processes - 1);

  PSCToolkitWrappers::BlockVector v(owned, comm);
  for (const auto i : v.locally_owned_elements())
    v(i) = i + 1.;
  v.compress(VectorOperation::insert);

  const double n            = v.size();
  const double squared_norm = n * (n + 1.) * (2. * n + 1.) / 6.;
  const auto   check_norm   = [&](const PSCToolkitWrappers::BlockVector &vector,
                              const double                           scale) {
    AssertThrow(std::abs(vector.norm_sqr() - scale * scale * squared_norm) <
                  1.e-10,
                ExcInternalError());
  };

  check_norm(v, 1.);
  AssertThrow(std::abs(v.mean_value() - (n + 1.) / 2.) < 1.e-12,
              ExcInternalError());
  AssertThrow(v.is_non_negative(), ExcInternalError());
  for (types::global_dof_index i = 0; i < v.size(); ++i)
    AssertThrow(v.in_local_range(i) == v.locally_owned_elements().is_element(i),
                ExcInternalError());

  PSCToolkitWrappers::BlockVector w(v);
  w *= 2.;
  w /= 2.;
  check_norm(w, 1.);
  w.add(2., v, -1., v);
  check_norm(w, 2.);
  w.sadd(0.5, 2., w, -1., v);
  check_norm(w, 4.);
  w.sadd(0.5, 1., v, -0.5, w, 1., v);
  check_norm(w, 2.);
  w.equ(1., v);
  AssertThrow(std::abs(w.add_and_dot(2., v, v) - 3. * squared_norm) < 1.e-10,
              ExcInternalError());

  std::vector<IndexSet>           ghosts{complete_index_set(owned[0].size()),
                               complete_index_set(owned[1].size())};
  PSCToolkitWrappers::BlockVector ghosted(owned, ghosts, comm);
  ghosted = v;
  ghosted.update_ghost_values();
  const auto &read_ghosted = ghosted;
  for (types::global_dof_index i = 0; i < v.size(); ++i)
    AssertThrow(std::abs(read_ghosted(i) - (i + 1.)) < 1.e-12,
                ExcInternalError());

  // Scalar extraction must recognize noncontiguously owned entries as well
  // as ghosts, including when a process owns no entries in this block.
  for (unsigned int b = 0; b < 2; ++b)
    {
      const auto          indices = ghosts[b].get_index_vector();
      std::vector<double> values(indices.size());
      read_ghosted.block(b).extract_subvector_to(indices, values);
      for (unsigned int i = 0; i < indices.size(); ++i)
        AssertThrow(std::abs(values[i] - (v.get_block_indices().block_start(b) +
                                          indices[i] + 1.)) < 1.e-12,
                    ExcInternalError());
    }

  PSCToolkitWrappers::BlockVector individually_initialized(2);
  for (unsigned int b = 0; b < 2; ++b)
    individually_initialized.block(b).reinit(owned[b], comm);
  individually_initialized.collect_sizes();
  individually_initialized = v;
  check_norm(individually_initialized, 1.);
  individually_initialized.reinit(v);
  AssertThrow(individually_initialized.all_zero(), ExcInternalError());

  PSCToolkitWrappers::BlockVector empty;
  swap(empty, individually_initialized);
  AssertThrow(individually_initialized.n_blocks() == 0, ExcInternalError());
  AssertThrow(empty.size() == v.size(), ExcInternalError());
  empty.clear();
  AssertThrow(empty.size() == 0 && empty.n_blocks() == 0, ExcInternalError());

  if (rank == 0)
    deallog << "OK" << std::endl;
}
