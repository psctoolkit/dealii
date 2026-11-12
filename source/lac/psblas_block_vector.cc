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

#ifdef DEAL_II_WITH_PSBLAS

DEAL_II_NAMESPACE_OPEN

namespace PSCToolkitWrappers
{
  BlockVector::BlockVector(const unsigned int n_blocks)
  {
    reinit(n_blocks);
  }



  BlockVector::BlockVector(const std::vector<IndexSet> &parallel_partitioning,
                           const MPI_Comm               communicator)
  {
    reinit(parallel_partitioning, communicator);
  }



  BlockVector::BlockVector(const std::vector<IndexSet> &parallel_partitioning,
                           const std::vector<IndexSet> &ghost_indices,
                           const MPI_Comm               communicator)
  {
    reinit(parallel_partitioning, ghost_indices, communicator);
  }



  BlockVector::BlockVector(const BlockVector &v)
  {
    reinit(v, true);
    *this = v;
  }



  BlockVector &
  BlockVector::operator=(const value_type value)
  {
    BaseClass::operator=(value);
    return *this;
  }



  BlockVector &
  BlockVector::operator=(const BlockVector &v)
  {
    if (this == &v)
      return *this;

    if (n_blocks() != v.n_blocks())
      reinit(v.n_blocks());

    for (unsigned int i = 0; i < n_blocks(); ++i)
      if (v.block(i).get_psblas_vector() == nullptr)
        block(i).clear();
      else
        {
          if (block(i).get_psblas_vector() == nullptr)
            block(i).reinit(v.block(i), true);
          block(i) = v.block(i);
        }

    collect_sizes();
    return *this;
  }



  void
  BlockVector::reinit(const unsigned int n_blocks)
  {
    std::vector<Vector> empty_blocks(n_blocks);
    components.swap(empty_blocks);
    block_indices.reinit(n_blocks, 0);
  }



  void
  BlockVector::reinit(const std::vector<IndexSet> &parallel_partitioning,
                      const MPI_Comm               communicator,
                      const bool                   omit_zeroing_entries)
  {
    if (n_blocks() != parallel_partitioning.size())
      reinit(parallel_partitioning.size());
    for (unsigned int i = 0; i < n_blocks(); ++i)
      block(i).reinit(parallel_partitioning[i],
                      communicator,
                      omit_zeroing_entries);
    collect_sizes();
  }



  void
  BlockVector::reinit(const std::vector<IndexSet> &parallel_partitioning,
                      const std::vector<IndexSet> &ghost_indices,
                      const MPI_Comm               communicator)
  {
    AssertDimension(parallel_partitioning.size(), ghost_indices.size());
    if (n_blocks() != parallel_partitioning.size())
      reinit(parallel_partitioning.size());
    for (unsigned int i = 0; i < n_blocks(); ++i)
      block(i).reinit(parallel_partitioning[i], ghost_indices[i], communicator);
    collect_sizes();
  }



  void
  BlockVector::reinit(const BlockVector &v, const bool omit_zeroing_entries)
  {
    if (n_blocks() != v.n_blocks())
      reinit(v.n_blocks());
    for (unsigned int i = 0; i < n_blocks(); ++i)
      if (v.block(i).get_psblas_vector() == nullptr)
        block(i).clear();
      else
        block(i).reinit(v.block(i), omit_zeroing_entries);
    collect_sizes();
  }



  void
  BlockVector::clear()
  {
    reinit(0);
  }



  bool
  BlockVector::has_ghost_elements() const
  {
    for (const auto &component : components)
      if (component.has_ghost_elements())
        return true;
    return false;
  }



  bool
  BlockVector::in_local_range(const size_type index) const
  {
    const auto local_index = block_indices.global_to_local(index);
    return block(local_index.first).in_local_range(local_index.second);
  }



  void
  BlockVector::swap(BlockVector &v) noexcept
  {
    components.swap(v.components);
    block_indices.swap(v.block_indices);
  }
} // namespace PSCToolkitWrappers

DEAL_II_NAMESPACE_CLOSE

#endif // DEAL_II_WITH_PSBLAS
