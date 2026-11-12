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

#ifndef dealii_psblas_block_vector_h
#define dealii_psblas_block_vector_h

#include <deal.II/base/config.h>

#ifdef DEAL_II_WITH_PSBLAS

#  include <deal.II/lac/block_vector_base.h>
#  include <deal.II/lac/psblas_vector.h>

DEAL_II_NAMESPACE_OPEN

namespace PSCToolkitWrappers
{
  /**
   * A block vector whose blocks are PSCToolkitWrappers::Vector objects.
   * Each block is distributed over the same MPI communicator, with its own
   * IndexSet describing the locally owned entries. Block operations act
   * directly on these native PSBLAS vectors.
   *
   * The constructors and reinit() functions take one IndexSet per block.
   * Indices in these sets are relative to the corresponding block. As with
   * other block vectors, call collect_sizes() after changing the size of an
   * individual block through block().
   *
   * @ingroup PSCToolkitWrappers
   * @ingroup Vectors
   * @see @ref GlossBlockLA "Block (linear algebra)"
   */
  class BlockVector : public BlockVectorBase<Vector>
  {
  public:
    using BaseClass  = BlockVectorBase<Vector>;
    using BlockType  = BaseClass::BlockType;
    using value_type = BaseClass::value_type;
    using size_type  = BaseClass::size_type;

    /**
     * Construct an empty vector without any blocks.
     */
    BlockVector() = default;

    /**
     * Construct @p n_blocks empty blocks. Initialize the individual blocks
     * through block() and call collect_sizes(), or use reinit().
     */
    explicit BlockVector(const unsigned int n_blocks);

    /**
     * Construct one block per IndexSet, with the given parallel partitioning.
     */
    explicit BlockVector(const std::vector<IndexSet> &parallel_partitioning,
                         const MPI_Comm communicator = MPI_COMM_WORLD);

    /**
     * Construct a block vector with read-only ghost entries in each block.
     */
    BlockVector(const std::vector<IndexSet> &parallel_partitioning,
                const std::vector<IndexSet> &ghost_indices,
                const MPI_Comm               communicator = MPI_COMM_WORLD);

    /**
     * Copy the block structure, partitioning, and values of @p v.
     */
    BlockVector(const BlockVector &v);

    /**
     * Assign @p value to all entries in all blocks.
     */
    BlockVector &
    operator=(const value_type value);

    /**
     * Copy the entries of @p v. Resize the number of blocks if necessary.
     * Assignment to each existing block follows Vector::operator=().
     */
    BlockVector &
    operator=(const BlockVector &v);

    /**
     * Replace the vector by @p n_blocks empty blocks.
     */
    void
    reinit(const unsigned int n_blocks);

    /**
     * Reinitialize each block with the given partitioning. The entries are
     * zeroed unless @p omit_zeroing_entries is true.
     */
    void
    reinit(const std::vector<IndexSet> &parallel_partitioning,
           const MPI_Comm               communicator         = MPI_COMM_WORLD,
           const bool                   omit_zeroing_entries = false);

    /**
     * Reinitialize the blocks with the given partitioning and ghost entries.
     */
    void
    reinit(const std::vector<IndexSet> &parallel_partitioning,
           const std::vector<IndexSet> &ghost_indices,
           const MPI_Comm               communicator = MPI_COMM_WORLD);

    /**
     * Copy the block structure and partitioning of @p v, without copying its
     * values. The entries are zeroed unless @p omit_zeroing_entries is true.
     */
    void
    reinit(const BlockVector &v, const bool omit_zeroing_entries = false);

    /**
     * Release all blocks and reset the vector to its default state.
     */
    void
    clear();

    /**
     * Return whether at least one block has ghost entries.
     */
    bool
    has_ghost_elements() const;

    /**
     * Return whether the entry at the global block-vector index @p index is
     * locally owned.
     */
    bool
    in_local_range(const size_type index) const;

    /**
     * Exchange the block structure and values with @p v.
     */
    void
    swap(BlockVector &v) noexcept;
  };

  /**
   * Exchange two PSBLAS block vectors.
   */
  inline void
  swap(BlockVector &u, BlockVector &v) noexcept
  {
    u.swap(v);
  }
} // namespace PSCToolkitWrappers

/**
 * Declare PSCToolkitWrappers::BlockVector as a distributed vector.
 */
template <>
struct is_serial_vector<PSCToolkitWrappers::BlockVector> : std::false_type
{};

DEAL_II_NAMESPACE_CLOSE

#endif // DEAL_II_WITH_PSBLAS

#endif
