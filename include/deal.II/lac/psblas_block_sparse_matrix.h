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

#ifndef dealii_psblas_block_sparse_matrix_h
#define dealii_psblas_block_sparse_matrix_h

#include <deal.II/base/config.h>

#ifdef DEAL_II_WITH_PSBLAS

#  include <deal.II/lac/block_matrix_base.h>
#  include <deal.II/lac/block_sparsity_pattern.h>
#  include <deal.II/lac/psblas_block_vector.h>
#  include <deal.II/lac/psblas_sparse_matrix.h>

DEAL_II_NAMESPACE_OPEN

namespace PSCToolkitWrappers
{
  /**
   * A distributed block matrix with PSCToolkitWrappers::SparseMatrix blocks.
   * The row and column spaces of each block have independent parallel
   * partitionings, so off-diagonal blocks may be rectangular. Every block is
   * distributed over the same MPI communicator.
   *
   * The matrix acts on PSCToolkitWrappers::BlockVector objects. Each block
   * product passes the native PSBLAS vector handles directly to PSBLAS; the
   * source and destination blocks must have the corresponding locally owned
   * column and row partitionings.
   *
   * Initialize all blocks through a partition-based reinit() overload, or
   * initialize the blocks individually after reinit(n_block_rows,
   * n_block_columns) and then call collect_sizes(). Matrix entries may then
   * be assembled using the global block-matrix indices provided by
   * BlockMatrixBase or the block-local indices of an individual block.
   *
   * @ingroup PSCToolkitWrappers
   * @ingroup Matrix1
   * @see @ref GlossBlockLA "Block (linear algebra)"
   */
  class BlockSparseMatrix : public BlockMatrixBase<SparseMatrix>
  {
  public:
    using BaseClass  = BlockMatrixBase<SparseMatrix>;
    using BlockType  = BaseClass::BlockType;
    using value_type = BaseClass::value_type;
    using size_type  = BaseClass::size_type;

    /**
     * Construct an empty matrix without any blocks.
     */
    BlockSparseMatrix() = default;

    /**
     * Construct a matrix with uninitialized blocks. Initialize the blocks
     * separately and call collect_sizes() before using the matrix.
     */
    BlockSparseMatrix(const unsigned int n_block_rows,
                      const unsigned int n_block_columns);

    /**
     * Copy construction is disabled.
     */
    BlockSparseMatrix(const BlockSparseMatrix &) = delete;

    /**
     * Copy assignment is disabled.
     */
    BlockSparseMatrix &
    operator=(const BlockSparseMatrix &) = delete;

    /**
     * Replace the matrix by the specified number of uninitialized blocks.
     * Call collect_sizes() after initializing the individual blocks.
     */
    void
    reinit(const unsigned int n_block_rows, const unsigned int n_block_columns);

    /**
     * Initialize blocks with independent row and column partitionings.
     * Indices in each IndexSet are relative to the corresponding block.
     */
    void
    reinit(const std::vector<IndexSet> &rows,
           const std::vector<IndexSet> &columns,
           const MPI_Comm               communicator = MPI_COMM_WORLD);

    /**
     * Initialize a square block structure with the same row and column
     * partitionings.
     */
    void
    reinit(const std::vector<IndexSet> &partitioning,
           const MPI_Comm               communicator = MPI_COMM_WORLD);

    /**
     * Initialize the block structure from a sparsity pattern and independent
     * row and column partitionings. The pattern must describe the entries
     * in the locally owned rows of every block.
     */
    void
    reinit(const std::vector<IndexSet>       &rows,
           const std::vector<IndexSet>       &columns,
           const BlockDynamicSparsityPattern &sparsity_pattern,
           const MPI_Comm                     communicator = MPI_COMM_WORLD);

    /**
     * Initialize from a sparsity pattern with equal row and column
     * partitionings.
     */
    void
    reinit(const std::vector<IndexSet>       &partitioning,
           const BlockDynamicSparsityPattern &sparsity_pattern,
           const MPI_Comm                     communicator = MPI_COMM_WORLD);

    /**
     * Assemble all blocks using their pending insertion operation.
     */
    void
    compress();

    /**
     * Assemble all blocks with the specified insertion operation.
     */
    void
    compress(const VectorOperation::values operation);

    /**
     * Compute <tt>dst = M*src</tt> using native PSBLAS vector blocks.
     */
    void
    vmult(BlockVector &dst, const BlockVector &src) const;

    /**
     * Compute <tt>dst += M*src</tt> using native PSBLAS vector blocks.
     */
    void
    vmult_add(BlockVector &dst, const BlockVector &src) const;

    /**
     * Compute <tt>dst = M^T*src</tt> using native PSBLAS vector blocks.
     */
    void
    Tvmult(BlockVector &dst, const BlockVector &src) const;

    /**
     * Compute <tt>dst += M^T*src</tt> using native PSBLAS vector blocks.
     */
    void
    Tvmult_add(BlockVector &dst, const BlockVector &src) const;

    /**
     * Return the locally owned column indices for each block column.
     * A matrix with block columns must have at least one block row.
     */
    std::vector<IndexSet>
    locally_owned_domain_indices() const;

    /**
     * Return the locally owned row indices for each block row.
     * A matrix with block rows must have at least one block column.
     */
    std::vector<IndexSet>
    locally_owned_range_indices() const;

    /**
     * Return the total number of stored entries over all blocks and processes.
     */
    std::uint64_t
    n_nonzero_elements() const;

    /**
     * Return the MPI communicator shared by the blocks, or MPI_COMM_SELF for
     * an empty matrix.
     */
    MPI_Comm
    get_mpi_communicator() const;

    /**
     * Release all blocks and reset the matrix to its default state.
     */
    using BaseClass::clear;

    /**
     * Update the block index maps after changing individual block sizes.
     */
    using BaseClass::collect_sizes;
  };
} // namespace PSCToolkitWrappers

DEAL_II_NAMESPACE_CLOSE

#endif // DEAL_II_WITH_PSBLAS

#endif
