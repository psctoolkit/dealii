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

#include <deal.II/lac/psblas_block_sparse_matrix.h>

#ifdef DEAL_II_WITH_PSBLAS

DEAL_II_NAMESPACE_OPEN

namespace PSCToolkitWrappers
{
  BlockSparseMatrix::BlockSparseMatrix(const unsigned int n_block_rows,
                                       const unsigned int n_block_columns)
  {
    reinit(n_block_rows, n_block_columns);
  }



  void
  BlockSparseMatrix::reinit(const unsigned int n_block_rows,
                            const unsigned int n_block_columns)
  {
    clear();
    sub_objects.reinit(n_block_rows, n_block_columns);
    row_block_indices.reinit(n_block_rows, 0);
    column_block_indices.reinit(n_block_columns, 0);
    for (unsigned int row = 0; row < n_block_rows; ++row)
      for (unsigned int column = 0; column < n_block_columns; ++column)
        sub_objects[row][column] = new BlockType();
  }



  void
  BlockSparseMatrix::reinit(const std::vector<IndexSet> &rows,
                            const std::vector<IndexSet> &columns,
                            const MPI_Comm               communicator)
  {
    reinit(rows.size(), columns.size());
    for (unsigned int row = 0; row < n_block_rows(); ++row)
      for (unsigned int column = 0; column < n_block_cols(); ++column)
        block(row, column).reinit(rows[row], columns[column], communicator);

    std::vector<size_type> row_sizes(rows.size());
    std::vector<size_type> column_sizes(columns.size());
    for (unsigned int row = 0; row < rows.size(); ++row)
      row_sizes[row] = rows[row].size();
    for (unsigned int column = 0; column < columns.size(); ++column)
      column_sizes[column] = columns[column].size();
    row_block_indices.reinit(row_sizes);
    column_block_indices.reinit(column_sizes);
  }



  void
  BlockSparseMatrix::reinit(const std::vector<IndexSet> &partitioning,
                            const MPI_Comm               communicator)
  {
    reinit(partitioning, partitioning, communicator);
  }



  void
  BlockSparseMatrix::reinit(const std::vector<IndexSet>       &rows,
                            const std::vector<IndexSet>       &columns,
                            const BlockDynamicSparsityPattern &sparsity_pattern,
                            const MPI_Comm                     communicator)
  {
    AssertDimension(rows.size(), sparsity_pattern.n_block_rows());
    AssertDimension(columns.size(), sparsity_pattern.n_block_cols());
    reinit(rows.size(), columns.size());
    for (unsigned int row = 0; row < n_block_rows(); ++row)
      for (unsigned int column = 0; column < n_block_cols(); ++column)
        block(row, column)
          .reinit(rows[row],
                  columns[column],
                  sparsity_pattern.block(row, column),
                  communicator);

    row_block_indices    = sparsity_pattern.get_row_indices();
    column_block_indices = sparsity_pattern.get_column_indices();
  }



  void
  BlockSparseMatrix::reinit(const std::vector<IndexSet>       &partitioning,
                            const BlockDynamicSparsityPattern &sparsity_pattern,
                            const MPI_Comm                     communicator)
  {
    reinit(partitioning, partitioning, sparsity_pattern, communicator);
  }



  void
  BlockSparseMatrix::compress()
  {
    for (unsigned int row = 0; row < n_block_rows(); ++row)
      for (unsigned int column = 0; column < n_block_cols(); ++column)
        block(row, column).compress();
  }



  void
  BlockSparseMatrix::compress(const VectorOperation::values operation)
  {
    BaseClass::compress(operation);
  }



  void
  BlockSparseMatrix::vmult(BlockVector &dst, const BlockVector &src) const
  {
    Assert(&dst != &src, ExcSourceEqualsDestination());
    if (n_block_cols() == 0)
      {
        AssertDimension(dst.n_blocks(), n_block_rows());
        AssertDimension(src.n_blocks(), 0);
        dst = 0.;
      }
    else
      BaseClass::vmult_block_block(dst, src);
  }



  void
  BlockSparseMatrix::vmult_add(BlockVector &dst, const BlockVector &src) const
  {
    Assert(&dst != &src, ExcSourceEqualsDestination());
    BaseClass::vmult_add(dst, src);
  }



  void
  BlockSparseMatrix::Tvmult(BlockVector &dst, const BlockVector &src) const
  {
    Assert(&dst != &src, ExcSourceEqualsDestination());
    BaseClass::Tvmult_block_block(dst, src);
  }



  void
  BlockSparseMatrix::Tvmult_add(BlockVector &dst, const BlockVector &src) const
  {
    Assert(&dst != &src, ExcSourceEqualsDestination());
    BaseClass::Tvmult_add(dst, src);
  }



  std::vector<IndexSet>
  BlockSparseMatrix::locally_owned_domain_indices() const
  {
    Assert(n_block_cols() == 0 || n_block_rows() > 0,
           ExcMessage("Column ownership requires at least one block row."));
    std::vector<IndexSet> owned(n_block_cols());
    for (unsigned int column = 0; column < n_block_cols(); ++column)
      owned[column] = block(0, column).locally_owned_domain_indices();
    return owned;
  }



  std::vector<IndexSet>
  BlockSparseMatrix::locally_owned_range_indices() const
  {
    Assert(n_block_rows() == 0 || n_block_cols() > 0,
           ExcMessage("Row ownership requires at least one block column."));
    std::vector<IndexSet> owned(n_block_rows());
    for (unsigned int row = 0; row < n_block_rows(); ++row)
      owned[row] = block(row, 0).locally_owned_range_indices();
    return owned;
  }



  std::uint64_t
  BlockSparseMatrix::n_nonzero_elements() const
  {
    std::uint64_t entries = 0;
    for (unsigned int row = 0; row < n_block_rows(); ++row)
      for (unsigned int column = 0; column < n_block_cols(); ++column)
        entries += block(row, column).n_nonzero_elements();
    return entries;
  }



  MPI_Comm
  BlockSparseMatrix::get_mpi_communicator() const
  {
    return n_block_rows() > 0 && n_block_cols() > 0 ?
             block(0, 0).get_mpi_communicator() :
             MPI_COMM_SELF;
  }
} // namespace PSCToolkitWrappers

DEAL_II_NAMESPACE_CLOSE

#endif // DEAL_II_WITH_PSBLAS
