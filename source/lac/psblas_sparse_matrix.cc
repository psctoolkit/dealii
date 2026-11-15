// -----------------------------------------------------------------------------
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception OR LGPL-2.1-or-later
// Copyright (C) 2019 - 2026 by the deal.II authors
//
// This file is part of the deal.II library.
//
// Detailed license information governing the source code and contributions
// can be found in LICENSE.md and CONTRIBUTING.md at the top level directory.
//
// -----------------------------------------------------------------------------

#include "deal.II/base/exception_macros.h"
#include "deal.II/base/exceptions.h"
#include "deal.II/base/index_set.h"
#include "deal.II/base/mpi.h"
#include "deal.II/base/tensor.h"
#include "deal.II/base/types.h"

#include "deal.II/lac/dynamic_sparsity_pattern.h"
#include "deal.II/lac/exceptions.h"
#include "deal.II/lac/psblas_common.h"
#include <deal.II/lac/psblas_sparse_matrix.h>

#include <boost/serialization/utility.hpp>

#include <psb_types.h>

#include <algorithm>
#include <cstdlib>
#include <utility>

#ifdef DEAL_II_WITH_PSBLAS

DEAL_II_NAMESPACE_OPEN

namespace PSCToolkitWrappers
{

  SparseMatrix::SparseMatrix()
    : communicator(MPI_COMM_NULL)
    , psblas_sparse_matrix(nullptr)
    , psblas_descriptor(nullptr)
    , psblas_context(nullptr)
    , state(internal::State::Default)
  {}

  SparseMatrix::SparseMatrix(const SparsityPattern &psblas_sparsity_pattern,
                             const MPI_Comm         communicator)
    : SparseMatrix()
  {
    reinit(psblas_sparsity_pattern, communicator);
  }



  SparseMatrix::~SparseMatrix()
  {
    clear();
  }



  void
  SparseMatrix::clear()
  {
    if (psblas_sparse_matrix != nullptr)
      {
        if (psblas_descriptor != nullptr)
          {
            const int err =
              psb_c_dspfree(psblas_sparse_matrix, psblas_descriptor.get());
            Assert(err == 0, ExcCallingPSBLASFunction(err, "psb_c_dspfree"));
          }
        std::free(psblas_sparse_matrix);
        psblas_sparse_matrix = nullptr;
      }
    psblas_column_descriptor.reset();
    psblas_descriptor.reset();
    owned_context.reset();
    psblas_context   = nullptr;
    communicator     = MPI_COMM_NULL;
    row_partition    = IndexSet();
    column_partition = IndexSet();
    pending_entries.clear();
    assembly_operation = 0;
    state              = internal::State::Default;
  }



  void
  SparseMatrix::copy_from(const SparseMatrix &other)
  {
    if (this == &other)
      return;

    DEAL_II_NOT_IMPLEMENTED();
  }


  void
  SparseMatrix::reinit(const IndexSet &index_set, const MPI_Comm comm)
  {
    Assert(index_set.n_elements() > 0,
           ExcMessage("An empty IndexSet has been given."));

    Assert(comm != MPI_COMM_NULL,
           ExcMessage("MPI_COMM_NULL passed to SparseMatrix::reinit()."));
    clear();
    communicator   = comm;
    psblas_context = InitFinalize::get_psblas_context();

    // Create a new PSBLAS descriptor
    psblas_descriptor.reset(psb_c_new_descriptor(),
                            PSCToolkitWrappers::internal::DescriptorDeleter());

    // Use get_index_vector() from IndexSet to get the indexes
    const std::vector<types::global_dof_index> &indexes =
      index_set.get_index_vector();

    psb_c_i_t number_of_local_indexes = indexes.size(); // Number of local indexes

    // Copy the indexes into an array called vl
    std::vector<psb_c_l_t> vl(std::max<std::size_t>(1, indexes.size()));
    for (psb_c_i_t i = 0; i < number_of_local_indexes; ++i)
      {
        const auto psblas_index = static_cast<psb_c_l_t>(indexes[i]);
        AssertIntegerConversion(psblas_index, indexes[i]);
        vl[i] = psblas_index;
      }

    // Insert the indexes into the descriptor
    psb_c_cdall_vl(number_of_local_indexes,
                   vl.data(),
                   *InitFinalize::get_psblas_context(),
                   psblas_descriptor.get());

    // Create a new PSBLAS sparse matrix
    psblas_sparse_matrix = psb_c_new_dspmat();

    // Initialize the sparse matrix with the descriptor
    int err =
      psb_c_dspall_remote(psblas_sparse_matrix, psblas_descriptor.get());
    Assert(err == 0, ExcAllocationPSBLASMatrix(err));

    state = internal::State::Build;
  }



  void
  SparseMatrix::reinit(const SparsityPattern &psblas_sparsity_pattern,
                       const MPI_Comm         communicator)
  {
    Assert(psblas_sparsity_pattern.psblas_descriptor.get() != nullptr,
           ExcMessage("The given SparsityPattern is not valid."));

    Assert(communicator != MPI_COMM_NULL,
           ExcMessage("MPI_COMM_NULL passed to SparseMatrix::reinit()."));

    clear();
    this->communicator = communicator;
    psblas_context     = InitFinalize::get_psblas_context();

    psblas_descriptor = psblas_sparsity_pattern.psblas_descriptor;

    // Create a new PSBLAS sparse matrix
    psblas_sparse_matrix = psb_c_new_dspmat();

    // Initialize the sparse matrix with the descriptor
    int err =
      psb_c_dspall_remote(psblas_sparse_matrix, psblas_descriptor.get());
    Assert(err == 0, ExcAllocationPSBLASMatrix(err));

    state = internal::State::Build;
  }



  void
  SparseMatrix::reinit(const IndexSet               &local_rows,
                       const DynamicSparsityPattern &sparsity_pattern,
                       const MPI_Comm                communicator)
  {
    Assert(sparsity_pattern.n_rows() == sparsity_pattern.n_cols(),
           ExcNotQuadratic());

    // Check dimensions match IndexSet
    Assert(sparsity_pattern.n_rows() == local_rows.size(),
           ExcMessage(
             "SparsityPattern and IndexSet have different number of rows"));

    Assert(local_rows.is_ascending_and_one_to_one(communicator),
           ExcNotImplemented());

    clear();
    this->communicator = communicator;
    psblas_context     = InitFinalize::get_psblas_context();


    if constexpr (running_in_debug_mode())
      {
        types::global_dof_index row_owners =
          Utilities::MPI::sum(local_rows.n_elements(), communicator);
        Assert(row_owners == sparsity_pattern.n_rows(),
               ExcMessage(
                 std::string(
                   "Each row has to be owned by exactly one owner (n_rows()=") +
                 std::to_string(sparsity_pattern.n_rows()) +
                 " but sum(local_rows.n_elements())=" +
                 std::to_string(row_owners) + ")"));
      }

    // Set up the PSBLAS descriptor from the local IndexSet
    psblas_descriptor.reset(psb_c_new_descriptor(),
                            PSCToolkitWrappers::internal::DescriptorDeleter());

    {
      const std::vector<types::global_dof_index> indexes =
        local_rows.get_index_vector();
      const psb_c_i_t n_local = static_cast<psb_c_i_t>(indexes.size());

      std::vector<psb_c_l_t> vl(std::max<std::size_t>(1, indexes.size()));
      for (psb_c_i_t i = 0; i < n_local; ++i)
        {
          const auto idx = static_cast<psb_c_l_t>(indexes[i]);
          AssertIntegerConversion(idx, indexes[i]);
          vl[i] = idx;
        }

      psb_c_cdall_vl(n_local,
                     vl.data(),
                     *InitFinalize::get_psblas_context(),
                     psblas_descriptor.get());
    }

    // Create and initialize the sparse matrix
    psblas_sparse_matrix = psb_c_new_dspmat();

    int err;

    if (local_rows.n_elements() > 0)
      {
        const psb_c_l_t local_row_start =
          static_cast<psb_c_l_t>(local_rows.nth_index_in_set(0));
        const psb_c_l_t local_row_end =
          local_row_start + static_cast<psb_c_l_t>(local_rows.n_elements());

        // Insert entries from the sparsity pattern row by row
        for (psb_c_l_t i = local_row_start; i < local_row_end; ++i)
          {
            const auto row_length =
              static_cast<psb_c_i_t>(sparsity_pattern.row_length(i));
            if (row_length == 0)
              continue;

            std::vector<psb_c_l_t> irw(row_length, i);
            std::vector<psb_c_l_t> icl(row_length);

            psb_c_i_t k = 0;
            for (typename DynamicSparsityPattern::iterator p =
                   sparsity_pattern.begin(i);
                 p != sparsity_pattern.end(i);
                 ++p, ++k)
              {
                const auto col = static_cast<psb_c_l_t>(p->column());
                AssertIntegerConversion(col, p->column());
                icl[k] = col;
              }

            err = psb_c_cdins(row_length,
                              irw.data(),
                              icl.data(),
                              psblas_descriptor.get());
            Assert(err == 0, ExcInsertionInPSBLASMatrix(err));
          }

        // Initialize the sparse matrix with the descriptor
        err =
          psb_c_dspall_remote(psblas_sparse_matrix, psblas_descriptor.get());
        Assert(err == 0, ExcAllocationPSBLASMatrix(err));
      }
    else
      {
        // empty local partition: just initialize
        err =
          psb_c_dspall_remote(psblas_sparse_matrix, psblas_descriptor.get());
        Assert(err == 0, ExcAllocationPSBLASMatrix(err));
      }

    state = internal::State::Build;
  }



  void
  SparseMatrix::reinit(const IndexSet &local_rows,
                       const IndexSet &local_columns,
                       const MPI_Comm  comm)
  {
    clear();
    communicator     = comm;
    row_partition    = local_rows;
    column_partition = local_columns;
    owned_context    = std::make_shared<internal::Context>(comm);
    psblas_context   = owned_context->get_psblas_context();

    const auto make_descriptor = [this](const IndexSet &partition) {
      // Keep a communicator-specific context alive even if a preconditioner
      // retains the descriptor after this matrix is destroyed.
      const auto                        context = owned_context;
      std::shared_ptr<psb_c_descriptor> descriptor(
        psb_c_new_descriptor(),
        [context](psb_c_descriptor *p) { internal::DescriptorDeleter()(p); });
      const auto indices = partition.get_index_vector();
      const auto count   = static_cast<psb_c_i_t>(indices.size());
      AssertIntegerConversion(count, indices.size());
      std::vector<psb_c_l_t> native_indices(
        std::max<std::size_t>(1, indices.size()));
      for (std::size_t i = 0; i < indices.size(); ++i)
        {
          native_indices[i] = static_cast<psb_c_l_t>(indices[i]);
          AssertIntegerConversion(native_indices[i], indices[i]);
        }
      const int err = psb_c_cdall_vl(count,
                                     native_indices.data(),
                                     *psblas_context,
                                     descriptor.get());
      Assert(err == 0, ExcInitializePSBLASDescriptor(err));
      return descriptor;
    };

    psblas_descriptor = make_descriptor(local_rows);
    const bool different_partitions =
      Utilities::MPI::logical_or(local_rows.size() != local_columns.size() ||
                                   local_rows != local_columns,
                                 communicator);
    psblas_column_descriptor =
      different_partitions ? make_descriptor(local_columns) : psblas_descriptor;
    psblas_sparse_matrix = psb_c_new_dspmat();
    const int err =
      different_partitions ?
        psb_c_dspall_2desc(psblas_sparse_matrix,
                           psblas_descriptor.get(),
                           psblas_column_descriptor.get()) :
        psb_c_dspall_remote(psblas_sparse_matrix, psblas_descriptor.get());
    Assert(err == 0, ExcAllocationPSBLASMatrix(err));
    state = internal::State::Build;
  }



  void
  SparseMatrix::reinit(const IndexSet               &local_rows,
                       const IndexSet               &local_columns,
                       const DynamicSparsityPattern &sparsity_pattern,
                       const MPI_Comm                comm)
  {
    AssertDimension(local_rows.size(), sparsity_pattern.n_rows());
    AssertDimension(local_columns.size(), sparsity_pattern.n_cols());
    reinit(local_rows, local_columns, comm);

    // Only the locally owned structure is needed. Contributions to nonlocal
    // rows are exchanged during compress(), before native insertion.
    for (const auto row : local_rows)
      {
        const auto count =
          static_cast<psb_c_i_t>(sparsity_pattern.row_length(row));
        AssertIntegerConversion(count, sparsity_pattern.row_length(row));
        if (count == 0)
          continue;
        const auto native_row = static_cast<psb_c_l_t>(row);
        AssertIntegerConversion(native_row, row);
        std::vector<psb_c_l_t> rows(count, native_row), columns(count);
        std::vector<psb_c_d_t> values(count, 0.0);
        psb_c_i_t              k = 0;
        for (auto entry = sparsity_pattern.begin(row);
             entry != sparsity_pattern.end(row);
             ++entry, ++k)
          {
            columns[k] = static_cast<psb_c_l_t>(entry->column());
            AssertIntegerConversion(columns[k], entry->column());
          }
        const int err = psblas_column_descriptor != psblas_descriptor ?
                          psb_c_dspins_2desc(count,
                                             rows.data(),
                                             columns.data(),
                                             values.data(),
                                             psblas_sparse_matrix,
                                             psblas_descriptor.get(),
                                             psblas_column_descriptor.get()) :
                          psb_c_dspins(count,
                                       rows.data(),
                                       columns.data(),
                                       values.data(),
                                       psblas_sparse_matrix,
                                       psblas_descriptor.get());
        Assert(err == 0, ExcInsertionInPSBLASMatrix(err));
      }
  }



  IndexSet
  SparseMatrix::locally_owned_range_indices() const
  {
    if (psblas_column_descriptor != nullptr)
      return row_partition;
    IndexSet indices(m());
    if (local_size() != 0)
      {
        std::vector<psb_c_l_t> native_indices(local_size());
        const int err = psb_c_cd_get_global_indices(native_indices.data(),
                                                    native_indices.size(),
                                                    true,
                                                    psblas_descriptor.get());
        Assert(err == 0,
               ExcCallingPSBLASFunction(err, "psb_c_cd_get_global_indices"));
        indices.add_indices(native_indices.begin(), native_indices.end());
      }
    indices.compress();
    return indices;
  }



  IndexSet
  SparseMatrix::locally_owned_domain_indices() const
  {
    return psblas_column_descriptor != nullptr ? column_partition :
                                                 locally_owned_range_indices();
  }



  psb_c_descriptor *
  SparseMatrix::get_psblas_column_descriptor() const
  {
    return psblas_column_descriptor != nullptr ?
             psblas_column_descriptor.get() :
             psblas_descriptor.get();
  }



  void
  SparseMatrix::prepare_add()
  {
    if (psblas_column_descriptor != nullptr)
      {
        Assert(state == internal::State::Build,
               ExcMessage(
                 "Reinitialize the matrix before assembling a new operator."));
        Assert(assembly_operation != 1, ExcWrongMode(2, assembly_operation));
        assembly_operation = 2;
      }
  }



  void
  SparseMatrix::prepare_set()
  {
    if (psblas_column_descriptor != nullptr)
      {
        Assert(state == internal::State::Build,
               ExcMessage(
                 "Reinitialize the matrix before assembling a new operator."));
        Assert(assembly_operation != 2, ExcWrongMode(1, assembly_operation));
        assembly_operation = 1;
      }
  }



  void
  SparseMatrix::set(const size_type   row,
                    const size_type   n_columns,
                    const size_type  *column_indices,
                    const value_type *values,
                    const bool        elide_zero_values)
  {
    for (size_type j = 0; j < n_columns; ++j)
      if (!elide_zero_values || values[j] != 0.0)
        set(row, column_indices[j], values[j]);
  }



  void
  SparseMatrix::add(const size_type                row,
                    const std::vector<size_type>  &column_indices,
                    const std::vector<value_type> &values,
                    const bool                     elide_zero_values)
  {
    AssertDimension(column_indices.size(), values.size());
    add(row,
        column_indices.size(),
        column_indices.data(),
        values.data(),
        elide_zero_values);
  }



  void
  SparseMatrix::compress(const VectorOperation::values operation)
  {
    if (psblas_column_descriptor != nullptr)
      {
        Assert(operation == VectorOperation::add ||
                 operation == VectorOperation::insert,
               ExcNotImplemented());
        const unsigned int mode = operation == VectorOperation::insert ? 1 : 2;
        Assert(assembly_operation == 0 || assembly_operation == mode,
               ExcWrongMode(mode, assembly_operation));
      }
    compress();
  }



  void
  SparseMatrix::compress_partitions()
  {
    if (state == internal::State::Assembled)
      return;

    const bool inserting =
      Utilities::MPI::logical_or(assembly_operation == 1, communicator);
    const bool adding =
      Utilities::MPI::logical_or(assembly_operation == 2, communicator);
    AssertThrow(
      !(inserting && adding),
      ExcMessage("All processes must use the same matrix assembly operation."));

    IndexSet requested_rows(row_partition.size());
    for (const auto &entry : pending_entries)
      requested_rows.add_index(entry.first.first);
    requested_rows.compress();
    const auto owners = Utilities::MPI::compute_index_owner(row_partition,
                                                            requested_rows,
                                                            communicator);
    using Entry       = std::pair<std::pair<size_type, size_type>, value_type>;
    std::map<unsigned int, std::vector<Entry>> outgoing;
    for (const auto &entry : pending_entries)
      outgoing[owners[requested_rows.index_within_set(entry.first.first)]]
        .push_back(entry);
    const auto incoming = Utilities::MPI::some_to_some(communicator, outgoing);

    std::map<std::pair<size_type, size_type>, value_type> entries;
    bool conflicting_values = false;
    for (const auto &sender : incoming)
      for (const auto &entry : sender.second)
        if (inserting)
          {
            const auto result = entries.emplace(entry);
            if (!result.second && result.first->second != entry.second)
              conflicting_values = true;
          }
        else
          entries[entry.first] += entry.second;
    AssertThrow(
      !Utilities::MPI::logical_or(conflicting_values, communicator),
      ExcMessage(
        "Different processes set the same matrix entry to different values."));

    // Resolve duplicates before insertion: native sparse assembly sums
    // duplicate entries, while set() must retain the last value from a sender.
    const auto count = static_cast<psb_c_i_t>(entries.size());
    AssertIntegerConversion(count, entries.size());
    if (count != 0)
      {
        std::vector<psb_c_l_t> rows(count), columns(count);
        std::vector<psb_c_d_t> values(count);
        psb_c_i_t              k = 0;
        for (const auto &entry : entries)
          {
            rows[k]    = static_cast<psb_c_l_t>(entry.first.first);
            columns[k] = static_cast<psb_c_l_t>(entry.first.second);
            AssertIntegerConversion(rows[k], entry.first.first);
            AssertIntegerConversion(columns[k], entry.first.second);
            values[k++] = entry.second;
          }
        const int err = psblas_column_descriptor != psblas_descriptor ?
                          psb_c_dspins_2desc(count,
                                             rows.data(),
                                             columns.data(),
                                             values.data(),
                                             psblas_sparse_matrix,
                                             psblas_descriptor.get(),
                                             psblas_column_descriptor.get()) :
                          psb_c_dspins(count,
                                       rows.data(),
                                       columns.data(),
                                       values.data(),
                                       psblas_sparse_matrix,
                                       psblas_descriptor.get());
        Assert(err == 0, ExcInsertionInPSBLASMatrix(err));
      }

    if (!psb_c_cd_is_asb(psblas_descriptor.get()))
      {
        const int err = psb_c_cdasb(psblas_descriptor.get());
        Assert(err == 0, ExcAssemblePSBLASDescriptor(err));
      }
    if (!psb_c_cd_is_asb(psblas_column_descriptor.get()))
      {
        const int err = psb_c_cdasb(psblas_column_descriptor.get());
        Assert(err == 0, ExcAssemblePSBLASDescriptor(err));
      }
    const int err =
      psblas_column_descriptor != psblas_descriptor ?
        psb_c_dspasb_2desc(psblas_sparse_matrix,
                           psblas_descriptor.get(),
                           psblas_column_descriptor.get()) :
        psb_c_dspasb(psblas_sparse_matrix, psblas_descriptor.get());
    Assert(err == 0, ExcAssemblePSBLASMatrix(err));
    pending_entries.clear();
    assembly_operation = 0;
    state              = internal::State::Assembled;
  }



  SparseMatrix::size_type
  SparseMatrix::local_size() const
  {
    return psblas_descriptor == nullptr ?
             0 :
             psb_c_cd_get_local_rows(psblas_descriptor.get());
  }


  std::pair<SparseMatrix::size_type, SparseMatrix::size_type>
  SparseMatrix::local_range() const
  {
    if (local_size() == 0)
      return {0, 0};

    std::vector<psb_c_l_t> local_indices(local_size());
    int                  err = psb_c_cd_get_global_indices(local_indices.data(),
                                          local_size(),
                                          true /*only owned indices*/,
                                          psblas_descriptor.get());
    Assert(err == 0,
           ExcCallingPSBLASFunction(err, "psb_c_cd_get_global_indices"));

    return {local_indices[0], local_indices.back() + 1};
  }



  bool
  SparseMatrix::in_local_range(const size_type index) const
  {
    if (psblas_column_descriptor != nullptr)
      return row_partition.is_element(index);
    std::pair<size_type, size_type> local_range = this->local_range();
    return index >= local_range.first && index < local_range.second;
  }



  SparseMatrix::size_type
  SparseMatrix::m() const
  {
    return psblas_descriptor == nullptr ?
             0 :
             psb_c_cd_get_global_rows(psblas_descriptor.get());
  }



  SparseMatrix::size_type
  SparseMatrix::n() const
  {
    if (psblas_column_descriptor != nullptr)
      return column_partition.size();
    return psblas_descriptor == nullptr ?
             0 :
             psb_c_cd_get_global_cols(psblas_descriptor.get());
  }



  SparseMatrix::size_type
  SparseMatrix::n_nonzero_elements() const
  {
    return psb_c_dnnz(psblas_sparse_matrix, psblas_descriptor.get());
  }



  void
  SparseMatrix::set(const SparseMatrix::size_type  i,
                    const SparseMatrix::size_type  j,
                    const SparseMatrix::value_type value)
  {
    if (psblas_column_descriptor != nullptr)
      {
        AssertIndexRange(i, m());
        AssertIndexRange(j, n());
        AssertIsFinite(value);
        prepare_set();
        pending_entries[{i, j}] = value;
        return;
      }

    // Insert a value into the sparse matrix
    psb_c_l_t irw = i;
    psb_c_l_t icl = j;
    psb_c_d_t val = value;

    int err = psb_c_dspins(
      1, &irw, &icl, &val, psblas_sparse_matrix, psblas_descriptor.get());
    Assert(err == 0, ExcInsertionInPSBLASMatrix(err));
  }



  SparseMatrix::value_type
  SparseMatrix::el(const SparseMatrix::size_type i,
                   const SparseMatrix::size_type j) const
  {
    const auto psblas_index_i = static_cast<psb_c_l_t>(i);
    AssertIntegerConversion(psblas_index_i, i);
    const auto psblas_index_j = static_cast<psb_c_l_t>(j);
    AssertIntegerConversion(psblas_index_j, j);

    if (psblas_column_descriptor != nullptr &&
        psblas_column_descriptor != psblas_descriptor)
      {
        Assert(row_partition.is_element(i),
               ExcMessage("The requested row is not locally owned."));
        psb_c_i_t count = 0;
        int     err   = psb_c_dspget_coo(psblas_sparse_matrix,
                                   psblas_descriptor.get(),
                                   psblas_column_descriptor.get(),
                                   0,
                                   &count,
                                   nullptr,
                                   nullptr,
                                   nullptr);
        Assert(err == 0, ExcCallingPSBLASFunction(err, "psb_c_dspget_coo"));
        if (count == 0)
          return 0.0;
        std::vector<psb_c_l_t> rows(count), columns(count);
        std::vector<psb_c_d_t> values(count);
        err = psb_c_dspget_coo(psblas_sparse_matrix,
                               psblas_descriptor.get(),
                               psblas_column_descriptor.get(),
                               count,
                               &count,
                               rows.data(),
                               columns.data(),
                               values.data());
        Assert(err == 0, ExcCallingPSBLASFunction(err, "psb_c_dspget_coo"));
        for (psb_c_i_t k = 0; k < count; ++k)
          if (rows[k] == psblas_index_i && columns[k] == psblas_index_j)
            return values[k];
        return 0.0;
      }

    return psb_c_dmatgetelem(psblas_sparse_matrix,
                             psblas_index_i,
                             psblas_index_j,
                             psblas_descriptor.get());
  }



  SparseMatrix::value_type
  SparseMatrix::diag_element(const SparseMatrix::size_type i) const
  {
    Assert(m() == n(), ExcNotQuadratic());
    return el(i, i);
  }



  SparseMatrix::value_type
  SparseMatrix::operator()(const SparseMatrix::size_type i,
                           const SparseMatrix::size_type j) const
  {
    if (psblas_column_descriptor == nullptr)
      Assert(m() == n(), ExcNotQuadratic());
    return el(i, j);
  }



  psb_c_dspmat *
  SparseMatrix::get_psblas_matrix() const
  {
    return psblas_sparse_matrix;
  }



  psb_c_descriptor *
  SparseMatrix::get_psblas_descriptor() const
  {
    return psblas_descriptor.get();
  }



  MPI_Comm
  SparseMatrix::get_mpi_communicator() const
  {
    return communicator;
  }



  void
  SparseMatrix::set(const std::vector<SparseMatrix::size_type> &indices,
                    const FullMatrix<double>                   &matrix)
  {
    Assert(psblas_sparse_matrix != nullptr,
           ExcMessage("PSBLAS matrix has not been initialized."));
    Assert(matrix.m() == indices.size(),
           ExcDimensionMismatch(matrix.m(), indices.size()));
    Assert(matrix.n() == indices.size(),
           ExcDimensionMismatch(matrix.n(), indices.size()));

    if (psblas_column_descriptor != nullptr)
      {
        for (unsigned int i = 0; i < indices.size(); ++i)
          for (unsigned int j = 0; j < indices.size(); ++j)
            set(indices[i], indices[j], matrix(i, j));
        return;
      }

    // Get the number of indices.
    const unsigned int n_indices = indices.size();
    psb_c_i_t            nz = n_indices * n_indices; // Number of non-zero entries

    // Fill the arrays with row indices, column indices, and values
    std::vector<psb_c_l_t> irw(nz);
    std::vector<psb_c_l_t> icl(nz);
    std::vector<psb_c_d_t> val(nz);
    for (unsigned int i = 0; i < n_indices; ++i)
      {
        for (unsigned int j = 0; j < n_indices; ++j)
          {
            const auto psblas_row_index = static_cast<psb_c_l_t>(indices[i]);
            const auto psblas_col_index = static_cast<psb_c_l_t>(indices[j]);
            AssertIntegerConversion(psblas_row_index, indices[i]);
            AssertIntegerConversion(psblas_col_index, indices[j]);
            irw[i * n_indices + j] = psblas_row_index;
            icl[i * n_indices + j] = psblas_col_index;
            val[i * n_indices + j] = matrix(i, j);
          }
      }

    // Insert the values into the sparse matrix
    int err = psb_c_dspins(nz,
                           irw.data(),
                           icl.data(),
                           val.data(),
                           psblas_sparse_matrix,
                           psblas_descriptor.get());

    Assert(err == 0, ExcInsertionInPSBLASMatrix(err));
  }



  void
  SparseMatrix::add(const SparseMatrix::size_type  i,
                    const SparseMatrix::size_type  j,
                    const SparseMatrix::value_type value)
  {
    AssertIsFinite(value);

    if (psblas_column_descriptor != nullptr)
      {
        AssertIndexRange(i, m());
        AssertIndexRange(j, n());
        prepare_add();
        pending_entries[{i, j}] += value;
        return;
      }

    psb_c_l_t irw  = i;
    psb_c_l_t icl  = j;
    int     info = psb_c_dspins(1 /*nz*/,
                            &irw,
                            &icl,
                            &value,
                            psblas_sparse_matrix,
                            psblas_descriptor.get());
    Assert(info == 0, ExcInsertionInPSBLASMatrix(info));
  }



  void
  SparseMatrix::add(const SparseMatrix::size_type               row,
                    const SparseMatrix::size_type               ncols,
                    const std::vector<SparseMatrix::size_type> &col_indices,
                    const SparseMatrix::value_type             *values,
                    const bool,
                    const bool)
  {
    Assert(col_indices.size() == ncols,
           ExcDimensionMismatch(col_indices.size(), ncols));
    // Call the function below taking a raw pointer from the vector
    add(row, ncols, col_indices.data(), values, false, false);
  }



  void
  SparseMatrix::add(const SparseMatrix::size_type                row,
                    const SparseMatrix::size_type                ncols,
                    const std::vector<SparseMatrix::size_type>  &col_indices,
                    const std::vector<SparseMatrix::value_type> &values,
                    const bool,
                    const bool)
  {
    Assert(col_indices.size() == ncols,
           ExcDimensionMismatch(col_indices.size(), ncols));
    Assert(values.size() == ncols, ExcDimensionMismatch(ncols, values.size()));
    // Call the function below taking a raw pointer from the vector
    add(row, ncols, col_indices.data(), values.data(), false, false);
  }



  void
  SparseMatrix::add(const SparseMatrix::size_type   row,
                    const SparseMatrix::size_type   ncols,
                    const SparseMatrix::size_type  *col_indices,
                    const SparseMatrix::value_type *values,
                    const bool,
                    const bool)
  {
    if (psblas_column_descriptor != nullptr)
      {
        for (size_type j = 0; j < ncols; ++j)
          add(row, col_indices[j], values[j]);
        return;
      }

    std::vector<psb_c_l_t> irw(ncols);
    std::vector<psb_c_l_t> icl(ncols);
    for (SparseMatrix::size_type i = 0; i < ncols; ++i)
      {
        irw[i] = static_cast<psb_c_l_t>(row);
        icl[i] = static_cast<psb_c_l_t>(col_indices[i]);
      }

    int info = psb_c_dspins(ncols /*nz*/,
                            irw.data(),
                            icl.data(),
                            values,
                            psblas_sparse_matrix,
                            psblas_descriptor.get());
    Assert(info == 0, ExcInsertionInPSBLASMatrix(info));
  }



  void
  SparseMatrix::compress()
  {
    if (psblas_column_descriptor != nullptr)
      {
        compress_partitions();
        return;
      }

    // We start by checking if the descriptor has already been assembled
    // elsewhere
    int err = -1;
    if (!psb_c_cd_is_asb(psblas_descriptor.get()))
      {
        err = psb_c_cdasb(psblas_descriptor.get());
        Assert(err == 0, ExcAssemblePSBLASDescriptor(err));
      }

    // Check if the sparse matrix is not already assembled
    if (!psb_c_dis_matasb(psblas_sparse_matrix, psblas_descriptor.get()))
      {
        // ... and the sparse matrix
        err = psb_c_dspasb(psblas_sparse_matrix, psblas_descriptor.get());
        Assert(err == 0, ExcAssemblePSBLASMatrix(err));
        state = internal::State::Assembled;
      }
  }



  void
  SparseMatrix::vmult(Vector &dst, const Vector &src) const
  {
    Assert(psblas_sparse_matrix != nullptr,
           ExcMessage("PSBLAS matrix has not been initialized."));
    Assert(src.psblas_vector != nullptr,
           ExcMessage("Source PSBLAS vector has not been initialized."));
    Assert(&src != &dst, ExcSourceEqualsDestination());
    Assert(state == internal::State::Assembled,
           ExcMessage("PSBLAS matrix has not been assembled."));

    if (psblas_column_descriptor != nullptr &&
        psblas_column_descriptor != psblas_descriptor)
      {
        const int err = psb_c_dspmm_2desc(1.0,
                                          psblas_sparse_matrix,
                                          src.psblas_vector,
                                          0.0,
                                          dst.psblas_vector,
                                          psblas_descriptor.get(),
                                          psblas_column_descriptor.get(),
                                          "N");
        Assert(err == 0, ExcMatVecPSBLAS(err));
        return;
      }

    int err = psb_c_dspmm(value_type(1.0), // alpha
                          psblas_sparse_matrix,
                          src.psblas_vector,
                          value_type(0.0), // beta
                          dst.psblas_vector,
                          psblas_descriptor.get());
    Assert(err == 0, ExcMatVecPSBLAS(err));
  }



  void
  SparseMatrix::vmult_add(Vector &dst, const Vector &src) const
  {
    Assert(psblas_sparse_matrix != nullptr,
           ExcMessage("PSBLAS matrix has not been initialized."));
    Assert(src.psblas_vector != nullptr,
           ExcMessage("Source PSBLAS vector has not been initialized."));
    Assert(&src != &dst, ExcSourceEqualsDestination());
    Assert(state == internal::State::Assembled,
           ExcMessage("PSBLAS matrix has not been assembled."));

    if (psblas_column_descriptor != nullptr &&
        psblas_column_descriptor != psblas_descriptor)
      {
        const int err = psb_c_dspmm_2desc(1.0,
                                          psblas_sparse_matrix,
                                          src.psblas_vector,
                                          1.0,
                                          dst.psblas_vector,
                                          psblas_descriptor.get(),
                                          psblas_column_descriptor.get(),
                                          "N");
        Assert(err == 0, ExcMatVecPSBLAS(err));
        return;
      }

    int err = psb_c_dspmm(value_type(1.0), // alpha
                          psblas_sparse_matrix,
                          src.psblas_vector,
                          value_type(1.0), // beta
                          dst.psblas_vector,
                          psblas_descriptor.get());
    Assert(err == 0, ExcMatVecPSBLAS(err));
  }



  void
  SparseMatrix::Tvmult(Vector &dst, const Vector &src) const
  {
    Assert(psblas_sparse_matrix != nullptr,
           ExcMessage("PSBLAS matrix has not been initialized."));
    Assert(src.psblas_vector != nullptr,
           ExcMessage("Source PSBLAS vector has not been initialized."));
    Assert(&src != &dst, ExcSourceEqualsDestination());
    Assert(state == internal::State::Assembled,
           ExcMessage("PSBLAS matrix has not been assembled."));

    if (psblas_column_descriptor != nullptr &&
        psblas_column_descriptor != psblas_descriptor)
      {
        const int err = psb_c_dspmm_2desc(1.0,
                                          psblas_sparse_matrix,
                                          src.psblas_vector,
                                          0.0,
                                          dst.psblas_vector,
                                          psblas_descriptor.get(),
                                          psblas_column_descriptor.get(),
                                          "T");
        Assert(err == 0, ExcMatVecPSBLAS(err));
        return;
      }

    char transpose = 'T';
    int  err       = psb_c_dspmm_opt(value_type(1.0), // alpha
                              psblas_sparse_matrix,
                              src.psblas_vector,
                              value_type(0.0), // beta
                              dst.psblas_vector,
                              psblas_descriptor.get(),
                              &transpose,
                              true);
    Assert(err == 0, ExcMatVecPSBLAS(err));
  }



  void
  SparseMatrix::Tvmult_add(Vector &dst, const Vector &src) const
  {
    Assert(psblas_sparse_matrix != nullptr,
           ExcMessage("PSBLAS matrix has not been initialized."));
    Assert(src.psblas_vector != nullptr,
           ExcMessage("Source PSBLAS vector has not been initialized."));
    Assert(&src != &dst, ExcSourceEqualsDestination());
    Assert(state == internal::State::Assembled,
           ExcMessage("PSBLAS matrix has not been assembled."));

    if (psblas_column_descriptor != nullptr &&
        psblas_column_descriptor != psblas_descriptor)
      {
        const int err = psb_c_dspmm_2desc(1.0,
                                          psblas_sparse_matrix,
                                          src.psblas_vector,
                                          1.0,
                                          dst.psblas_vector,
                                          psblas_descriptor.get(),
                                          psblas_column_descriptor.get(),
                                          "T");
        Assert(err == 0, ExcMatVecPSBLAS(err));
        return;
      }

    char transpose = 'T';
    int  err       = psb_c_dspmm_opt(value_type(1.0), // alpha
                              psblas_sparse_matrix,
                              src.psblas_vector,
                              value_type(1.0), // beta
                              dst.psblas_vector,
                              psblas_descriptor.get(),
                              &transpose,
                              true);
    Assert(err == 0, ExcMatVecPSBLAS(err));
  }



  SparseMatrix::value_type
  SparseMatrix::l1_norm() const
  {
    DEAL_II_NOT_IMPLEMENTED();
  }



  SparseMatrix::value_type
  SparseMatrix::linfty_norm() const
  {
    DEAL_II_NOT_IMPLEMENTED();
  }



  SparseMatrix::value_type
  SparseMatrix::frobenius_norm() const
  {
    DEAL_II_NOT_IMPLEMENTED();
  }



  SparseMatrix::value_type
  SparseMatrix::trace() const
  {
    Assert(m() == n(), ExcNotQuadratic());
    value_type local_diagonal_sum = 0;
    if (psblas_column_descriptor != nullptr)
      for (const auto row : row_partition)
        local_diagonal_sum += diag_element(row);
    else
      {
        const auto local_indices = local_range();
        for (types::global_dof_index idx = local_indices.first;
             idx < local_indices.second;
             ++idx)
          local_diagonal_sum += diag_element(idx);
      }

    return Utilities::MPI::sum(local_diagonal_sum, communicator);
  }



} // namespace PSCToolkitWrappers

DEAL_II_NAMESPACE_CLOSE
#endif // DEAL_II_WITH_PSBLAS
