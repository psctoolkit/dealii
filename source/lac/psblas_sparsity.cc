// ------------------------------------------------------------------------
//
// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (C) 2019 - 2023 by the deal.II authors
//
// This file is part of the deal.II library.
//
// Part of the source code is dual licensed under Apache-2.0 WITH
// LLVM-exception OR LGPL-2.1-or-later. Detailed license information
// governing the source code and code contributions can be found in
// LICENSE.md and CONTRIBUTING.md at the top level directory of deal.II.
//
// ------------------------------------------------------------------------


#include "deal.II/base/config.h"

#include "deal.II/base/mpi.h"
#include <deal.II/base/index_set.h>
#include <deal.II/base/logstream.h>

#include <psb_c_base.h>
#include <psb_c_dbase.h>

#include <cstddef>
#include <cstdlib>

#ifdef DEAL_II_WITH_PSBLAS
#  include <deal.II/lac/psblas_sparsity.h>

DEAL_II_NAMESPACE_OPEN

namespace PSCToolkit
{


  // SparsityPattern
  SparsityPattern::SparsityPattern()
  {
    psblas_descriptor.reset();
  }



  // SparsityPattern
  SparsityPattern::SparsityPattern(const IndexSet &index_set,
                                   const MPI_Comm  communicator)
  {
    SparsityPatternBase::resize(index_set.size(), index_set.size());

    Assert(communicator != MPI_COMM_NULL,
           ExcMessage("MPI_COMM_NULL passed to SparseMatrix::reinit()."));
    // Convert MPI_Comm to Fortran-style communicator
    MPI_Fint f_comm = MPI_Comm_c2f(communicator);
    psblas_context  = psb_c_new_ctxt();
    psb_c_init_from_fint(psblas_context, f_comm);

    psblas_descriptor.reset(psb_c_new_descriptor());


    // Use get_index_vector() from IndexSet to get the indexes
    const std::vector<types::global_dof_index> &indexes =
      index_set.get_index_vector();

    psb_i_t number_of_local_indexes = indexes.size(); // Number of local indexes
    // Copy the indexes into a psb_l_t array called vl
    psb_l_t *vl = (psb_l_t *)malloc(number_of_local_indexes * sizeof(psb_l_t));
    for (psb_i_t i = 0; i < number_of_local_indexes; ++i)
      {
        vl[i] = static_cast<psb_l_t>(indexes[i]);
      }

    // Insert the indexes into the descriptor
    psb_c_cdall_vl(number_of_local_indexes,
                   vl,
                   *psblas_context,
                   psblas_descriptor.get());

    // Free the vl array
    free(vl);
  }



  void
  SparsityPattern::add(const PSCToolkit::SparsityPattern::size_type i,
                       const PSCToolkit::SparsityPattern::size_type j)
  {
    add_entries(i, &j, &j + 1);
  }



  template <typename ForwardIterator>
  inline void
  SparsityPattern::add_entries(const PSCToolkit::SparsityPattern::size_type row,
                               ForwardIterator begin,
                               ForwardIterator end,
                               const bool      indices_are_sorted)
  {
    if (begin == end)
      return;

    (void)indices_are_sorted;
    psb_i_t  nz = static_cast<int>(end - begin);
    psb_l_t *ia = (psb_l_t *)malloc(nz * sizeof(psb_l_t));
    psb_l_t *ja = (psb_l_t *)malloc(nz * sizeof(psb_l_t));

    for (int k = 0; k < nz; ++k)
      {
        ia[k] = row;          // row index
        ja[k] = *(begin + k); // column index
      }
    int err = psb_c_cdins(nz, ia, ja, psblas_descriptor.get());
    Assert(err == 0,
           ExcMessage("Error inserting entries into PSBLAS descriptor."));
  }



  void
  SparsityPattern::add_row_entries(const size_type                  &row,
                                   const ArrayView<const size_type> &columns,
                                   const bool indices_are_sorted)
  {
    add_entries(row, columns.begin(), columns.end(), indices_are_sorted);
  }
} // namespace PSCToolkit

DEAL_II_NAMESPACE_CLOSE
#endif
