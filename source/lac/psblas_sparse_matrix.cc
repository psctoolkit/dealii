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

#include <deal.II/base/index_set.h>
#include <deal.II/base/logstream.h>

#include <cstddef>
#include <cstdlib>

#ifdef DEAL_II_WITH_PSBLAS
#  include <deal.II/lac/psblas_sparse_matrix.h>


DEAL_II_NAMESPACE_OPEN

namespace PSCToolkit
{


  SparseMatrix::SparseMatrix()
  {
    psblas_sparse_matrix = nullptr;
    psblas_descriptor.reset();
    psb_c_set_index_base(0); // Set index base to 0
  }



  SparseMatrix::SparseMatrix(const SparsityPattern &psblas_sparsity_pattern,
                             const MPI_Comm         communicator)
  {
    Assert((psblas_sparse_matrix == nullptr &&
            psblas_descriptor.get() == nullptr),
           ExcMessage(
             "PSBLAS sparse matrix or descriptor must not be initialized."));

    Assert(psblas_sparsity_pattern.psblas_descriptor.get() != nullptr,
           ExcMessage("The given SparsityPattern is not valid."));
    psb_c_set_index_base(0); // Set index base to 0

    this->communicator = communicator;
    Assert(communicator != MPI_COMM_NULL,
           ExcMessage("MPI_COMM_NULL passed to SparseMatrix::reinit()."));
    // Convert MPI_Comm to Fortran-style communicator
    MPI_Fint f_comm = MPI_Comm_c2f(communicator);
    psblas_context  = psb_c_new_ctxt();
    psb_c_init_from_fint(psblas_context, f_comm);

    psblas_descriptor = psblas_sparsity_pattern.psblas_descriptor;

    // Create a new PSBLAS sparse matrix
    psblas_sparse_matrix = psb_c_new_dspmat();

    // Initialize the sparse matrix with the descriptor
    int err =
      psb_c_dspall_remote(psblas_sparse_matrix, psblas_descriptor.get());
    Assert(err == 0, ExcMessage("Error initializing PSBLAS sparse matrix."));
  }



  SparseMatrix::~SparseMatrix()
  {
    Assert((psblas_sparse_matrix != nullptr &&
            psblas_descriptor.get() != nullptr),
           ExcMessage("PSBLAS sparse matrix or descriptor is null."));

    // We first clear the sparse matrix
    int err = psb_c_dspfree(psblas_sparse_matrix, psblas_descriptor.get());
    Assert(err == 0, ExcMessage("Error freeing PSBLAS sparse matrix."));

    // ... and then the descriptor
    err = psb_c_cdfree(psblas_descriptor.get());
    Assert(err == 0, ExcMessage("Error freeing PSBLAS descriptor."));
  }



  void
  SparseMatrix::reinit(const IndexSet &index_set, const MPI_Comm comm)
  {
    Assert((psblas_sparse_matrix == nullptr &&
            psblas_descriptor.get() == nullptr),
           ExcMessage(
             "PSBLAS sparse matrix or descriptor must not be initialized."));

    // Create the PSBLAS context from the MPI communicator. First, I convert the
    // MPI communicator to a Fortran-style communicator and initialize the
    // PSBLAS context from it.
    Assert(comm != MPI_COMM_NULL,
           ExcMessage("MPI_COMM_NULL passed to SparseMatrix::reinit()."));
    communicator = comm;
    // Convert MPI_Comm to Fortran-style communicator
    MPI_Fint f_comm = MPI_Comm_c2f(communicator);
    psblas_context  = psb_c_new_ctxt();
    psb_c_init_from_fint(psblas_context, f_comm);
    // Create a new PSBLAS descriptor
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

    // Create a new PSBLAS sparse matrix
    psblas_sparse_matrix = psb_c_new_dspmat();

    // Initialize the sparse matrix with the descriptor
    int err =
      psb_c_dspall_remote(psblas_sparse_matrix, psblas_descriptor.get());
    Assert(err == 0, ExcMessage("Error initializing PSBLAS sparse matrix."));
  }



  SparseMatrix::size_type
  SparseMatrix::local_size() const
  {
    return psb_c_cd_get_local_rows(psblas_descriptor.get());
  }



  SparseMatrix::size_type
  SparseMatrix::m() const
  {
    return psb_c_cd_get_global_rows(psblas_descriptor.get());
  }



  SparseMatrix::size_type
  SparseMatrix::n() const
  {
    // TODO: function psb_c_cd_get_global_cols not exposed from PSBLAS
    return psb_c_cd_get_global_rows(psblas_descriptor.get());
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
    // Insert a value into the sparse matrix
    psb_l_t irw = i;
    psb_l_t icl = j;
    psb_d_t val = value;

    int err = psb_c_dspins(
      1, &irw, &icl, &val, psblas_sparse_matrix, psblas_descriptor.get());
    Assert(err == 0, ExcMessage("Failed insertion into PSBLAS sparse matrix."));
  }



  SparseMatrix::value_type
  SparseMatrix::el(const SparseMatrix::size_type,
                   const SparseMatrix::size_type) const
  {
    // Not implemented yet from PSBLAS.
    AssertThrow(false, ExcNotImplemented());
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
    // Get the number of local dofs
    const unsigned int dofs_per_cell = indices.size();
    psb_i_t nz = dofs_per_cell * dofs_per_cell; // Number of non-zero entries

    // Allocate memory for row and column indices and values
    psb_l_t *irw = (psb_l_t *)malloc(nz * sizeof(psb_l_t));
    psb_l_t *icl = (psb_l_t *)malloc(nz * sizeof(psb_l_t));
    psb_d_t *val = (psb_d_t *)malloc(nz * sizeof(psb_d_t));

    // Fill the arrays with local indices and values
    for (unsigned int i = 0; i < dofs_per_cell; ++i)
      {
        for (unsigned int j = 0; j < dofs_per_cell; ++j)
          {
            irw[i * dofs_per_cell + j] = indices[i];
            icl[i * dofs_per_cell + j] = indices[j];
            val[i * dofs_per_cell + j] = matrix(i, j);
          }
      }

    // Insert the values into the sparse matrix
    int err = psb_c_dspins(
      nz, irw, icl, val, psblas_sparse_matrix, psblas_descriptor.get());

    // Free allocated memory
    free(irw);
    free(icl);
    free(val);
    Assert(err == 0, ExcMessage("Failed insertion into PSBLAS sparse matrix."));
  }


  void
  SparseMatrix::add(const SparseMatrix::size_type  i,
                    const SparseMatrix::size_type  j,
                    const SparseMatrix::value_type value)
  {
    psb_l_t irw = i;
    psb_l_t icl = j;

    int info = psb_c_dspins(1 /*nz*/,
                            &irw,
                            &icl,
                            &value,
                            psblas_sparse_matrix,
                            psblas_descriptor.get());
    Assert(info == 0,
           ExcMessage("Error inserting values into PSBLAS sparse matrix."));
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
           ExcMessage("Column indices size mismatch."));
    // Call the function below taking a raw pointer from the vector
    add(row, ncols, col_indices.data(), values, false, false);
    ExcMessage("Error inserting values into PSBLAS sparse matrix.");
  }



  void
  SparseMatrix::add(const SparseMatrix::size_type   row,
                    const SparseMatrix::size_type   ncols,
                    const SparseMatrix::size_type  *col_indices,
                    const SparseMatrix::value_type *values,
                    const bool,
                    const bool)
  {
    psb_l_t *irw = (psb_l_t *)malloc(ncols * sizeof(psb_l_t));
    psb_l_t *icl = (psb_l_t *)malloc(ncols * sizeof(psb_l_t));
    for (SparseMatrix::size_type i = 0; i < ncols; ++i)
      {
        irw[i] = row;
        icl[i] = col_indices[i];
      }

    int info = psb_c_dspins(ncols /*nz*/,
                            irw,
                            icl,
                            values,
                            psblas_sparse_matrix,
                            psblas_descriptor.get());
    Assert(info == 0,
           ExcMessage("Error inserting values into PSBLAS sparse matrix."));
    free(irw);
    free(icl);
  }



  void
  SparseMatrix::compress()
  {
    // Finalize descriptor...
    int err = psb_c_cdasb(psblas_descriptor.get());
    Assert(err == 0, ExcMessage("Error while compressing the matrix."));

    // Check if the sparse matrix is not already assembled
    if (!psb_c_dis_matasb(psblas_sparse_matrix, psblas_descriptor.get()))
      {
        // ... and the sparse matrix
        err = psb_c_dspasb(psblas_sparse_matrix, psblas_descriptor.get());
        Assert(err == 0,
               ExcMessage("Error while assembling the PSBLAS sparse matrix."));
      }
  }



  void
  SparseMatrix::vmult(Vector &dst, const Vector &src) const
  {
    Assert(psblas_sparse_matrix != nullptr,
           ExcMessage("PSBLAS matrix has not been initialized."));
    Assert(src.psblas_vector != nullptr,
           ExcMessage("Source PSBLAS vector has not been initialized."));
    Assert(dst.psblas_vector != nullptr,
           ExcMessage("Destination PSBLAS vector has not been initialized."));

    int err = psb_c_dspmm(1.0, // alpha
                          psblas_sparse_matrix,
                          src.psblas_vector,
                          0.0, // beta
                          dst.psblas_vector,
                          psblas_descriptor.get());
    Assert(err == 0,
           ExcMessage("Error in PSBLAS matrix-vector multiplication."));
  }

  void
  SparseMatrix::Tvmult(Vector &dst, const Vector &src) const
  {
    Assert(psblas_sparse_matrix != nullptr,
           ExcMessage("PSBLAS matrix has not been initialized."));
    Assert(src.psblas_vector != nullptr,
           ExcMessage("Source PSBLAS vector has not been initialized."));
    Assert(dst.psblas_vector != nullptr,
           ExcMessage("Destination PSBLAS vector has not been initialized."));

    char transpose = 'T';
    int  err       = psb_c_dspmm_opt(1.0, // alpha
                              psblas_sparse_matrix,
                              src.psblas_vector,
                              0.0, // beta
                              dst.psblas_vector,
                              psblas_descriptor.get(),
                              &transpose,
                              true);
    Assert(err == 0,
           ExcMessage(
             "Error in PSBLAS transposed matrix-vector multiplication."));
  }

} // namespace PSCToolkit

DEAL_II_NAMESPACE_CLOSE
#endif
