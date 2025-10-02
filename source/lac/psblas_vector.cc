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
#  include <deal.II/lac/psblas_vector.h>

#  include <psb_c_base.h>
#  include <psb_c_dbase.h>



DEAL_II_NAMESPACE_OPEN

namespace PSCToolkit
{

  Vector::Vector()
  {
    psblas_vector = nullptr;
  }


  Vector::~Vector()
  {
    Assert(psblas_vector != nullptr, ExcMessage("PSBLAS vector is null."));
    int err = psb_c_dgefree(psblas_vector, psblas_descriptor.get());
    Assert(err == 0, ExcMessage("Error freeing PSBLAS vector."));
  }


  Vector::Vector(const IndexSet &local_partitioning, const MPI_Comm comm)
  {
    communicator  = comm;
    psblas_vector = nullptr;
    // forward the call to reinit function below.
    reinit(local_partitioning, communicator);
  }



  void
  Vector::reinit(const IndexSet &local_partitioning, const MPI_Comm comm)
  {
    Assert(psblas_vector == nullptr,
           ExcMessage("PSBLAS vector must not be initialized."));

    Assert(communicator != MPI_COMM_NULL,
           ExcMessage("MPI_COMM_NULL passed to Vector::reinit()."));
    communicator = communicator;
    // Convert MPI_Comm to Fortran-style communicator
    MPI_Fint f_comm = MPI_Comm_c2f(comm);
    psblas_context  = psb_c_new_ctxt();
    psb_c_init_from_fint(psblas_context, f_comm);

    // Create a new PSBLAS descriptor
    psblas_descriptor.reset(psb_c_new_descriptor());

    // Use get_index_vector() from IndexSet to get the indexes
    const std::vector<types::global_dof_index> &indexes =
      local_partitioning.get_index_vector();
    owned_elements = local_partitioning;

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

    // Create a new PSBLAS vector
    psblas_vector = psb_c_new_dvector();

    int err = psb_c_dgeall_remote(psblas_vector, psblas_descriptor.get());
    Assert(err == 0, ExcMessage("Error initializing PSBLAS vector."));
  }



  MPI_Comm
  Vector::get_mpi_communicator() const
  {
    return communicator;
  }



  psb_c_descriptor *
  Vector::get_psblas_descriptor() const
  {
    return psblas_descriptor.get();
  }


  void
  Vector::clear()
  {
    // Reset the vector
    psblas_vector = nullptr;
    owned_elements.clear();
    owned_elements.set_size(0);
  }


  double
  Vector::linfty_norm() const
  {
    return psb_c_dgenrmi(psblas_vector, psblas_descriptor.get());
  }



  double
  Vector::l1_norm() const
  {
    return psb_c_dgeasum(psblas_vector, psblas_descriptor.get());
  }


  double
  Vector::l2_norm() const
  {
    return psb_c_dgenrm2(psblas_vector, psblas_descriptor.get());
  }



  Vector::size_type
  Vector::local_size() const
  {
    return psb_c_dvect_get_nrows(psblas_vector);
  }



  void
  Vector::set(const std::vector<Vector::size_type>  &indices,
              const std::vector<Vector::value_type> &values)
  {
    Assert(psblas_vector != nullptr && psblas_descriptor.get() != nullptr,
           ExcMessage("PSBLAS vector or descriptor is null."));
    Assert(indices.size() == values.size(),
           ExcMessage("Indices and values size mismatch."));

    psb_i_t nz = indices.size(); // Number of non-zero entries

    // Allocate memory for row indices and values. We need to subtract the
    // current value in order to set the value.
    psb_l_t *irw = (psb_l_t *)malloc(nz * sizeof(psb_l_t));
    psb_d_t *val = (psb_d_t *)malloc(nz * sizeof(psb_d_t));
    for (psb_i_t i = 0; i < nz; ++i)
      {
        irw[i] = indices[i];
        val[i] = values[i] -
                 psb_c_dgetelem(psblas_vector, irw[i], psblas_descriptor.get());
      }

    int err =
      psb_c_dgeins(nz /*nz*/, irw, val, psblas_vector, psblas_descriptor.get());
    Assert(err == 0, ExcMessage("Error inserting values into PSBLAS vector."));

    // Free allocated memory
    free(irw);
    free(val);
  }

  void
  Vector::add(const std::vector<Vector::size_type>  &indices,
              const std::vector<Vector::value_type> &values)
  {
    Assert(psblas_vector != nullptr && psblas_descriptor.get() != nullptr,
           ExcMessage("PSBLAS vector or descriptor is null."));
    Assert(indices.size() == values.size(),
           ExcMessage("Indices and values size mismatch."));

    psb_i_t nz = indices.size(); // Number of non-zero entries

    // Allocate memory for row indices and values
    psb_l_t *irw = (psb_l_t *)malloc(nz * sizeof(psb_l_t));
    psb_d_t *val = (psb_d_t *)malloc(nz * sizeof(psb_d_t));
    for (psb_i_t i = 0; i < nz; ++i)
      {
        irw[i] = indices[i];
        val[i] = values[i];
      }

    int err =
      psb_c_dgeins(nz /*nz*/, irw, val, psblas_vector, psblas_descriptor.get());
    Assert(err == 0, ExcMessage("Error inserting values into PSBLAS vector."));

    // Free allocated memory
    free(irw);
    free(val);
  }



  Vector::value_type
  Vector::operator()(const Vector::size_type index) const
  {
    // TODO: check index in range.
    return psb_c_dgetelem(psblas_vector, index, psblas_descriptor.get());
  }

  Vector::VectorReference
  Vector::operator()(const size_type index)
  {
    // TODO: check index in range.
    return VectorReference(*this, index);
  }


  void
  Vector::compress()
  {
    Assert(psblas_vector != nullptr && psblas_descriptor.get() != nullptr,
           ExcMessage("PSBLAS vector or descriptor is null."));

    int err = psb_c_cdasb(psblas_descriptor.get());
    Assert(err == 0, ExcMessage("Error while finalizing descriptor."));

    err = psb_c_dgeasb(psblas_vector, psblas_descriptor.get());
    Assert(err == 0, ExcMessage("Error compressing PSBLAS vector."));
  }


  bool
  Vector::has_ghost_elements() const
  {
    // TODO
    return false;
  }

} // namespace PSCToolkit

DEAL_II_NAMESPACE_CLOSE
#endif
