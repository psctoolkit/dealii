// ------------------------------------------------------------------------
//
// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (C) 2008 - 2023 by the deal.II authors
//
// This file is part of the deal.II library.
//
// Part of the source code is dual licensed under Apache-2.0 WITH
// LLVM-exception OR LGPL-2.1-or-later. Detailed license information
// governing the source code and code contributions can be found in
// LICENSE.md and CONTRIBUTING.md at the top level directory of deal.II.
//
// ------------------------------------------------------------------------

#include <deal.II/lac/psblas_precondition.h>

#ifdef DEAL_II_WITH_AMG4PSBLAS

DEAL_II_NAMESPACE_OPEN

namespace PSCToolkit
{

  PreconditionAMG::PreconditionAMG()
  {
    psblas_preconditioner = amg_c_dprec_new();
  }


  void
  PreconditionAMG::initialize(const SparseMatrix   &matrix,
                              const AdditionalData &additional_data)
  {
    Assert(matrix.psblas_sparse_matrix != nullptr,
           ExcMessage("Matrix has not been initialized."));

    // set descriptor
    psblas_descriptor = matrix.psblas_descriptor;

    // TODO: fill ptype through AdditionalData parameters

    char ptype[40];
    strcpy(ptype, "ML");

    int err = amg_c_dprecinit(*InitFinalize::get_psblas_context(),
                              psblas_preconditioner,
                              ptype);
    amg_c_dprecseti(psblas_preconditioner, "SMOOTHER_SWEEPS", 2);
    amg_c_dprecseti(psblas_preconditioner, "SUB_FILLIN", 1);
    amg_c_dprecsetc(psblas_preconditioner, "COARSE_SOLVE", "BJAC");
    amg_c_dprecsetc(psblas_preconditioner, "COARSE_SUBSOLVE", "ILU");
    amg_c_dprecseti(psblas_preconditioner, "COARSE_FILLIN", 0);

    // build AMG hierarchy
    err = amg_c_dhierarchy_build(matrix.psblas_sparse_matrix,
                                 psblas_descriptor.get(),
                                 psblas_preconditioner);
    //... and smoothers
    AssertThrow(err == 0, ExcMessage("Error while building AMG hierarchy."));
    err = amg_c_dsmoothers_build(matrix.psblas_sparse_matrix,
                                 psblas_descriptor.get(),
                                 psblas_preconditioner);
    AssertThrow(err == 0, ExcMessage("Error while building AMG smoothers."));
  }


  PreconditionAMG::~PreconditionAMG()
  {
    free(psblas_preconditioner);
  }


  MPI_Comm
  PreconditionAMG::get_mpi_communicator() const
  {
    return communicator;
  }



  void
  PreconditionAMG::vmult(Vector &dst, const Vector &src) const
  {
    // TODO: expose apply routine to C interface
    // int err = amg_c_dapply(dst.psblas_vector,
    //                        src.psblas_vector,
    //                        psblas_descriptor.get(),
    //                        false /* trans*/);
    // Assert(ierr == 0,
    //        ExcMessage("Failure while applying preconditioner on a vector."));
  }



  void
  PreconditionAMG::Tvmult(Vector &dst, const Vector &src) const
  {
    // TODO: expose apply routine to C interface
    // int err = amg_c_dapply(dst.psblas_vector,
    //                        src.psblas_vector,
    //                        psblas_descriptor.get(),
    //                        true /* trans*/);
    // Assert(ierr == 0,
    //        ExcMessage(
    //          "Failure while applying preconditioner (transpose) on a
    //          vector."));
  }



} // namespace PSCToolkit

DEAL_II_NAMESPACE_CLOSE

#endif