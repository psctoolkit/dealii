// -----------------------------------------------------------------------------
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception OR LGPL-2.1-or-later
// Copyright (C) 2008 - 2026 by the deal.II authors
//
// This file is part of the deal.II library.
//
// Detailed license information governing the source code and contributions
// can be found in LICENSE.md and CONTRIBUTING.md at the top level directory.
//
// -----------------------------------------------------------------------------



#include <deal.II/lac/psblas_precondition.h>

#ifdef DEAL_II_WITH_AMG4PSBLAS

#  include <cstdlib>

DEAL_II_NAMESPACE_OPEN

namespace PSCToolkitWrappers
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
    AssertDimension(matrix.m(), matrix.n());
    Assert(matrix.get_psblas_descriptor() ==
             matrix.get_psblas_column_descriptor(),
           ExcMessage("AMG requires identical row and column partitions."));

    if (psblas_preconditioner == nullptr)
      psblas_preconditioner = amg_c_dprec_new();

    // Share the matrix descriptor used by the native AMG operations.
    psblas_descriptor = matrix.psblas_descriptor;

    int err =
      amg_c_dprecinit(*matrix.psblas_context, psblas_preconditioner, "ML");
    AssertThrow(err == 0,
                ExcMessage("Error " + std::to_string(err) +
                           " while initializing AMG preconditioner."));

    amg_c_dprecsetc(psblas_preconditioner,
                    "ML_CYCLE",
                    additional_data.cycle_type);
    amg_c_dprecseti(psblas_preconditioner,
                    "CYCLE_SWEEPS",
                    additional_data.n_cycles);
    amg_c_dprecsetr(psblas_preconditioner,
                    "AGGR_THRSH",
                    additional_data.aggregation_threshold);
    amg_c_dprecsetc(psblas_preconditioner,
                    "PAR_AGGR_ALG",
                    additional_data.parallel_aggr_algorithm);
    amg_c_dprecsetc(psblas_preconditioner,
                    "AGGR_TYPE",
                    additional_data.aggregation_type);
    amg_c_dprecseti(psblas_preconditioner,
                    "AGGR_SIZE",
                    additional_data.aggregation_size);
    amg_c_dprecsetc(psblas_preconditioner,
                    "AGGR_PROL",
                    additional_data.aggr_prol);
    amg_c_dprecsetc(psblas_preconditioner,
                    "AGGR_FILTER",
                    additional_data.aggr_filter);
    amg_c_dprecsetc(psblas_preconditioner,
                    "SMOOTHER_TYPE",
                    additional_data.smoother_type);

    // From the AMG4PSBLAS manual: if "SMOOTHER_TYPE" is set to
    // "POLY", then "SMOOTHER_SWEEPS" is ignored and the number of smoother
    // sweeps equals the polynomial degree is used instead.
    if (std::strcmp(additional_data.smoother_type, "POLY") == 0)
      amg_c_dprecseti(psblas_preconditioner,
                      "POLY_DEGREE",
                      additional_data.smoother_sweeps);
    else
      amg_c_dprecseti(psblas_preconditioner,
                      "SMOOTHER_SWEEPS",
                      additional_data.smoother_sweeps);
    amg_c_dprecsetc(psblas_preconditioner,
                    "COARSE_SOLVE",
                    additional_data.coarse_type);

    // build AMG hierarchy
    err = amg_c_dhierarchy_build(matrix.psblas_sparse_matrix,
                                 psblas_descriptor.get(),
                                 psblas_preconditioner);
    AssertThrow(err == 0,
                ExcMessage("Error " + std::to_string(err) +
                           " while building AMG hierarchy."));
    //... and smoothers
    err = amg_c_dsmoothers_build(matrix.psblas_sparse_matrix,
                                 psblas_descriptor.get(),
                                 psblas_preconditioner);
    AssertThrow(err == 0,
                ExcMessage("Error " + std::to_string(err) +
                           " while building AMG smoothers."));

    if (additional_data.output_details == true)
      {
        err = amg_c_ddescr(psblas_preconditioner);
        AssertThrow(err == 0,
                    ExcMessage("Error " + std::to_string(err) +
                               " while printing AMG description."));
      }
  }


  PreconditionAMG::~PreconditionAMG()
  {
    if (psblas_preconditioner)
      try
        {
          clear();
        }
      catch (...)
        {}
  }



  void
  PreconditionAMG::clear()
  {
    if (psblas_preconditioner != nullptr)
      {
        // An allocated C handle need not contain an initialized Fortran
        // preconditioner yet. AMG4PSBLAS cannot free an uninitialized one.
        if (psblas_preconditioner->dprec != nullptr)
          {
            const int err = amg_c_dprec_delete(psblas_preconditioner);
            AssertThrow(err == 0,
                        ExcCallingPSBLASFunction(err, "amg_c_dprec_delete"));
          }
        else
          std::free(psblas_preconditioner);

        psblas_preconditioner = nullptr;
      }

    psblas_descriptor.reset();
  }



  amg_c_dprec *
  PreconditionAMG::get_psblas_preconditioner()
  {
    return psblas_preconditioner;
  }



  void
  PreconditionAMG::vmult(Vector &dst, const Vector &src) const
  {
    AssertDimension(dst.size(), src.size());
    int err = amg_c_dprecapply(psblas_preconditioner,
                               src.psblas_vector,
                               dst.psblas_vector,
                               psblas_descriptor.get());
    Assert(err == 0,
           ExcMessage("Failure while applying preconditioner on a vector."));
  }



  void
  PreconditionAMG::Tvmult(Vector &dst, const Vector &src) const
  {
    AssertDimension(dst.size(), src.size());
    DEAL_II_NOT_IMPLEMENTED();
  }



} // namespace PSCToolkitWrappers

DEAL_II_NAMESPACE_CLOSE

#endif
