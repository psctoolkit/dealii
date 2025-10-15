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

#ifndef dealii_psblas_precondition_h
#define dealii_psblas_precondition_h

#include <deal.II/lac/psblas_sparse_matrix.h>
#include <deal.II/lac/psblas_vector.h>


#ifdef DEAL_II_WITH_AMG4PSBLAS

#  include <amg_cbind.h>

DEAL_II_NAMESPACE_OPEN

namespace PSCToolkit
{

  /**
   * This class implements an interface to the AMG4PSBLAS algebraic
   * multigrid preconditioner.
   *
   * @note This class is only available if deal.II was configured with
   * <tt>--with-psctoolkit</tt> and <tt>--with-amg4psblas</tt>.
   *
   */

  class PreconditionAMG : public EnableObserverPointer
  {
  public:
    /**
     * Declare the type for container size.
     */
    using size_type = dealii::types::global_dof_index;

    using value_type = double;

    /**
     * Standardized data struct to pipe additional flags to the
     * preconditioner. See the AMG4PSBLAS documentation for possible
     * parameters and their meaning.
     */
    struct AdditionalData
    {
      AdditionalData(const char        *cycle_type            = "VCYCLE",
                     const unsigned int n_cycles              = 1,
                     const double       aggregation_threshold = 1e-2,
                     const char        *aggregation_type      = "SOC1",
                     const char        *smoother_type         = "FBGS",
                     const unsigned int smoother_sweeps       = 2,
                     const unsigned int smoother_degree       = 2,
                     const char        *aggr_prol             = "SMOOTHED",
                     const char        *coarse_type           = "BJAC",
                     const char        *coarse_mat_type       = "REPL")
        : cycle_type(cycle_type)
        , n_cycles(n_cycles)
        , aggregation_threshold(aggregation_threshold)
        , aggregation_type(strdup(aggregation_type))
        , smoother_type(smoother_type)
        , smoother_sweeps(smoother_sweeps)
        , smoother_degree(smoother_degree)
        , aggr_prol(aggr_prol)
        , coarse_type(coarse_type)
        , coarse_mat_type(coarse_mat_type)
      {}


      /**
       * Multilevel cycle. Possible values are "VCYCLE", "WCYCLE", "KCYCLE", and
       * "ADD". See the AMG4PSBLAS documentation for details.
       */
      const char *cycle_type;

      /*
       * Number of multilevel cycles to be performed.
       */
      unsigned int n_cycles;

      /**
       * Threshold for the strength of connection algorithm. Defaults to 0.01.
       */
      double aggregation_threshold;

      /**
       * Type of aggregation algorithm. Possible values are "SOC1", "SOC2",
       * "MATCHBOXP".
       */
      char *aggregation_type;

      /**
       * Type of smoother used in the multilevel preconditioner.
       */
      const char *smoother_type;

      /**
       * Number of sweeps of the smoother or one-level preconditioner. It is
       * ignored if the smoother is "POLY".
       */
      unsigned int smoother_sweeps;

      /**
       * Degree of the polynomial smoother, which equals the number of
       * matrix-vector products performed by the smoother. Ignored if the
       * smoother is not 'POLY'.
       */
      unsigned int smoother_degree;

      /**
       * Prolongator used by the aggregation algorithm: smoothed or unsmoothed
       * (i.e., tentative prolongator).
       */
      const char *aggr_prol;

      /**
       * Solver used at the coarsest level. See AMG4PSBLAS documentation for
       * possible values.
       */
      const char *coarse_type;

      /**
       * Coarsest matrix layout: distributed among processes, or replicated on
       * each one.
       */
      const char *coarse_mat_type;
    };

    /**
     * Constructor. Does not do anything. The <tt>initialize</tt> function of
     * the derived classes will have to create the preconditioner from a given
     * sparse matrix.
     */
    PreconditionAMG();

    /**
     * Destructor.
     */
    ~PreconditionAMG();

    /**
     * Let AMG4PSBLAS compute a multilevel hierarchy for the solution of a
     * linear system with the given matrix. The function uses the matrix
     * format specified in PSCToolkit::SparseMatrix.
     */
    void
    initialize(const SparseMatrix   &matrix,
               const AdditionalData &additional_data = AdditionalData());

    /**
     * Destroys the preconditioner, leaving an object like just after having
     * called the constructor.
     */
    void
    clear();

    /**
     * Return the underlying MPI communicator.
     */
    MPI_Comm
    get_mpi_communicator() const;

    /**
     * Return the pointer the underlying preconditioner. This is for advanced
     * users only who want to try out additional features of the AMG4PSBLAS
     * preconditioner not supported through the current interface.
     */
    amg_c_dprec *
    get_psblas_preconditioner();

    /**
     * Apply the preconditioner.
     */
    void
    vmult(Vector &dst, const Vector &src) const;

    /**
     * Apply the transpose preconditioner.
     */
    void
    Tvmult(Vector &dst, const Vector &src) const;


    friend class SolverBase;

  private:
    /**
     * This is a pointer to the preconditioner object that is used when
     * applying the preconditioner.
     */
    amg_c_dprec *psblas_preconditioner;

    /*
     * PSBLAS descriptor.
     */
    std::shared_ptr<psb_c_descriptor> psblas_descriptor;
  };

} // namespace PSCToolkit

DEAL_II_NAMESPACE_CLOSE

#endif
#endif
