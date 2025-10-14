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

    /**
     * Standardized data struct to pipe additional flags to the
     * preconditioner.
     */
    struct AdditionalData
    {};

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
     * Let Trilinos compute a multilevel hierarchy for the solution of a
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
     * Sets an internal flag so that all operations performed by the matrix,
     * i.e., multiplications, are done in transposed order. However, this does
     * not reshape the matrix to transposed form directly, so care should be
     * taken when using this flag.
     *
     * @note Calling this function any even number of times in succession will
     * return the object to its original state.
     */
    void
    transpose();

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

  protected:
    /**
     * This is a pointer to the preconditioner object that is used when
     * applying the preconditioner.
     */
    amg_c_dprec *psblas_preconditioner;

    /*
     * PSBLAS descriptor.
     */
    std::shared_ptr<psb_c_descriptor> psblas_descriptor;


    /**
     * Internal communication pattern in case the matrix needs to be copied
     * from deal.II format.
     */
    MPI_Comm communicator;
  };

} // namespace PSCToolkit

DEAL_II_NAMESPACE_CLOSE

#endif
#endif
