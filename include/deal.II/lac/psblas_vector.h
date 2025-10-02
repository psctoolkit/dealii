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

#ifndef dealii_psblas_vector_h
#define dealii_psblas_vector_h

#include <deal.II/base/config.h>

#include "deal.II/base/enable_observer_pointer.h"
#include <deal.II/base/types.h>

#include <memory.h>

#include <cstddef>

#ifdef DEAL_II_WITH_PSBLAS

#  include <deal.II/lac/psctoolkit.h>

DEAL_II_NAMESPACE_OPEN

namespace PSCToolkit
{

  class Vector : public EnableObserverPointer
  {
  private:
    /**
     * This class provides a wrappers for accessing psblas vector elements.
     */
    class VectorReference
    {
    private:
      using size_type = types::global_dof_index;

      using value_type = double;

      /**
       * Constructor.
       */
      VectorReference(Vector &vector, const size_type index)
        : vector(vector)
        , index(index)
      {}

    public:
      /**
       * Set the referenced element of the vector to <tt>s</tt>.
       */
      const VectorReference &
      operator=(const value_type &s) const
      {
        std::vector<size_type>  idx{index};
        std::vector<value_type> value{s};
        vector.set(idx, value);
        return *this;
      }

      /**
       * Add <tt>s</tt> to the referenced element of the vector.
       */
      const VectorReference &
      operator+=(const TrilinosScalar &s) const
      {
        std::vector<size_type>  idx{index};
        std::vector<value_type> value{s};
        vector.add(idx, value);
        return *this;
      }

      /**
       * Subtract <tt>s</tt> to the referenced element of the vector.
       */
      const VectorReference &
      operator-=(const TrilinosScalar &s) const
      {
        std::vector<size_type>  idx{index};
        std::vector<value_type> value{-s};
        vector.add(idx, value);
        return *this;
      }

      /**
       * Multiply <tt>s</tt> to the referenced element of the vector.
       */
      const VectorReference &
      operator*=(const TrilinosScalar &s) const
      {
        std::vector<size_type>  idx{index};
        value_type              new_value = static_cast<value_type>(*this) * s;
        std::vector<value_type> value{new_value};
        vector.set(idx, value);
        return *this;
      }

      /**
       * Divide <tt>s</tt> to the referenced element of the vector.
       */
      const VectorReference &
      operator/=(const TrilinosScalar &s) const
      {
        std::vector<size_type>  idx{index};
        value_type              new_value = static_cast<value_type>(*this) / s;
        std::vector<value_type> value{new_value};
        vector.set(idx, value);
        return *this;
      }

      /*
       * Convert the reference to an actual value, i.e. return the value of
       * the referenced element of the vector.
       */
      operator value_type() const
      {
        return psb_c_dgetelem(vector.psblas_vector,
                              index,
                              vector.psblas_descriptor.get());
      };

    private:
      Vector &vector;

      const size_type index;

      friend class Vector;
    };

  public:
    using size_type = dealii::types::global_dof_index;

    using value_type = double;

    Vector();

    Vector(const IndexSet &local_partitioning,
           const MPI_Comm  communicator = MPI_COMM_WORLD);

    ~Vector();

    void
    reinit(const IndexSet &local_partitioning,
           const MPI_Comm  communicator = MPI_COMM_WORLD);

    size_type
    size() const;

    /*
     *  Return the local size of the vector, i.e., the number of indices owned
     * locally.
     */
    size_type
    local_size() const;

    void
    set(const std::vector<size_type>  &indices,
        const std::vector<value_type> &values);

    void
    add(const std::vector<size_type>  &indices,
        const std::vector<value_type> &values);


    /**
     * Provide read-only access to an element.
     */
    value_type
    operator()(const size_type index) const;

    /**
     * Provide read-write access to an element of the vector.
     */
    VectorReference
    operator()(const size_type index);

    bool
    has_ghost_elements() const;

    void
    compress();

    MPI_Comm
    get_mpi_communicator() const;

    /**
     * Get the underlying PSBLAS descriptor.
     */
    psb_c_descriptor *
    get_psblas_descriptor() const;

    void
    clear();

    double
    linfty_norm() const;

    double
    l1_norm() const;

    double
    l2_norm() const;

  private:
    /*
     * Pointer to the underlying PSBLAS vector.
     */
    psb_c_dvector *psblas_vector;

    /*
     * PSBLAS context.
     */
    psb_c_ctxt *psblas_context;

    /*
     * PSBLAS descriptor.
     */
    std::shared_ptr<psb_c_descriptor> psblas_descriptor;

    MPI_Comm communicator;

    IndexSet owned_elements;

    friend class SparseMatrix;
  };


  /*
   * Return the global size of the vector, i.e. the sum of the local sizes
   * over all MPI processes.
   */
  inline Vector::size_type
  Vector::size() const
  {
    return owned_elements.size();
  }

} // namespace PSCToolkit
DEAL_II_NAMESPACE_CLOSE

#endif // DEAL_II_WITH_PSBLAS
#endif // dealii_psblas_vector_h
