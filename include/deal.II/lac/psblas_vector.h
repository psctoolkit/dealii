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

#include <deal.II/base/exceptions.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/types.h>

#include <memory.h>

#include <cstddef>

#ifdef DEAL_II_WITH_PSBLAS

#  include <deal.II/lac/psctoolkit.h>

DEAL_II_NAMESPACE_OPEN

namespace PSCToolkit
{

  class Vector : public ReadVector<double>
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
        Assert(!vector.has_ghost_elements(), ExcGhostsPresent());
        std::vector<size_type>  idx{index};
        std::vector<value_type> value{s};
        vector.set(idx, value);
        return *this;
      }

      /**
       * Add <tt>s</tt> to the referenced element of the vector.
       */
      const VectorReference &
      operator+=(const value_type &s) const
      {
        Assert(!vector.has_ghost_elements(), ExcGhostsPresent());
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
        Assert(!vector.has_ghost_elements(), ExcGhostsPresent());
        std::vector<size_type>  idx{index};
        std::vector<value_type> value{-s};
        vector.add(idx, value);
        return *this;
      }

      /**
       * Multiply <tt>s</tt> to the referenced element of the vector.
       */
      const VectorReference &
      operator*=(const value_type &s) const
      {
        Assert(!vector.has_ghost_elements(), ExcGhostsPresent());
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
      operator/=(const value_type &s) const
      {
        Assert(!vector.has_ghost_elements(), ExcGhostsPresent());
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

    Vector(const IndexSet &local_partitioning,
           const IndexSet &ghost_indices,
           const MPI_Comm  communicator = MPI_COMM_WORLD);

    ~Vector();

    void
    reinit(const IndexSet &local_partitioning,
           const MPI_Comm  communicator = MPI_COMM_WORLD);

    /**
     * Construct a new parallel ghosted PSBLAS vector from IndexSets.
     *
     * Note that the @p ghost IndexSet may be empty and that any indices
     * already contained in @p local are ignored during construction. The
     * global indices in ghost are supplied as ghost indices so that they can be
     * read locally.
     *
     * @note This operation always creates a ghosted vector, which is considered
     * read-only.
     */
    void
    reinit(const IndexSet &local_partitioning,
           const IndexSet &ghost_indices,
           const MPI_Comm  communicator = MPI_COMM_WORLD);

    /**
     * Copy operator. Vectors are assumed to be of the same sizes.
     * TODO(s): - be conforming with PETSc behaviour (resizing if necessary)
     *          - document better
     */
    Vector &
    operator=(const Vector &v);

    /**
     * Return the global size of the vector, i.e. the sum of the local sizes
     * over all MPI processes.
     */
    size_type
    size() const override;

    virtual void
    extract_subvector_to(
      const ArrayView<const types::global_dof_index> &indices,
      const ArrayView<double>                        &elements) const override;


    template <typename ForwardIterator, typename OutputIterator>
    void
    extract_subvector_to(ForwardIterator indices_begin,
                         ForwardIterator indices_end,
                         OutputIterator  values_begin) const;

    /*
     *  Return the local size of the vector, i.e., the number of indices owned
     * locally.
     */
    size_type
    locally_owned_size() const;

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

    const IndexSet &
    locally_owned_elements() const;

    /**
     * Returns whether or not the vector has ghost elements.
     */
    bool
    has_ghost_elements() const;

    /**
     * Gathers the values of the ghost elements.
     */
    void
    update_ghost_values() const;

    void
    compress();

    value_type *
    begin();

    const value_type *
    begin() const;

    value_type *
    end();

    const value_type *
    end() const;

    MPI_Comm
    get_mpi_communicator() const;

    /**
     * Get the underlying PSBLAS descriptor.
     */
    psb_c_descriptor *
    get_psblas_descriptor() const;

    /**
     * Get the underlying PSBLAS context.
     */
    psb_c_ctxt *
    get_psblas_context() const;

    /**
     * Get a pointer to the underlying PSBLAS vector.
     */
    psb_c_dvector *
    get_psblas_vector() const;

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

    value_type *data;

    /*
     * PSBLAS descriptor.
     */
    std::shared_ptr<psb_c_descriptor> psblas_descriptor;

    MPI_Comm communicator;

    IndexSet owned_elements;

    IndexSet ghost_indices;

    bool ghosted;

    friend class SparseMatrix;
  };


  /* ----------------------------- Inline functions and templates
   * ---------------- */


  inline PSCToolkit::Vector::value_type *
  PSCToolkit::Vector::begin()
  {
    return data;
  }



  inline const PSCToolkit::Vector::value_type *
  PSCToolkit::Vector::begin() const
  {
    return data;
  }



  inline PSCToolkit::Vector::value_type *
  PSCToolkit::Vector::end()
  {
    return data + locally_owned_size();
  }



  inline const PSCToolkit::Vector::value_type *
  PSCToolkit::Vector::end() const
  {
    return data + locally_owned_size();
  }



  /*
   * Return the global size of the vector, i.e. the sum of the local sizes
   * over all MPI processes.
   */
  inline Vector::size_type
  Vector::size() const
  {
    return owned_elements.size();
  }



  inline const IndexSet &
  Vector::locally_owned_elements() const
  {
    return owned_elements;
  }



  inline bool
  Vector::has_ghost_elements() const
  {
    return ghosted;
  }



  inline void
  Vector::extract_subvector_to(
    const ArrayView<const types::global_dof_index> &indices,
    const ArrayView<double>                        &elements) const
  {
    AssertDimension(indices.size(), elements.size());
    extract_subvector_to(indices.begin(), indices.end(), elements.begin());
  }



  template <typename ForwardIterator, typename OutputIterator>
  inline void
  Vector::extract_subvector_to(ForwardIterator indices_begin,
                               ForwardIterator indices_end,
                               OutputIterator  output) const
  {
    if (indices_begin == indices_end)
      return;

    if (ghosted)
      {
        Assert(false, ExcNotImplemented());
      }
    else
      {
        // no ghost elements, so we can
        // just access the local
        // elements directly
        while (indices_begin != indices_end)
          {
            const size_type index = *indices_begin;
            Assert(owned_elements.is_element(index),
                   ExcMessage("You are accessing elements of a vector without "
                              "ghost elements that are not actually owned by "
                              "this vector. A typical case where this may "
                              "happen is if you are passing a non-ghosted "
                              "(completely distributed) vector to a function "
                              "that expects a vector that stores ghost "
                              "elements for all locally relevant or locally "
                              "active vector entries."));

            *output =
              psb_c_dgetelem(psblas_vector, index, psblas_descriptor.get());

            ++indices_begin;
            ++output;
          }
      }
  }


} // namespace PSCToolkit
DEAL_II_NAMESPACE_CLOSE

#endif // DEAL_II_WITH_PSBLAS
#endif // dealii_psblas_vector_h
