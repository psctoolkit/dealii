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


#include <deal.II/base/memory_consumption.h>
#include <deal.II/base/mpi.h>

#include <deal.II/lac/psblas_vector.h>

#include <cstdlib>

#ifdef DEAL_II_WITH_PSBLAS
#  include <psb_base_cbind.h>
#  include <psb_c_base.h>
#  include <psb_c_dbase.h>

DEAL_II_NAMESPACE_OPEN

namespace PSCToolkitWrappers
{

  Vector::Vector()
    : psblas_vector(nullptr)
    , psblas_context(InitFinalize::get_psblas_context())
    , psblas_descriptor(nullptr)
    , communicator(MPI_COMM_NULL)
    , ghosted(false)
    , state(internal::State::Default)
    , last_action(VectorOperation::unknown)
    , remote_entries_pending(false)
  {}


  Vector::Vector(const Vector &v)
    : Vector::Vector()
  {
    if (v.has_ghost_elements())
      reinit(v.owned_elements, v.ghost_indices, v.communicator);
    else
      reinit(v.owned_elements, v.communicator);

    this->operator=(v);
  }


  Vector::~Vector()
  {
    release_vector();
  }



  void
  Vector::release_vector()
  {
    if (psblas_vector != nullptr)
      {
        const int ierr = psb_c_dgefree(psblas_vector, psblas_descriptor.get());
        AssertNothrow(ierr == 0, ExcFreePSBLASVector(ierr));
        std::free(psblas_vector);
        psblas_vector = nullptr;
      }
  }



  Vector::Vector(const IndexSet &local_partitioning, const MPI_Comm comm)
    : Vector()
  {
    // forward the call to reinit function below.
    reinit(local_partitioning, comm);
  }



  Vector::Vector(const IndexSet &local_partitioning,
                 const IndexSet &ghost_indices,
                 const MPI_Comm  comm)
    : Vector()
  {
    // forward the call to reinit function taking ghost indices.
    reinit(local_partitioning, ghost_indices, comm);
  }



  void
  Vector::reinit(const IndexSet &local_partitioning,
                 const MPI_Comm  comm,
                 const bool      omit_zeroing_entries)
  {
    Assert(comm != MPI_COMM_NULL,
           ExcMessage("MPI_COMM_NULL passed to Vector::reinit()."));

    // A PSBLAS descriptor describes the parallel partitioning, so it can only
    // be recycled if that partitioning stays the same. Note: comparing global
    // sizes is not enough because two **different** partitionings can have the
    // same global size.
    const bool partitioning_changes_locally =
      (psblas_descriptor.get() == nullptr) || (ghosted == true) ||
      (communicator != comm) ||
      (owned_elements.size() != local_partitioning.size()) ||
      (owned_elements != local_partitioning);

    // Creating a descriptor is a collective operation, so the processes have to
    // agree on whether one is built. If changes are only on some
    // of the processes, we must therefore  trigger the rebuild everywhere.
    const bool partitioning_will_change =
      Utilities::MPI::logical_or(partitioning_changes_locally, comm);

    release_vector();

    communicator   = comm;
    ghosted        = false;
    owned_elements = local_partitioning;
    // index_within_set() requires a compressed index set
    owned_elements.compress();

    int ierr;
    if (partitioning_will_change == true)
      {
        const auto context = std::make_shared<internal::Context>(comm);
        psblas_context     = context->get_psblas_context();
        psblas_descriptor.reset(psb_c_new_descriptor(),
                                [context](psb_c_descriptor *descriptor) {
                                  internal::DescriptorDeleter()(descriptor);
                                });

        // Use get_index_vector() from IndexSet to get the indexes
        const std::vector<types::global_dof_index> &indexes =
          local_partitioning.get_index_vector();

        const auto number_of_local_indexes =
          indexes.size(); // Number of local indexes

        std::vector<psb_c_l_t> vl(number_of_local_indexes);
        for (std::size_t i = 0; i < number_of_local_indexes; ++i)
          {
            const auto psblas_index = static_cast<psb_c_l_t>(indexes[i]);
            AssertIntegerConversion(psblas_index, indexes[i]);
            vl[i] = psblas_index;
          }

        // Insert the indexes into the descriptor
        ierr = psb_c_cdall_vl(number_of_local_indexes,
                              vl.data(),
                              *psblas_context,
                              psblas_descriptor.get());

        // Free the vl array
        Assert(ierr == 0, ExcInitializePSBLASDescriptor(ierr));
      }

    if (!psb_c_cd_is_asb(psblas_descriptor.get()))
      {
        ierr = psb_c_cdasb(psblas_descriptor.get());
        Assert(ierr == 0, ExcAssemblePSBLASDescriptor(ierr));
      }

    // Create a new PSBLAS vector and allocate mem space for vector
    psblas_vector = psb_c_new_dvector();

    ierr = psb_c_dgeall_remote_options(psblas_vector,
                                       psblas_descriptor.get(),
                                       PSB_MATBLD_REMOTE,
                                       PSB_DUPL_DEF);
    Assert(ierr == 0, ExcInitializePSBLASVector(ierr));

    // The native vector must be assembled too: rectangular matrix products
    // consume this handle directly, just like the existing square products.
    ierr = psb_c_dgeasb_options(psblas_vector,
                                psblas_descriptor.get(),
                                PSB_DUPL_DEF);
    Assert(ierr == 0, ExcAssemblePSBLASVector(ierr));

    if (omit_zeroing_entries == false)
      {
        ierr = psb_c_dvect_set_scal(psblas_vector, 0.0);
        Assert(ierr == 0,
               ExcCallingPSBLASFunction(ierr, "psb_c_dvect_set_scal"));
      }

    state                  = internal::State::Assembled;
    last_action            = VectorOperation::unknown;
    remote_entries_pending = false;
  }



  void
  Vector::reinit(const IndexSet &local_partitioning,
                 const IndexSet &ghosts,
                 const MPI_Comm  comm)
  {
    Assert(comm != MPI_COMM_NULL,
           ExcMessage("MPI_COMM_NULL passed to Vector::reinit()."));

    IndexSet new_ghost_indices = ghosts;
    new_ghost_indices.subtract_set(local_partitioning);

    // A descriptor describes the parallel partitioning including the halo, so
    // it can only be recycled if both stay the same. Note: comparing global
    // sizes is not enough because two **different** partitionings can have the
    // same global size.
    const bool partitioning_changes_locally =
      (psblas_descriptor.get() == nullptr) || (ghosted == false) ||
      (communicator != comm) ||
      (owned_elements.size() != local_partitioning.size()) ||
      (owned_elements != local_partitioning) ||
      (ghost_indices.size() != new_ghost_indices.size()) ||
      (ghost_indices != new_ghost_indices);

    // Creating a descriptor is a collective operation, so the processes have to
    // agree on whether one is built.
    const bool partitioning_will_change =
      Utilities::MPI::logical_or(partitioning_changes_locally, comm);

    release_vector();

    communicator   = comm;
    ghosted        = true;
    owned_elements = local_partitioning;
    ghost_indices  = std::move(new_ghost_indices);
    // index_within_set() requires a  compressed index set
    owned_elements.compress();

    int ierr;
    if (partitioning_will_change == true)
      {
        const auto context = std::make_shared<internal::Context>(comm);
        psblas_context     = context->get_psblas_context();
        psblas_descriptor.reset(psb_c_new_descriptor(),
                                [context](psb_c_descriptor *descriptor) {
                                  internal::DescriptorDeleter()(descriptor);
                                });

        // Use get_index_vector() from IndexSet to get the indexes
        const std::vector<types::global_dof_index> &indexes =
          local_partitioning.get_index_vector();

        const auto number_of_local_indexes =
          indexes.size(); // Number of local indexes

        std::vector<psb_c_l_t> vl(number_of_local_indexes);
        std::vector<psb_c_i_t> lidx(number_of_local_indexes);
        for (std::size_t i = 0; i < number_of_local_indexes; ++i)
          {
            const auto psblas_index = static_cast<psb_c_l_t>(indexes[i]);
            const auto idx          = static_cast<psb_c_i_t>(i);
            AssertIntegerConversion(psblas_index, indexes[i]);
            AssertIntegerConversion(idx, i);
            vl[i]   = psblas_index;
            lidx[i] = idx;
          }

        // Ghost case. From the manual:
        // 1) psb_cdall() with vl (global) and lidx (local)
        // 2) psb_cdins(nz,ja,desc,info,lidx=lidx), where:
        // - ja contains halo indices
        // - lidx: corresponding local indices

        // Insert the indexes into the descriptor
        ierr = psb_c_cdall_vl_lidx(number_of_local_indexes,
                                   vl.data(),
                                   lidx.data(),
                                   *psblas_context,
                                   psblas_descriptor.get());

        Assert(ierr == 0, ExcInitializePSBLASDescriptor(ierr));

        // ... insert the ghost indices ...
        const std::vector<types::global_dof_index> &ghost_indexes =
          ghost_indices.get_index_vector();
        const auto number_of_ghost_indices = ghost_indexes.size();

        std::vector<psb_c_l_t> global_ghost_indices(number_of_ghost_indices);
        std::vector<psb_c_i_t> local_ghost_indices(number_of_ghost_indices);

        psb_c_i_t extended_idx_counter = number_of_local_indexes;
        for (std::size_t i = 0; i < number_of_ghost_indices; ++i)
          {
            const auto psblas_index = static_cast<psb_c_l_t>(ghost_indexes[i]);
            AssertIntegerConversion(psblas_index, ghost_indexes[i]);
            global_ghost_indices[i] = psblas_index;
            const auto idx = static_cast<psb_c_i_t>(extended_idx_counter);
            AssertIntegerConversion(idx, extended_idx_counter);
            local_ghost_indices[i] = extended_idx_counter++;
          }


        ierr = psb_c_cdins_lidx(number_of_ghost_indices,
                                global_ghost_indices.data(),
                                local_ghost_indices.data(),
                                psblas_descriptor.get());

        Assert(ierr == 0, ExcCallingPSBLASFunction(ierr, "psb_c_cdins_lidx"));
      }

    // Assemble the descriptor first so that the halo (ghost) information is
    // known before the vector is allocated and assembled.
    if (!psb_c_cd_is_asb(psblas_descriptor.get()))
      {
        ierr = psb_c_cdasb(psblas_descriptor.get());
        Assert(ierr == 0, ExcAssemblePSBLASDescriptor(ierr));
      }

    // Create a new PSBLAS vector and allocate mem space for it
    psblas_vector = psb_c_new_dvector();

    ierr = psb_c_dgeall_remote_options(psblas_vector,
                                       psblas_descriptor.get(),
                                       PSB_MATBLD_REMOTE,
                                       PSB_DUPL_DEF);
    Assert(ierr == 0, ExcInitializePSBLASVector(ierr));

    // Assemble the vector so that storage for the ghost (halo) elements is
    // allocated. This is required for update_ghost_values() (psb_c_dhalo) to
    // work correctly.
    ierr = psb_c_dgeasb_options(psblas_vector,
                                psblas_descriptor.get(),
                                PSB_DUPL_DEF);
    Assert(ierr == 0, ExcAssemblePSBLASVector(ierr));

    state                  = internal::State::Assembled;
    last_action            = VectorOperation::unknown;
    remote_entries_pending = false;
  }



  void
  Vector::reinit(const Vector &v, const bool omit_zeroing_entries)
  {
    release_vector();
    psblas_context    = v.psblas_context;
    psblas_descriptor = v.psblas_descriptor;
    if (v.has_ghost_elements())
      {
        reinit(v.locally_owned_elements(),
               v.ghost_indices,
               v.get_mpi_communicator());
        if (!omit_zeroing_entries)
          {
            int ierr = psb_c_dvect_set_scal(psblas_vector, 0.0);
            Assert(ierr == 0,
                   ExcCallingPSBLASFunction(ierr, "psb_c_dvect_set_scal"));
          }
      }
    else
      {
        reinit(v.owned_elements,
               v.get_mpi_communicator(),
               omit_zeroing_entries);
      }
  }



  Vector &
  Vector::operator=(const Vector &v)
  {
    Assert(v.last_action == VectorOperation::unknown,
           ExcWrongMode(VectorOperation::unknown, v.last_action));
    Assert(last_action == VectorOperation::unknown,
           ExcWrongMode(VectorOperation::unknown, last_action));
    Assert(v.state == internal::State::Assembled,
           ExcInvalidStateAssembled(v.state));

    // Vectors can have different sizes. If so, we first resize the vector
    if (size() != v.size())
      {
        release_vector();
        psblas_context = v.psblas_context;
        //  since v has been assembled, we can recycle its descriptor
        psblas_descriptor = v.psblas_descriptor;
        if (v.has_ghost_elements())
          reinit(v.locally_owned_elements(),
                 v.ghost_indices,
                 v.get_mpi_communicator());
        else
          reinit(v.owned_elements, v.get_mpi_communicator(), true);
      }

    // Copy only the locally owned entries from v into this vector. We use
    // psb_c_dgeaxpby (this = 1.0 * v + 0.0 * this)
    int ierr = psb_c_dgeaxpby(value_type(1.0),
                              v.psblas_vector,
                              value_type(0.0),
                              psblas_vector,
                              psblas_descriptor.get());
    AssertThrow(ierr == 0, ExcAXPBY(ierr));

    // If current vector has ghost elements, update them
    if (has_ghost_elements())
      update_ghost_values();

    state                  = internal::State::Assembled;
    remote_entries_pending = false;

    return *this;
  }



  Vector &
  Vector::operator=(const value_type s)
  {
    AssertIsFinite(s);
    Assert(state != internal::State::Default, ExcInvalidState(state));
    int ierr = psb_c_dvect_set_scal(psblas_vector, s);
    Assert(ierr == 0, ExcCallingPSBLASFunction(ierr, "psb_c_dvect_set_scal"));
    return *this;
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



  psb_c_dvector *
  Vector::get_psblas_vector() const
  {
    return psblas_vector;
  }



  void
  Vector::clear()
  {
    release_vector();
    psblas_descriptor.reset();
    psblas_context = InitFinalize::get_psblas_context();
    owned_elements.clear();
    owned_elements.set_size(0);
    ghost_indices.clear();
    ghost_indices.set_size(0);
    communicator           = MPI_COMM_NULL;
    ghosted                = false;
    state                  = internal::State::Default;
    last_action            = VectorOperation::unknown;
    remote_entries_pending = false;
  }



  Vector::value_type
  Vector::linfty_norm() const
  {
    Assert(state == internal::State::Assembled,
           ExcInvalidStateAssembled(state));
    return psb_c_dgenrmi(psblas_vector, psblas_descriptor.get());
  }



  Vector::value_type
  Vector::l1_norm() const
  {
    Assert(state == internal::State::Assembled,
           ExcInvalidStateAssembled(state));
    return psb_c_dgeasum(psblas_vector, psblas_descriptor.get());
  }



  Vector::value_type
  Vector::l2_norm() const
  {
    Assert(state == internal::State::Assembled,
           ExcInvalidStateAssembled(state));
    return psb_c_dgenrm2(psblas_vector, psblas_descriptor.get());
  }



  Vector::real_type
  Vector::norm_sqr() const
  {
    return *this * *this;
  }



  Vector::value_type
  Vector::mean_value() const
  {
    const value_type *start_ptr = psb_c_dvect_f_get_pnt(psblas_vector);

    value_type local_sum_of_values = 0.0;
    for (size_type i = 0; i < locally_owned_size(); ++i)
      local_sum_of_values += start_ptr[i];

    return Utilities::MPI::sum(local_sum_of_values, communicator) / size();
  }



  bool
  Vector::all_zero() const
  {
    // we get a pointer to the underlying vector and check if all
    // entries are zero.
    const value_type *start_ptr         = psb_c_dvect_f_get_pnt(psblas_vector);
    bool              has_nonzero_local = false;
    for (size_type i = 0; i < locally_owned_size(); ++i)
      has_nonzero_local = has_nonzero_local || start_ptr[i] != value_type();

    unsigned int num_nonzero =
      Utilities::MPI::sum(has_nonzero_local ? 1 : 0, communicator);
    return num_nonzero == 0;
  }



  bool
  Vector::is_non_negative() const
  {
    const value_type *const values = psb_c_dvect_f_get_pnt(psblas_vector);
    bool                    local_is_non_negative = true;
    for (size_type i = 0; i < locally_owned_size(); ++i)
      local_is_non_negative = local_is_non_negative && values[i] >= 0.;

    return Utilities::MPI::logical_and(local_is_non_negative, communicator);
  }



  bool
  Vector::in_local_range(const size_type index) const
  {
    return owned_elements.is_element(index);
  }



  void
  Vector::equ(const value_type a, const Vector &v)
  {
    AssertDimension(size(), v.size());
    Assert(!has_ghost_elements(), ExcGhostsPresent());
    AssertIsFinite(a);

    // do the analogous to VecAXPBY(vector, a, 0.0, v.vector) in PETSc
    int ierr = psb_c_dgeaxpby(
      a, v.psblas_vector, 0.0, psblas_vector, psblas_descriptor.get());
    AssertThrow(ierr == 0, ExcAXPBY(ierr));
  }



  Vector::size_type
  Vector::locally_owned_size() const
  {
    return owned_elements.n_elements();
  }



  Vector::value_type
  Vector::operator*(const Vector &v) const
  {
    AssertDimension(size(), v.size());
    return psb_c_dgedot(psblas_vector,
                        v.psblas_vector,
                        psblas_descriptor.get());
  }



  Vector &
  Vector::operator+=(const Vector &v)
  {
    AssertDimension(size(), v.size());
    Assert(!has_ghost_elements(), ExcGhostsPresent());
    int ierr = psb_c_dgeaxpby(value_type(1.0),
                              v.psblas_vector,
                              1.0,
                              psblas_vector,
                              psblas_descriptor.get());
    AssertThrow(ierr == 0, ExcAXPBY(ierr));
    return *this;
  }


  Vector &
  Vector::operator-=(const Vector &v)
  {
    AssertDimension(size(), v.size());
    Assert(!has_ghost_elements(), ExcGhostsPresent());
    int ierr = psb_c_dgeaxpby(value_type(-1.0),
                              v.psblas_vector,
                              1.0,
                              psblas_vector,
                              psblas_descriptor.get());
    AssertThrow(ierr == 0, ExcAXPBY(ierr));
    return *this;
  }


  Vector &
  Vector::operator*=(const value_type factor)
  {
    equ(factor, *this);
    return *this;
  }



  Vector &
  Vector::operator/=(const value_type factor)
  {
    AssertIsFinite(factor);
    Assert(factor != 0., ExcDivideByZero());
    return *this *= value_type(1.) / factor;
  }



  void
  Vector::set(const std::vector<Vector::size_type>  &indices,
              const std::vector<Vector::value_type> &values)
  {
    Assert(!has_ghost_elements(), ExcGhostsPresent());
    Assert(state != internal::State::Default, ExcInvalidState(state));
    AssertDimension(indices.size(), values.size());

    value_type *const local_values = psb_c_dvect_f_get_pnt(psblas_vector);

    for (std::size_t i = 0; i < indices.size(); ++i)
      {
        Assert(owned_elements.is_element(indices[i]),
               ExcMessage("You are trying to write to an element of the vector "
                          "that is not locally owned. This is not allowed for "
                          "the current interface to PSBLAS vectors."));
        local_values[owned_elements.index_within_set(indices[i])] = values[i];
      }
  }



  void
  Vector::add(const std::vector<Vector::size_type>  &indices,
              const std::vector<Vector::value_type> &values)
  {
    Assert(!has_ghost_elements(), ExcGhostsPresent());
    Assert(state != internal::State::Default, ExcInvalidState(state));
    Assert(indices.size() == values.size(),
           ExcMessage("Indices and values size mismatch."));

    value_type *const local_values = psb_c_dvect_f_get_pnt(psblas_vector);

    // Contributions to entries owned by another process cannot be applied
    // locally; hand them to PSBLAS, which exchanges them in compress().
    std::vector<psb_c_l_t> irw;
    std::vector<psb_c_d_t> val;

    for (std::size_t i = 0; i < indices.size(); ++i)
      if (owned_elements.is_element(indices[i]))
        local_values[owned_elements.index_within_set(indices[i])] += values[i];
      else
        {
          const auto psblas_index = static_cast<psb_c_l_t>(indices[i]);
          AssertIntegerConversion(psblas_index, indices[i]);
          irw.push_back(psblas_index);
          val.push_back(values[i]);
        }

    if (irw.empty() == false)
      {
        const auto nz = static_cast<psb_c_i_t>(irw.size());
        AssertIntegerConversion(nz, irw.size());

        const int ierr = psb_c_dgeins_options(nz,
                                              irw.data(),
                                              val.data(),
                                              psblas_vector,
                                              psblas_descriptor.get(),
                                              PSB_DUPL_ADD);
        Assert(ierr == 0, ExcInsertionInPSBLASVector(ierr));

        remote_entries_pending = true;
      }
  }



  void
  Vector::add(const value_type s, const Vector &V)
  {
    AssertIsFinite(s);
    Assert(!has_ghost_elements(), ExcGhostsPresent());
    int ierr = psb_c_dgeaxpby(s,
                              V.psblas_vector,
                              value_type(1.0),
                              psblas_vector,
                              psblas_descriptor.get());
    Assert(ierr == 0, ExcAXPBY(ierr));
  }



  void
  Vector::add(const value_type a,
              const Vector    &V,
              const value_type b,
              const Vector    &W)
  {
    sadd(1., a, V, b, W);
  }



  void
  Vector::add(const value_type s)
  {
    AssertIsFinite(s);
    Assert(!has_ghost_elements(), ExcGhostsPresent());
    value_type *start_ptr = psb_c_dvect_f_get_pnt(psblas_vector);
    for (size_type i = 0; i < locally_owned_size(); ++i)
      start_ptr[i] += s;
  }



  void
  Vector::sadd(const value_type s, const Vector &V)
  {
    Assert(!has_ghost_elements(), ExcGhostsPresent());
    AssertIsFinite(s);
    int ierr = psb_c_dgeaxpby(value_type(1.0),
                              V.psblas_vector,
                              s,
                              psblas_vector,
                              psblas_descriptor.get());
    Assert(ierr == 0, ExcAXPBY(ierr));
  }


  void
  Vector::sadd(const value_type s, const value_type a, const Vector &V)
  {
    Assert(!has_ghost_elements(), ExcGhostsPresent());
    AssertIsFinite(s);
    int ierr = psb_c_dgeaxpby(
      a, V.psblas_vector, s, psblas_vector, psblas_descriptor.get());
    Assert(ierr == 0, ExcAXPBY(ierr));
  }


  void
  Vector::sadd(const value_type s,
               const value_type a,
               const Vector    &V,
               const value_type b,
               const Vector    &W)
  {
    AssertDimension(size(), V.size());
    AssertDimension(size(), W.size());
    AssertIsFinite(a);
    AssertIsFinite(b);

    // Account for aliases before changing any entries.
    *this *= s + (&V == this ? a : 0.) + (&W == this ? b : 0.);
    if (&V != this)
      add(a, V);
    if (&W != this)
      add(b, W);
  }



  void
  Vector::sadd(const value_type s,
               const value_type a,
               const Vector    &V,
               const value_type b,
               const Vector    &W,
               const value_type c,
               const Vector    &X)
  {
    AssertDimension(size(), V.size());
    AssertDimension(size(), W.size());
    AssertDimension(size(), X.size());
    AssertIsFinite(a);
    AssertIsFinite(b);
    AssertIsFinite(c);

    *this *=
      s + (&V == this ? a : 0.) + (&W == this ? b : 0.) + (&X == this ? c : 0.);
    if (&V != this)
      add(a, V);
    if (&W != this)
      add(b, W);
    if (&X != this)
      add(c, X);
  }



  void
  Vector::scale(const Vector &v)
  {
    Assert(!has_ghost_elements(), ExcGhostsPresent());
    AssertDimension(size(), v.size());
    value_type *start_ptr   = psb_c_dvect_f_get_pnt(psblas_vector);
    value_type *start_ptr_v = psb_c_dvect_f_get_pnt(v.psblas_vector);
    for (size_type i = 0; i < locally_owned_size(); ++i)
      start_ptr[i] *= start_ptr_v[i];
  }



  Vector::value_type
  Vector::add_and_dot(const value_type a, const Vector &V, const Vector &W)
  {
    // forward the call to add()...
    add(a, V);
    // ... and then do the dot product
    return psb_c_dgedot(psblas_vector,
                        W.psblas_vector,
                        psblas_descriptor.get());
  }



  void
  Vector::swap(Vector &v)
  {
    // simply swap pointers
    std::swap(psblas_vector, v.psblas_vector);
    std::swap(psblas_context, v.psblas_context);
    std::swap(psblas_descriptor, v.psblas_descriptor);
    std::swap(ghosted, v.ghosted);
    std::swap(this->last_action, v.last_action);
    std::swap(communicator, v.communicator);
    std::swap(state, v.state);
    std::swap(remote_entries_pending, v.remote_entries_pending);
    std::swap(owned_elements, v.owned_elements);
    // We use a temp variable to swap the IndexSets
    IndexSet temp(ghost_indices);
    ghost_indices   = v.ghost_indices;
    v.ghost_indices = temp;
  }



  void
  Vector::compress(const VectorOperation::values operation)
  {
    Assert(has_ghost_elements() == false,
           ExcMessage("Calling compress() is only useful if a vector "
                      "has been written into, but this is a vector with ghost "
                      "elements and consequently is read-only. It does "
                      "not make sense to call compress() for such "
                      "vectors."));

    // We check the state of the vector...
    Assert(state != internal::State::Default, ExcInvalidDefault());
    // ... and if the last action was compatible with what we want to do
    AssertThrow(
      last_action == VectorOperation::unknown || last_action == operation,
      ExcMessage(
        "Missing compress() or calling with wrong VectorOperation argument."));

    // We check if the descriptor has already been assembled somewhere
    int ierr;
    if (!psb_c_cd_is_asb(psblas_descriptor.get()))
      {
        ierr = psb_c_cdasb(psblas_descriptor.get());
        Assert(ierr == 0, ExcAssemblePSBLASDescriptor(ierr));
      }

    // Entries belonging to this process have already been written into the
    // local storage, so only contributions destined for other processes
    // require an assembly step. Since the assembly is a collective, all
    // processes must agree on whether it is performed.
    if (Utilities::MPI::logical_or(remote_entries_pending, communicator))
      {
        Assert(operation == VectorOperation::add ||
                 operation == VectorOperation::insert,
               ExcNotImplemented());

        // Processes that only receive contributions must also enter the
        // native update state. A zero-length insertion preserves the owned
        // entries and resets the remote insertion list after the previous
        // assembly. Processes that have already inserted remote entries are
        // already in update state, so their pending contributions are kept.
        psb_c_l_t dummy_index = 0;
        psb_c_d_t dummy_value = 0.;
        ierr                = psb_c_dgeins_options(0,
                                    &dummy_index,
                                    &dummy_value,
                                    psblas_vector,
                                    psblas_descriptor.get(),
                                    PSB_DUPL_ADD);
        Assert(ierr == 0, ExcInsertionInPSBLASVector(ierr));

        ierr = psb_c_dgeasb_options(psblas_vector,
                                    psblas_descriptor.get(),
                                    operation == VectorOperation::add ?
                                      PSB_DUPL_ADD :
                                      PSB_DUPL_DEF);
        Assert(ierr == 0, ExcAssemblePSBLASVector(ierr));

        remote_entries_pending = false;
      }

    state       = internal::State::Assembled;
    last_action = VectorOperation::unknown;
  }



  void
  Vector::update_ghost_values() const
  {
    if (ghosted)
      {
        int ierr = psb_c_dhalo(psblas_vector, psblas_descriptor.get());
        Assert(ierr == 0, ExcMessage("Error while updating ghost values."));
      }
  }


  std::size_t
  Vector::memory_consumption() const
  {
    std::size_t mem = MemoryConsumption::memory_consumption(ghosted) +
                      MemoryConsumption::memory_consumption(ghost_indices);

    // mem += psb_c_sizeof(psblas_vector); TODO[MF]: missing from PSBLAS's
    // interface
    return mem;
  }


} // namespace PSCToolkitWrappers

DEAL_II_NAMESPACE_CLOSE
#endif
