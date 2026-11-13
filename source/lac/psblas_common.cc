// -----------------------------------------------------------------------------
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception OR LGPL-2.1-or-later
// Copyright (C) 2026 by the deal.II authors
//
// This file is part of the deal.II library.
//
// Detailed license information governing the source code and contributions
// can be found in LICENSE.md and CONTRIBUTING.md at the top level directory.
//
// -----------------------------------------------------------------------------

#include <deal.II/base/init_finalize.h>

#include <deal.II/lac/psblas_common.h>

#ifdef DEAL_II_WITH_PSBLAS

DEAL_II_NAMESPACE_OPEN

namespace PSCToolkitWrappers
{
  namespace internal
  {
    Context::Context(const MPI_Comm input_communicator)
      : context(InitFinalize::get_psblas_context())
    {
      Assert(input_communicator != MPI_COMM_NULL,
             ExcMessage("MPI_COMM_NULL is not a valid PSBLAS communicator."));
      int       comparison = MPI_UNEQUAL;
      const int error =
        MPI_Comm_compare(input_communicator, MPI_COMM_WORLD, &comparison);
      AssertThrowMPI(error);
      if (comparison != MPI_IDENT && comparison != MPI_CONGRUENT)
        {
          const int duplicate_error =
            MPI_Comm_dup(input_communicator, &communicator);
          AssertThrowMPI(duplicate_error);
          const MPI_Fint fortran_communicator = MPI_Comm_c2f(communicator);
          const psb_c_i_t  converted = static_cast<psb_c_i_t>(fortran_communicator);
          AssertIntegerConversion(converted, fortran_communicator);
          psb_c_init_from_fint(&owned_context, converted);
          context = &owned_context;
        }
    }

    Context::~Context()
    {
      if (owned_context.ctxt != nullptr)
        psb_c_exit_ctxt(owned_context);
      if (communicator != MPI_COMM_NULL)
        MPI_Comm_free(&communicator);
    }

    psb_c_ctxt *
    Context::get_psblas_context() const
    {
      return context;
    }
  } // namespace internal
} // namespace PSCToolkitWrappers

DEAL_II_NAMESPACE_CLOSE

#endif
