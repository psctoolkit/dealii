// -----------------------------------------------------------------------------
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception OR LGPL-2.1-or-later
// Copyright (C) 2025 by the deal.II authors
//
// This file is part of the deal.II library.
//
// Detailed license information governing the source code and contributions
// can be found in LICENSE.md and CONTRIBUTING.md at the top level directory.
//
// -----------------------------------------------------------------------------

// Minimal sanity check for PSBLAS CUDA support. Initialize the PSBLAS GPU
// backend and verify that at least one CUDA device is visible.

#include <deal.II/base/logstream.h>

#include <psb_base_cbind.h>
#include <psb_c_base.h>

#include "../tests.h"

using namespace dealii;

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  AssertThrow(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) == 2,
              ExcMessage("This test needs to be run with 2 MPI processes."));

  MPILogInitAll log;

  // Initialize PSBLAS context (mirrors InitializeLibrary::PSBLAS)
  psb_c_ctxt *cctxt = psb_c_new_ctxt();
  psb_c_init(cctxt);
  psb_c_set_index_base(0);

#ifdef PSB_HAVE_CUDA
  // Initialize the CUDA backend
  psb_c_cuda_init(cctxt);

  // number of visible CUDA devices
  const psb_m_t n_devices = psb_c_cuda_getDeviceCount();

  AssertThrow(n_devices >= 1,
              ExcMessage("Expected at least one CUDA device, but got " +
                         std::to_string(n_devices) + "."));

  deallog << "CUDA device count >= 1: OK" << std::endl;

  psb_c_cuda_exit();
#else
  deallog << "PSB_HAVE_CUDA not defined, skipping GPU device check"
          << std::endl;
#endif

  return 0;
}
