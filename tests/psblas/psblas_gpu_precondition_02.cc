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

// Test AMG4PSBLAS preconditioner on the GPU (CUDA backend) with a FEM
// stiffness matrix assembled on a 2-D mesh. The stiffness matrix for a Poisson
// problem on the unit square is assembled using Q1 elements on a uniformly
// refined mesh and stored in HLG (GPU) format.

#include <deal.II/base/init_finalize.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/types.h>

#include <deal.II/distributed/fully_distributed_tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/psblas_common.h>
#include <deal.II/lac/psblas_precondition.h>
#include <deal.II/lac/psblas_sparse_matrix.h>
#include <deal.II/lac/psblas_vector.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/sparsity_tools.h>

#include <deal.II/numerics/vector_tools.h>

#include "../tests.h"

using namespace dealii;

int
main(int argc, char **argv)
{
  InitFinalize init(argc,
                    argv,
                    InitializeLibrary::MPI | InitializeLibrary::PSBLAS |
                      InitializeLibrary::PSBLASCuda);

  const MPI_Comm comm  = MPI_COMM_WORLD;
  const int      nproc = Utilities::MPI::n_mpi_processes(comm);

  AssertThrow(nproc == 2,
              ExcMessage(
                "This test must be run with exactly 2 MPI processes."));

  initlog();

  // GPU storage
  const PSCToolkitWrappers::StorageFormat mat_fmt =
    PSCToolkitWrappers::StorageFormat::cuda("HLG");
  const PSCToolkitWrappers::StorageFormat vec_fmt =
    PSCToolkitWrappers::StorageFormat::cuda();


  Triangulation<2> tria_serial;
  GridGenerator::hyper_cube(tria_serial, 0, 1);
  tria_serial.refine_global(3);
  GridTools::partition_triangulation(nproc, tria_serial);

  const TriangulationDescription::Description<2> description =
    TriangulationDescription::Utilities::create_description_from_triangulation(
      tria_serial, comm);

  parallel::fullydistributed::Triangulation<2> triangulation(comm);
  triangulation.create_triangulation(description);


  FE_Q<2>       fe(1);
  DoFHandler<2> dof_handler(triangulation);
  dof_handler.distribute_dofs(fe);

  const IndexSet locally_owned_dofs = dof_handler.locally_owned_dofs();
  IndexSet       locally_relevant_dofs;
  DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);

  AffineConstraints<double> constraints;
  constraints.reinit(locally_relevant_dofs);
  VectorTools::interpolate_boundary_values(dof_handler,
                                           0,
                                           Functions::ZeroFunction<2>(),
                                           constraints);
  constraints.close();

  PSCToolkitWrappers::SparseMatrix A;
  A.reinit(locally_owned_dofs, comm, mat_fmt);

  PSCToolkitWrappers::Vector b(locally_owned_dofs, comm, vec_fmt);

  QGauss<2>   quadrature_formula(fe.degree + 1);
  FEValues<2> fe_values(fe,
                        quadrature_formula,
                        update_values | update_gradients |
                          update_quadrature_points | update_JxW_values);

  const unsigned int dofs_per_cell = fe.dofs_per_cell;
  const unsigned int n_q_points    = quadrature_formula.size();

  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double>     cell_rhs(dofs_per_cell);
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      if (!cell->is_locally_owned())
        continue;

      fe_values.reinit(cell);
      cell_matrix = 0.0;
      cell_rhs    = 0.0;

      for (unsigned int q = 0; q < n_q_points; ++q)
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          {
            for (unsigned int j = 0; j < dofs_per_cell; ++j)
              cell_matrix(i, j) += fe_values.shape_grad(i, q) *
                                   fe_values.shape_grad(j, q) *
                                   fe_values.JxW(q);

            cell_rhs(i) += 1.0 * fe_values.shape_value(i, q) * fe_values.JxW(q);
          }

      cell->get_dof_indices(local_dof_indices);
      constraints.distribute_local_to_global(
        cell_matrix, cell_rhs, local_dof_indices, A, b);
    }

  A.compress();
  b.compress(VectorOperation::add);

  PSCToolkitWrappers::PreconditionAMG                 prec;
  PSCToolkitWrappers::PreconditionAMG::AdditionalData prec_data;
  prec_data.cycle_type  = "VCYCLE";
  prec_data.aggr_prol   = "SMOOTHED";
  prec_data.n_cycles    = 1;
  prec_data.coarse_type = "ILU";
  prec.initialize(A, prec_data);

  PSCToolkitWrappers::Vector x_sol(locally_owned_dofs, comm, vec_fmt);

  SolverControl solver_control(500, 1e-6 * b.l2_norm(), false, false);
  SolverCG<PSCToolkitWrappers::Vector> solver(solver_control);

  check_solver_within_range(solver.solve(A, x_sol, b, prec),
                            solver_control.last_step(),
                            3,
                            30);

  return 0;
}
