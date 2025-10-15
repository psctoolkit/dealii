// ------------------------------------------------------------------------
//
// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (C) 2017 - 2022 by the deal.II authors
//
// This file is part of the deal.II library.
//
// Part of the source code is dual licensed under Apache-2.0 WITH
// LLVM-exception OR LGPL-2.1-or-later. Detailed license information
// governing the source code and code contributions can be found in
// LICENSE.md and CONTRIBUTING.md at the top level directory of deal.II.
//
// ------------------------------------------------------------------------

#include <deal.II/base/exception_macros.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/types.h>

#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/numerics/data_out.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/lac/petsc_precondition.h>
#include <deal.II/lac/petsc_sparse_matrix.h>
#include <deal.II/lac/petsc_vector.h>
#include <deal.II/lac/psblas_sparse_matrix.h>
#include <deal.II/lac/psblas_vector.h>
#include <deal.II/lac/psblas_precondition.h>

#include <deal.II/lac/solver_control.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <psb_c_dbase.h>

#include <iostream>

#include "../../tests/tests.h"

using namespace dealii;

// Test the consistency of the PSBLAS matrix and vector classes. We check that
// the mat-vec product with a rhs vector is identical to a PETSc one.

int main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  MPI_Comm                         mpi_communicator = MPI_COMM_WORLD;
  // AssertThrow(Utilities::MPI::n_mpi_processes(mpi_communicator) == 2,
  //             ExcMessage("This test needs to be run with 2 MPI processes."));

  mpi_initlog(10);

  const unsigned int dim = 2;

  // Create distributed triangulation of the unit square
  Triangulation<dim, dim> tria_base;

  // Create a serial triangulation of unit cube
  GridGenerator::hyper_cube(tria_base, 0, 1);
  tria_base.refine_global(6);

  // Partition
  GridTools::partition_triangulation(
    Utilities::MPI::n_mpi_processes(mpi_communicator), tria_base);

  // Create building blocks:
  const TriangulationDescription::Description<dim, dim> description =
    TriangulationDescription::Utilities::create_description_from_triangulation(
      tria_base, mpi_communicator);

  // Create a fully distributed triangulation:
  parallel::fullydistributed::Triangulation<dim, dim> triangulation(
    mpi_communicator);
  triangulation.create_triangulation(description);

  // Finite element and DoFHandler
  FE_Q<dim>       fe(1);
  DoFHandler<dim> dof_handler(triangulation);
  dof_handler.distribute_dofs(fe);

  IndexSet locally_owned_dofs = dof_handler.locally_owned_dofs();

  IndexSet locally_relevant_dofs;
  DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);

  AffineConstraints<double> constraints;
  constraints.clear();
  VectorTools::interpolate_boundary_values(dof_handler,
                                           types::boundary_id(0),
                                           Functions::ZeroFunction<dim>(),
                                           constraints);

  constraints.close();

  PSCToolkit::SparseMatrix psblas_matrix;
  psblas_matrix.reinit(locally_owned_dofs, mpi_communicator);

  PSCToolkit::Vector psblas_rhs_vector(locally_owned_dofs, mpi_communicator);
  PETScWrappers::MPI::Vector petsc_test_vector;
  petsc_test_vector.reinit(locally_owned_dofs, mpi_communicator);

  PETScWrappers::MPI::SparseMatrix petsc_matrix;
  DynamicSparsityPattern           dsp(locally_relevant_dofs);

  DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false);
  SparsityTools::distribute_sparsity_pattern(dsp,
                                             locally_owned_dofs,
                                             mpi_communicator,
                                             locally_relevant_dofs);

  petsc_matrix.reinit(locally_owned_dofs,
                      locally_owned_dofs,
                      dsp,
                      mpi_communicator);

  QGauss<dim>   quadrature_formula(fe.degree + 1);
  FEValues<dim> fe_values(fe,
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
      if (cell->is_locally_owned())
        {
          fe_values.reinit(cell);

          cell_matrix = 0.;
          cell_rhs    = 0.;

          for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
            {
              for (unsigned int i = 0; i < dofs_per_cell; ++i)
                {
                  for (unsigned int j = 0; j < dofs_per_cell; ++j)
                    cell_matrix(i, j) += fe_values.shape_grad(i, q_point) *
                                         fe_values.shape_grad(j, q_point) *
                                         fe_values.JxW(q_point);

                  cell_rhs(i) += 1. * fe_values.shape_value(i, q_point) *
                                 fe_values.JxW(q_point);
                }
            }

          cell->get_dof_indices(local_dof_indices);

          constraints.distribute_local_to_global(cell_matrix,
                                                 cell_rhs,
                                                 local_dof_indices,
                                                 psblas_matrix,
                                                 psblas_rhs_vector);


          constraints.distribute_local_to_global(cell_matrix,
                                                 cell_rhs,
                                                 local_dof_indices,
                                                 petsc_matrix,
                                                 petsc_test_vector);
        }
    }
  petsc_test_vector.compress(VectorOperation::add);
  petsc_matrix.compress(VectorOperation::add);
  psblas_matrix.compress();
  psblas_rhs_vector.compress();


  // Test matrix-vector product with PSBLAS
  PSCToolkit::Vector dst_psblas(locally_owned_dofs, mpi_communicator);
  psblas_matrix.vmult(dst_psblas, psblas_rhs_vector);

  // ... with PETSc
  PETScWrappers::MPI::Vector dst_petsc(locally_owned_dofs, mpi_communicator);
  petsc_matrix.vmult(dst_petsc, petsc_test_vector);

  double difference = 0.;
  for (const types::global_dof_index idx : locally_owned_dofs)
    difference += std::fabs(dst_petsc(idx) - dst_psblas(idx));

  AssertThrow(Utilities::MPI::sum(difference, mpi_communicator) < 1e-15,
              ExcMessage("Error too large."));

  // Finally, we test the solver

  SolverControl                solver_control(1000,
                               1e-7 * psblas_rhs_vector.l2_norm(),
                               true,
                               true);
  SolverCG<PSCToolkit::Vector> solver(solver_control);
  PSCToolkit::Vector           solution(locally_owned_dofs, mpi_communicator);

  deallog << "------- SOLVING WITH CG ONLY -------" << std::endl;
  if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
    std::cout << "Number of DoFs: " << dof_handler.n_dofs() << std::endl;
  if (dof_handler.n_dofs() < 1e5)
    solver.solve(psblas_matrix,
                 solution,
                 psblas_rhs_vector,
                 PreconditionIdentity());

  //  Use AMG preconditioner from PSBLAS
  PSCToolkit::Vector solution_amg;
  solution_amg.reinit(locally_owned_dofs, mpi_communicator);
  PSCToolkit::PreconditionAMG                          preconditioner;
  typename PSCToolkit::PreconditionAMG::AdditionalData prec_data;
  prec_data.cycle_type            = "VCYCLE";
  prec_data.aggr_prol             = "SMOOTHED";
  prec_data.smoother_sweeps       = 3;
  prec_data.n_cycles              = 1;
  prec_data.aggregation_threshold = 1e-2;
  prec_data.smoother_type         = "JACOBI";
  // prec_data.smoother_degree       = 1;
  prec_data.coarse_type = "ILU";
  preconditioner.initialize(psblas_matrix, prec_data);
  deallog << "------- SOLVING WITH AMG PRECONDITIONER -------" << std::endl;
  solver.solve(psblas_matrix, solution_amg, psblas_rhs_vector, preconditioner);

  solution_amg -= solution;
  deallog << "Norm of difference: " << solution_amg.l2_norm() << std::endl;



  // Output solution
  // PSCToolkit::Vector locally_relevant_solution(locally_owned_dofs,
  //                                              locally_relevant_dofs,
  //                                              mpi_communicator);
  // locally_relevant_solution = solution;
  // DataOut<dim> data_out;
  // data_out.attach_dof_handler(dof_handler);
  // data_out.add_data_vector(locally_relevant_solution, "u");

  // Vector<float> subdomain(triangulation.n_active_cells());
  // for (unsigned int i = 0; i < subdomain.size(); ++i)
  //   subdomain(i) = triangulation.locally_owned_subdomain();
  // data_out.add_data_vector(subdomain, "subdomain");

  // data_out.build_patches();

  // data_out.write_vtu_with_pvtu_record(
  //   "./", "solution", 0, mpi_communicator, 2, 8);

  // //  Use AMG preconditioner from PETSc
  // PETScWrappers::MPI::Vector solution_amg_petsc;
  // solution_amg_petsc.reinit(locally_owned_dofs, mpi_communicator);
  // PETScWrappers::PreconditionBoomerAMG preconditioner_petsc;
  // preconditioner_petsc.initialize(petsc_matrix);
  // deallog << "------- SOLVING WITH AMG PRECONDITIONER (PETSc) -------"
  //         << std::endl;
  // SolverCG<PETScWrappers::MPI::Vector> solver_petsc(solver_control);
  // solver_petsc.solve(petsc_matrix,
  //                    solution_amg_petsc,
  //                    petsc_test_vector,
  //                    preconditioner_petsc);

  // solution_petsc -= solution;
  // deallog << "Norm of difference: " << solution_amg.l2_norm() << std::endl;



  return 0;
}
