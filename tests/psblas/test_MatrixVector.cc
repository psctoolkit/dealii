#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/utilities.h>

#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>

#include "deal.II/lac/affine_constraints.h"
#include <deal.II/lac/psblas_sparse_matrix.h>
#include <deal.II/lac/psblas_vector.h>

#include <deal.II/meshworker/mesh_loop.h>

#include <fstream>
#include <iostream>

#include "../tests.h"

using namespace dealii;

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  MPI_Comm                         mpi_communicator = MPI_COMM_WORLD;

  const int iam   = Utilities::MPI::this_mpi_process(mpi_communicator);
  const int nproc = Utilities::MPI::n_mpi_processes(mpi_communicator);

  std::string   ofname = "output_" + std::to_string(iam);
  std::ofstream output(ofname);

  static constexpr unsigned int dim = 2;

  // Create distributed triangulation of the unit square
  Triangulation<dim, dim> tria_base;

  // Create a serial triangulation (here by reading an external mesh):
  GridGenerator::hyper_cube(tria_base, 0, 1);
  tria_base.refine_global(5);

  // Partition
  GridTools::partition_triangulation(nproc, tria_base);

  // Create building blocks:
  const TriangulationDescription::Description<dim, dim> description =
    TriangulationDescription::Utilities::create_description_from_triangulation(
      tria_base, mpi_communicator);

  // Create a fully distributed triangulation:
  parallel::fullydistributed::Triangulation<dim, dim> triangulation(
    mpi_communicator);
  triangulation.create_triangulation(description);

  // Finite element and DoFHandler
  FE_Q<dim>       fe(dim);
  DoFHandler<dim> dof_handler(triangulation);
  dof_handler.distribute_dofs(fe);

  IndexSet locally_owned_dofs = dof_handler.locally_owned_dofs();
  IndexSet locally_relevant_dofs;
  DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);

  PSCToolkit::SparseMatrix psblas_matrix;
  psblas_matrix.reinit(locally_owned_dofs, mpi_communicator);
  PSCToolkit::Vector psblas_rhs_vector(locally_owned_dofs, mpi_communicator);

  // Assemble system
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

  AffineConstraints<double> constraints;

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      if (cell->is_locally_owned())
        {
          fe_values.reinit(cell);

          cell_matrix = 0.;
          cell_rhs    = 0.;

          for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
            {
              const double rhs_value =
                (fe_values.quadrature_point(q_point)[1] >
                     0.5 +
                       0.25 * std::sin(4.0 * numbers::PI *
                                       fe_values.quadrature_point(q_point)[0]) ?
                   1. :
                   -1.);

              for (unsigned int i = 0; i < dofs_per_cell; ++i)
                {
                  for (unsigned int j = 0; j < dofs_per_cell; ++j)
                    cell_matrix(i, j) += fe_values.shape_grad(i, q_point) *
                                         fe_values.shape_grad(j, q_point) *
                                         fe_values.JxW(q_point);

                  cell_rhs(i) += rhs_value * fe_values.shape_value(i, q_point) *
                                 fe_values.JxW(q_point);
                }
            }

          cell->get_dof_indices(local_dof_indices);

          // Distribute local indices and values to the PSBLAS sparse matrix and
          // vector
          constraints.distribute_local_to_global(cell_matrix,
                                                 cell_rhs,
                                                 local_dof_indices,
                                                 psblas_matrix,
                                                 psblas_rhs_vector);
        }
    }
  psblas_matrix.compress();
  psblas_rhs_vector.compress();



  output << "Process " << iam << " of  " << nproc
         << " I have assembled a matrix of  " << psblas_matrix.local_size()
         << " x " << psblas_matrix.n() << " size with "
         << psblas_matrix.n_nonzero_elements()
         << " non-zero entries and a vector with "
         << psblas_rhs_vector.locally_owned_size() << " entries."
         << " The locally owned dofs are: " << locally_owned_dofs.n_elements()
         << " and the locally relevant dofs are: "
         << locally_relevant_dofs.n_elements() << std::endl;
  output.close();


  // Concatenate output_i to the file "output"
  if (iam == 0)
    {
      std::ofstream final_output("output");
      for (int i = 0; i < nproc; i++)
        {
          std::string   ofname = "output_" + std::to_string(i);
          std::ifstream input(ofname);
          final_output << input.rdbuf();
          input.close();
          std::remove(ofname.c_str());
        }
    }
}
