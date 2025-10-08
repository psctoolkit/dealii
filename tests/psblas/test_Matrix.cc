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

#include <deal.II/lac/psblas_sparse_matrix.h>
#include <deal.II/lac/psctoolkit.h>

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
  const unsigned int nproc = Utilities::MPI::n_mpi_processes(mpi_communicator);
  const int          iam   = Utilities::MPI::this_mpi_process(mpi_communicator);

  std::string   ofname = "output_" + std::to_string(iam);
  std::ofstream output(ofname);

  const unsigned int dim = 2;

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

  // Assemble system
  QGauss<dim>   quadrature_formula(fe.degree + 1);
  FEValues<dim> fe_values(fe,
                          quadrature_formula,
                          update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);

  const unsigned int dofs_per_cell = fe.dofs_per_cell;
  const unsigned int n_q_points    = quadrature_formula.size();

  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
  AffineConstraints<double>            constraints;

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      if (cell->is_locally_owned())
        {
          fe_values.reinit(cell);
          cell_matrix = 0;

          for (unsigned int q = 0; q < n_q_points; ++q)
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              for (unsigned int j = 0; j < dofs_per_cell; ++j)
                cell_matrix(i, j) += fe_values.shape_grad(i, q) *
                                     fe_values.shape_grad(j, q) *
                                     fe_values.JxW(q);

          cell->get_dof_indices(local_dof_indices);
          constraints.distribute_local_to_global(cell_matrix,
                                                 local_dof_indices,
                                                 psblas_matrix);
        }
    }
  psblas_matrix.compress();


  output << "Process " << iam << " of  " << nproc
         << " I have assembled a matrix of  " << psblas_matrix.m() << " x "
         << psblas_matrix.n() << " size with "
         << psblas_matrix.n_nonzero_elements() << " non-zero entries."
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

  // Clean up
  // info = PSCToolkit::Matrix::FreeSparseMatrix(psblas_sparse_matrix,
  // descriptor); if (info != 0)
  //   {
  //     deallog << "Error freeing PSBLAS sparse matrix: " << info << std::endl;
  //   }
  // // Free the PSBLAS descriptor
  // info = PSCToolkit::Communicator::DescriptorFree(descriptor);
  // if (info != 0)
  //   {
  //     deallog << "Error freeing PSBLAS descriptor: " << info << std::endl;
  //   }
  // return info;
}