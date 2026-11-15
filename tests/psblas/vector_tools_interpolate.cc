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

#include <deal.II/base/function.h>
#include <deal.II/base/mpi.h>

#include <deal.II/distributed/fully_distributed_tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_description.h>

#include <deal.II/hp/mapping_collection.h>

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/psblas_block_vector.h>
#include <deal.II/lac/psblas_vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools_interpolate.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "../tests.h"

using namespace dealii;

// Use only public declaration headers: the library must supply the native
// scalar and block vector instantiations of interpolation and DataOut.

namespace
{
  constexpr double sentinel = -19.0;

  template <int dim>
  class Profile : public Function<dim>
  {
  public:
    explicit Profile(const unsigned int n_components)
      : Function<dim>(n_components)
    {}

    double
    value(const Point<dim> &point, const unsigned int component = 0) const override
    {
      double result = 1.0 + 2.0 * component;
      for (unsigned int d = 0; d < dim; ++d)
        result += (d + 1) * point[d];
      if (this->n_components > 1 && component < dim)
        result += (component + 1) * point[0] * point[0];
      return result;
    }
  };


  template <int dim, typename VectorType>
  void
  check_values(const DoFHandler<dim> &dofs,
               const MappingQ1<dim> &mapping,
               const Profile<dim>   &profile,
               const VectorType     &ghosted,
               const bool            masked = false)
  {
    const auto &fe = dofs.get_fe();
    std::vector<types::global_dof_index> indices(fe.n_dofs_per_cell());
    double                              error = 0.0;
    for (const auto &cell : dofs.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          cell->get_dof_indices(indices);
          for (unsigned int i = 0; i < indices.size(); ++i)
            {
              const auto component = fe.system_to_component_index(i).first;
              const auto point = mapping.transform_unit_to_real_cell(
                cell, fe.unit_support_point(i));
              const double expected =
                masked && component == dim ? sentinel :
                                               profile.value(point, component);
              const double actual = ghosted(indices[i]);
              error = std::isfinite(actual) ?
                        std::max(error, std::abs(actual - expected)) :
                        std::numeric_limits<double>::infinity();
            }
        }
    AssertThrow(Utilities::MPI::max(
                  error, dofs.get_triangulation().get_mpi_communicator()) <
                  1.e-12,
                ExcInternalError());
  }


  template <int dim, typename VectorType>
  void
  check_interpolation(const DoFHandler<dim> &dofs,
                      VectorType           &values,
                      VectorType           &ghosted)
  {
    const MappingQ1<dim>             mapping;
    const hp::MappingCollection<dim> mappings(mapping);
    const unsigned int              n_components = dofs.get_fe().n_components();
    const Profile<dim>              profile(n_components);
    const auto check_result = [&](const bool masked = false) {
      ghosted = values;
      ghosted.update_ghost_values();
      check_values(dofs, mapping, profile, ghosted, masked);
    };

    for (unsigned int overload = 0; overload < 3; ++overload)
      {
        values = sentinel;
        if (overload == 0)
          VectorTools::interpolate(mapping, dofs, profile, values);
        else if (overload == 1)
          VectorTools::interpolate(mappings, dofs, profile, values);
        else
          VectorTools::interpolate(dofs, profile, values);
        check_result();
      }

    if (n_components > 1)
      {
        // Update the Q2 velocity while preserving the Q1 pressure entries.
        std::vector<bool> selected(n_components, true);
        selected.back() = false;
        values          = sentinel;
        VectorTools::interpolate(
          mapping, dofs, profile, values, ComponentMask(selected));
        check_result(true);
      }

    const std::map<types::material_id, const Function<dim> *> functions = {
      {0, &profile}, {1, &profile}};
    values = sentinel;
    VectorTools::interpolate_based_on_material_id(
      mapping, dofs, functions, values);
    check_result();

    // Transfer from a separate ghosted source into the owned destination.
    FullMatrix<double> transfer(dofs.get_fe().n_dofs_per_cell());
    for (unsigned int i = 0; i < transfer.m(); ++i)
      transfer(i, i) = 1.0;
    values = sentinel;
    VectorTools::interpolate(dofs, dofs, transfer, ghosted, values);
    check_result();

    DataOut<dim> data;
    data.attach_dof_handler(dofs);
    std::vector<std::string> names(n_components, "velocity");
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
      interpretation(n_components,
                     DataComponentInterpretation::component_is_part_of_vector);
    names.back() = n_components == 1 ? "scalar" : "pressure";
    interpretation.back() = DataComponentInterpretation::component_is_scalar;
    data.add_data_vector(
      ghosted, names, DataOut<dim>::type_dof_data, interpretation);
    data.build_patches(mapping, 2);
    std::ostringstream output;
    data.write_vtu(output);
    AssertThrow(output.str().find("<VTKFile") != std::string::npos,
                ExcInternalError());
  }


  template <int dim>
  void
  run(const bool empty_rank)
  {
    const MPI_Comm     communicator = MPI_COMM_WORLD;
    const unsigned int rank = Utilities::MPI::this_mpi_process(communicator);
    const unsigned int n_processes =
      Utilities::MPI::n_mpi_processes(communicator);

    Triangulation<dim> serial;
    GridGenerator::hyper_cube(serial);
    if (!empty_rank)
      serial.refine_global(1);
    for (const auto &cell : serial.active_cell_iterators())
      {
        cell->set_subdomain_id(
          empty_rank ? n_processes - 1 :
                       cell->active_cell_index() * n_processes /
                         serial.n_active_cells());
        cell->set_material_id(cell->active_cell_index() % 2);
      }

    parallel::fullydistributed::Triangulation<dim> triangulation(communicator);
    triangulation.create_triangulation(
      TriangulationDescription::Utilities::create_description_from_triangulation(
        serial, communicator));

    {
      const FE_Q<dim>  fe(1);
      DoFHandler<dim> dofs(triangulation);
      dofs.distribute_dofs(fe);
      const auto owned    = dofs.locally_owned_dofs();
      const auto relevant = DoFTools::extract_locally_relevant_dofs(dofs);
      if (empty_rank && rank + 1 < n_processes)
        AssertThrow(owned.n_elements() == 0, ExcInternalError());
      PSCToolkitWrappers::Vector values(owned, communicator);
      PSCToolkitWrappers::Vector ghosted(owned, relevant, communicator);
      check_interpolation(dofs, values, ghosted);
      if (rank == 0)
        deallog << dim << "d " << (empty_rank ? "empty-rank" : "distributed")
                << " scalar interpolation OK" << std::endl;
    }

    {
      const FESystem<dim> fe(FE_Q<dim>(2), dim, FE_Q<dim>(1), 1);
      DoFHandler<dim>     dofs(triangulation);
      dofs.distribute_dofs(fe);
      std::vector<unsigned int> component_blocks(dim + 1, 0);
      component_blocks.back() = 1;
      DoFRenumbering::component_wise(dofs, component_blocks);
      const auto count = DoFTools::count_dofs_per_fe_block(dofs, component_blocks);
      const auto owned = dofs.locally_owned_dofs();
      const auto relevant = DoFTools::extract_locally_relevant_dofs(dofs);
      const std::vector<IndexSet> owned_blocks = {
        owned.get_view(0, count[0]),
        owned.get_view(count[0], count[0] + count[1])};
      const std::vector<IndexSet> relevant_blocks = {
        relevant.get_view(0, count[0]),
        relevant.get_view(count[0], count[0] + count[1])};
      if (empty_rank && rank + 1 < n_processes)
        AssertThrow(owned.n_elements() == 0, ExcInternalError());
      PSCToolkitWrappers::BlockVector values(owned_blocks, communicator);
      PSCToolkitWrappers::BlockVector ghosted(
        owned_blocks, relevant_blocks, communicator);
      check_interpolation(dofs, values, ghosted);
      if (rank == 0)
        deallog << dim << "d " << (empty_rank ? "empty-rank" : "distributed")
                << " block interpolation OK" << std::endl;
    }
  }
} // namespace


int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  mpi_initlog();
  AssertThrow(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) >= 2,
              ExcInternalError());
  run<2>(false);
  run<2>(true);
  run<3>(false);
  run<3>(true);
}
