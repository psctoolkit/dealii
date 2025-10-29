/* ------------------------------------------------------------------------
 *
 * SPDX-License-Identifier: LGPL-2.1-or-later
 * Copyright (C) 2010 - 2024 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * Part of the source code is dual licensed under Apache-2.0 WITH
 * LLVM-exception OR LGPL-2.1-or-later. Detailed license information
 * governing the source code and code contributions can be found in
 * LICENSE.md and CONTRIBUTING.md at the top level directory of deal.II.
 *
 * ------------------------------------------------------------------------
 *
 */


#include <deal.II/base/exception_macros.h>
#include <deal.II/base/init_finalize.h>
#include <deal.II/base/patterns.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/parsed_function.h>
#include <deal.II/base/tensor_function.h>

#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/numerics/vector_tools_interpolate.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/base/utilities.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/base/logstream.h>

#define USE_PSBLAS

#ifdef USE_PSBLAS
#  include <deal.II/lac/psblas_precondition.h>
#  include <deal.II/lac/psblas_sparse_matrix.h>
#  include <deal.II/lac/psblas_vector.h>
#else

namespace LA
{
#  if defined(DEAL_II_WITH_PETSC) && !defined(DEAL_II_PETSC_WITH_COMPLEX) && \
    !(defined(DEAL_II_WITH_TRILINOS) && defined(FORCE_USE_OF_TRILINOS))
  using namespace dealii::LinearAlgebraPETSc;
#    define USE_PETSC_LA
#  elif defined(DEAL_II_WITH_TRILINOS)
  using namespace dealii::LinearAlgebraTrilinos;
#  else
#    error DEAL_II_WITH_PETSC or DEAL_II_WITH_TRILINOS required
#  endif
} // namespace LA

#endif

#include <fstream>
#include <iostream>

// Constants for the conductivity tensor and angle
static constexpr double epsilon_conductivity = 100.0;
static constexpr double theta                = M_PI / 6.0;

namespace Benchmark
{
  using namespace dealii;


  template <int dim>
  class ConductivityTensor : public TensorFunction<2, dim>
  {
  public:
    ConductivityTensor()
      : TensorFunction<2, dim>()
    {}

    virtual void value_list(const std::vector<Point<dim>> &points,
                            std::vector<Tensor<2, dim>> &values) const override;
  };



  template <int dim>
  void
  ConductivityTensor<dim>::value_list(const std::vector<Point<dim>> &points,
                                      std::vector<Tensor<2, dim>> &values) const
  {
    (void)points;
    AssertDimension(dim, 2); // make sure we call this only in 2D
    AssertDimension(points.size(), values.size());

    Tensor<2, dim> K;
    K[0][0] = std::cos(theta) * std::cos(theta) +
              epsilon_conductivity * std::sin(theta) * std::sin(theta);
    K[0][1] = (1.0 - epsilon_conductivity) * std::sin(theta) * std::cos(theta);
    K[1][0] = K[0][1];
    K[1][1] = std::sin(theta) * std::sin(theta) +
              epsilon_conductivity * std::cos(theta) * std::cos(theta);

    for (auto &value : values)
      value = K;
  }



  struct AMGParameters
  {
    std::string  cycle_type;
    unsigned int n_cycles;
    std::string  smoother_type;
    std::string  aggregation_type;
    unsigned int aggregation_size;
    double       aggregation_threshold;
    std::string  parallel_aggregation_algorithm;
    std::string  prolongator_aggregation;
    std::string  aggregation_filter;
    unsigned int smoother_sweeps;
    std::string  coarse_solver;
    std::string  coarse_subsolver;
    std::string  coarse_mat_type;
    bool         verbose_amg_info;

    void declare_parameters(ParameterHandler &param)
    {
      param.declare_entry("Cycle type",
                          "VCYCLE",
                          Patterns::Selection("VCYCLE|WCYCLE"));
      param.declare_entry("Number of cycles", "1", Patterns::Integer());
      param.declare_entry("Smoother type",
                          "FBGS",
                          Patterns::Selection("FBGS|JACOBI|BJAC|L1-JACOBI"));
      param.declare_entry("Smoother sweeps", "2", Patterns::Integer());
      param.declare_entry("Aggregation type",
                          "SOC1",
                          Patterns::Selection("SOC1|SOC2|MATCHBOXP"));
      param.declare_entry("Aggregation size", "4", Patterns::Integer());
      param.declare_entry("Aggregation threshold", "1e-2", Patterns::Double());
      param.declare_entry("Prolongator aggregation",
                          "SMOOTHED",
                          Patterns::Selection("SMOOTHED|UNSMOOTHED"));
      param.declare_entry("Aggregation filter",
                          "FILTER",
                          Patterns::Selection("FILTER|NOFILTER"));
      param.declare_entry("Parallel aggregation algorithm",
                          "DECOUPLED",
                          Patterns::Selection("DECOUPLED|COUPLED"));
      param.declare_entry("Coarse solver",
                          "BJAC",
                          Patterns::Selection("BJAC|ILU|MUMPS|UMF|SLUDIST"));
      param.declare_entry("Coarse subsolver",
                          "ILU",
                          Patterns::Selection("ILU"),
                          "Type of solver used on the coarse level of the AMG");
      param.declare_entry("Coarse matrix type",
                          "DIST",
                          Patterns::Selection("DIST|REPL"));
      param.declare_entry("Verbose AMG info",
                          "false",
                          Patterns::Bool(),
                          "Whether to print detailed information during the "
                          "AMG setup and solve phases.");
    }

    void parse_parameters(ParameterHandler &param)
    {
      cycle_type              = param.get("Cycle type");
      n_cycles                = param.get_integer("Number of cycles");
      smoother_type           = param.get("Smoother type");
      smoother_sweeps         = param.get_integer("Smoother sweeps");
      aggregation_type        = param.get("Aggregation type");
      aggregation_size        = param.get_integer("Aggregation size");
      prolongator_aggregation = param.get("Prolongator aggregation");
      aggregation_filter      = param.get("Aggregation filter");
      parallel_aggregation_algorithm =
        param.get("Parallel aggregation algorithm");
      aggregation_threshold = param.get_double("Aggregation threshold");
      coarse_solver         = param.get("Coarse solver");
      coarse_subsolver      = param.get("Coarse subsolver");
      coarse_mat_type       = param.get("Coarse matrix type");
      verbose_amg_info      = param.get_bool("Verbose AMG info");
    }
  };



  template <int dim>
  class LaplaceProblem
  {
  public:
    class Parameters : public ParameterAcceptor
    {
    public:
      Parameters();

      std::list<types::boundary_id> dirichlet_ids{0, 1, 2, 3};

      // Level of verbosity to use in the output
      unsigned int verbosity_level = 10;

      // Whether to use the deal.II solver or the PSBLAS solver
      bool use_dealii_solver = true;

      bool adaptive_refinement = false;

      unsigned int n_refinement_cycles = 10;

      // A flag to keep track if we were initialized or not
      bool initialized = false;
    };

    LaplaceProblem(const Parameters &);

    void run();

  private:
    void setup_system();
    void assemble_system();
    void solve();
    void refine_grid();
    void output_results(const unsigned int cycle);

    MPI_Comm mpi_communicator;

    parallel::distributed::Triangulation<dim> triangulation;

    const FE_Q<dim> fe;
    DoFHandler<dim> dof_handler;

    IndexSet locally_owned_dofs;
    IndexSet locally_relevant_dofs;

    AffineConstraints<double> constraints;

#ifdef USE_PSBLAS
    PSCToolkit::SparseMatrix system_matrix;
    PSCToolkit::Vector       locally_relevant_solution;
    PSCToolkit::Vector       system_rhs;
#else
    LA::MPI::SparseMatrix  system_matrix;
    LA::MPI::Vector        locally_relevant_solution;
    LA::MPI::Vector        system_rhs;

#endif

    ConditionalOStream pcout;
    TimerOutput        computing_timer;
    const Parameters  &parameters;

    ParameterAcceptorProxy<Functions::ParsedFunction<dim>> rhs_function;
    ParameterAcceptorProxy<ReductionControl>               solver_parameters;
    ParameterAcceptorProxy<AMGParameters>                  AMG_control;
  };



  template <int dim>
  LaplaceProblem<dim>::Parameters::Parameters()
    : ParameterAcceptor("/Laplace Problem<" + Utilities::int_to_string(dim) +
                        ">/")
  {
    add_parameter("Dirichlet boundary ids", dirichlet_ids);
    add_parameter("Verbosity level", verbosity_level);
    add_parameter("Use deal.II solver", use_dealii_solver);
    add_parameter("Perform adaptive refinement", adaptive_refinement);
    add_parameter("Number of refinement cycles", n_refinement_cycles);

    // Once the parameter file has been parsed, then the parameters are good to
    // go. Set the internal variable `initialized` to true.
    parse_parameters_call_back.connect([&]() -> void { initialized = true; });
  }


  template <int dim>
  LaplaceProblem<dim>::LaplaceProblem(const Parameters &param)
    : mpi_communicator(MPI_COMM_WORLD)
    , triangulation(mpi_communicator,
                    typename Triangulation<dim>::MeshSmoothing(
                      Triangulation<dim>::smoothing_on_refinement |
                      Triangulation<dim>::smoothing_on_coarsening))
    , fe(1)
    , dof_handler(triangulation)
    , pcout(std::cout,
            (Utilities::MPI::this_mpi_process(mpi_communicator) == 0))
    , computing_timer(mpi_communicator,
                      pcout,
                      TimerOutput::never,
                      TimerOutput::wall_times)
    , parameters(param)
    , rhs_function("Rhs function", 1)
    , solver_parameters("Solver parameters")
    , AMG_control("AMG control")
  {
    rhs_function.declare_parameters_call_back.connect([]() -> void {
      ParameterAcceptor::prm.set("Function expression", "1.");
    });

    solver_parameters.declare_parameters_call_back.connect([]() -> void {
      ParameterAcceptor::prm.set("Max steps", "1000");
      ParameterAcceptor::prm.set("Reduction", "1.e-12");
      ParameterAcceptor::prm.set("Tolerance", "1.e-12");
    });

    AMG_control.declare_parameters_call_back.connect([]() -> void {
      ParameterAcceptor::prm.set("Cycle type", "VCYCLE");
      ParameterAcceptor::prm.set("Smoother sweeps", "2");
      ParameterAcceptor::prm.set("Coarse solver", "BJAC");
      ParameterAcceptor::prm.set("Coarse subsolver", "ILU");
    });
  }



  template <int dim>
  void LaplaceProblem<dim>::setup_system()
  {
    TimerOutput::Scope t(computing_timer, "setup");

    dof_handler.distribute_dofs(fe);

    pcout << "   Number of active cells:       "
          << triangulation.n_global_active_cells() << std::endl
          << "   Number of degrees of freedom: " << dof_handler.n_dofs()
          << std::endl;


    locally_owned_dofs = dof_handler.locally_owned_dofs();
    locally_relevant_dofs =
      DoFTools::extract_locally_relevant_dofs(dof_handler);

    locally_relevant_solution.reinit(locally_owned_dofs,
                                     locally_relevant_dofs,
                                     mpi_communicator);
    system_rhs.reinit(locally_owned_dofs, mpi_communicator);

    constraints.clear();
    constraints.reinit(locally_owned_dofs, locally_relevant_dofs);
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);

    for (const types::boundary_id id : parameters.dirichlet_ids)
      {
        VectorTools::interpolate_boundary_values(dof_handler,
                                                 id,
                                                 Functions::ZeroFunction<dim>(),
                                                 constraints);
      }
    constraints.close();


#ifdef USE_PSBLAS
    PSCToolkit::SparsityPattern sparsity_pattern(locally_owned_dofs,
                                                 mpi_communicator);
    system_matrix.reinit(sparsity_pattern, mpi_communicator);
    // system_matrix.reinit(locally_owned_dofs, mpi_communicator);
#else
    DynamicSparsityPattern dsp(locally_relevant_dofs);

    DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false);
    SparsityTools::distribute_sparsity_pattern(dsp,
                                               dof_handler.locally_owned_dofs(),
                                               mpi_communicator,
                                               locally_relevant_dofs);
    system_matrix.reinit(locally_owned_dofs,
                         locally_owned_dofs,
                         dsp,
                         mpi_communicator);
#endif
  }



  template <int dim>
  void LaplaceProblem<dim>::assemble_system()
  {
    TimerOutput::Scope t(computing_timer, "assembly");

    const QGauss<dim> quadrature_formula(fe.degree + 1);

    FEValues<dim> fe_values(fe,
                            quadrature_formula,
                            update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);

    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    const unsigned int n_q_points    = quadrature_formula.size();

    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double>     cell_rhs(dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    const ConductivityTensor<dim> conductivity_tensor;
    std::vector<Tensor<2, dim>>   conductivity_values(n_q_points);
    std::vector<double>           rhs_values(n_q_points);

    for (const auto &cell : dof_handler.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          fe_values.reinit(cell);

          cell_matrix = 0.;
          cell_rhs    = 0.;

          if constexpr (dim == 2)
            {
              conductivity_tensor.value_list(fe_values.get_quadrature_points(),
                                             conductivity_values);
            }

          rhs_function.value_list(fe_values.get_quadrature_points(),
                                  rhs_values);

          for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
            {
              for (unsigned int i = 0; i < dofs_per_cell; ++i)
                {
                  for (unsigned int j = 0; j < dofs_per_cell; ++j)
                    cell_matrix(i, j) +=
                      fe_values.shape_grad(i, q_point) *
                      (dim == 2 ? conductivity_values[q_point] :
                                  unit_symmetric_tensor<dim>()) *
                      fe_values.shape_grad(j, q_point) * fe_values.JxW(q_point);

                  cell_rhs(i) += rhs_values[q_point] *
                                 fe_values.shape_value(i, q_point) *
                                 fe_values.JxW(q_point);
                }
            }

          cell->get_dof_indices(local_dof_indices);
          constraints.distribute_local_to_global(cell_matrix,
                                                 cell_rhs,
                                                 local_dof_indices,
                                                 system_matrix,
                                                 system_rhs);
        }
#ifdef USE_PSBLAS
    system_matrix.compress();
#else
    system_matrix.compress(VectorOperation::add);
#endif
    system_rhs.compress(VectorOperation::add);
  }



  template <int dim>
  void LaplaceProblem<dim>::solve()
  {
    TimerOutput::Scope t(computing_timer, "solve");

#ifdef USE_PSBLAS
    PSCToolkit::Vector completely_distributed_solution(locally_owned_dofs,
                                                       mpi_communicator);


    if (parameters.use_dealii_solver == true)
      {
        SolverCG<PSCToolkit::Vector> solver(solver_parameters);

        typename PSCToolkit::PreconditionAMG::AdditionalData prec_data;
        prec_data.cycle_type            = AMG_control.cycle_type.c_str();
        prec_data.smoother_type         = AMG_control.smoother_type.c_str();
        prec_data.smoother_sweeps       = AMG_control.smoother_sweeps;
        prec_data.coarse_type           = AMG_control.coarse_solver.c_str();
        prec_data.coarse_mat_type       = AMG_control.coarse_mat_type.c_str();
        prec_data.output_details        = AMG_control.verbose_amg_info;
        prec_data.aggregation_type      = AMG_control.aggregation_type.c_str();
        prec_data.aggregation_size      = AMG_control.aggregation_size;
        prec_data.aggregation_threshold = AMG_control.aggregation_threshold;
        prec_data.aggr_prol   = AMG_control.prolongator_aggregation.c_str();
        prec_data.aggr_filter = AMG_control.aggregation_filter.c_str();
        prec_data.parallel_aggr_algorithm =
          AMG_control.parallel_aggregation_algorithm.c_str();
        PSCToolkit::PreconditionAMG preconditioner;
        preconditioner.initialize(system_matrix, prec_data);

        solver.solve(system_matrix,
                     completely_distributed_solution,
                     system_rhs,
                     preconditioner);

        pcout << "   Solved with AMG4PSBLAS in "
              << solver_parameters.last_step() << " iterations." << std::endl;
      }
    else
      {
        char                ptype[40];
        psb_c_SolverOptions options;
        psb_c_descriptor   *cdh = system_matrix.get_psblas_descriptor();
        double              t1, t2, err;
        double              one = 1.0, zero = 0.0, res2;
        int                 info, iter, ret;
        strcpy(ptype, "ML");

        psb_c_ctxt  *cctxt = InitFinalize::get_psblas_context();
        amg_c_dprec *ph    = amg_c_dprec_new();
        amg_c_dprecinit(*cctxt, ph, ptype);

        amg_c_dprecsetc(ph, "ML_CYCLE", AMG_control.cycle_type.c_str());
        // smoothers
        amg_c_dprecsetc(ph, "SMOOTHER_TYPE", AMG_control.smoother_type.c_str());
        amg_c_dprecseti(ph, "SMOOTHER_SWEEPS", AMG_control.smoother_sweeps);
        amg_c_dprecseti(ph, "CYCLE_SWEEPS", AMG_control.n_cycles);

        // aggregation parameters
        amg_c_dprecsetr(ph, "AGGR_THRSH", AMG_control.aggregation_threshold);
        amg_c_dprecsetc(ph,
                        "PAR_AGGR_ALG",
                        AMG_control.parallel_aggregation_algorithm.c_str());
        amg_c_dprecsetc(ph, "AGGR_TYPE", AMG_control.aggregation_type.c_str());
        amg_c_dprecseti(ph, "AGGR_SIZE", AMG_control.aggregation_size);
        amg_c_dprecsetc(ph,
                        "AGGR_PROL",
                        AMG_control.prolongator_aggregation.c_str());
        amg_c_dprecsetc(ph,
                        "AGGR_FILTER",
                        AMG_control.aggregation_filter.c_str());

        // coarse solvers

        amg_c_dprecsetc(ph, "COARSE_SOLVE", AMG_control.coarse_solver.c_str());
        amg_c_dprecsetc(ph, "COARSE_MAT", AMG_control.coarse_mat_type.c_str());

        if ((ret = amg_c_dhierarchy_build(system_matrix.get_psblas_matrix(),
                                          system_matrix.get_psblas_descriptor(),
                                          ph)) != 0)
          fprintf(stderr, "From hierarchy_build: %d\n", ret);
        if ((ret = amg_c_dsmoothers_build(system_matrix.get_psblas_matrix(),
                                          system_matrix.get_psblas_descriptor(),
                                          ph)) != 0)
          fprintf(stderr, "From smoothers_build: %d\n", ret);

        psb_c_barrier(*cctxt);
        psb_c_DefaultSolverOptions(&options);
        int istop      = 2;
        int itmax      = solver_parameters.max_steps();
        options.eps    = solver_parameters.reduction();
        options.itmax  = itmax;
        options.itrace = solver_parameters.log_history() ? 1 : 0;
        options.istop  = istop; // scaled 2-norm of the residual
        psb_c_seterraction_ret();

        if (AMG_control.verbose_amg_info == true)
          amg_c_ddescr(ph);

        t1   = psb_c_wtime();
        ret  = amg_c_dkrylov("CG",
                            system_matrix.get_psblas_matrix(),
                            ph,
                            system_rhs.get_psblas_vector(),
                            completely_distributed_solution.get_psblas_vector(),
                            cdh,
                            &options);
        t2   = psb_c_wtime();
        iter = options.iter;
        err  = options.err;
        // fprintf(stderr,
        //         "From krylov: %d %lf, %d%d\n",
        //         iter,
        //         err,
        //         ret,
        //         psb_c_get_errstatus());
        if (psb_c_get_errstatus() != 0)
          {
            psb_c_print_errmsg();
          }

        /* Check 2-norm of residual on exit */
        psb_c_dvector *rh;
        rh = psb_c_new_dvector();
        psb_c_dgeall(rh, cdh);
        if ((info = psb_c_dgeasb(rh, cdh)) != 0)
          Assert(false, ExcInternalError());

        psb_c_dgeaxpby(one, system_rhs.get_psblas_vector(), zero, rh, cdh);
        psb_c_dspmm(-one,
                    system_matrix.get_psblas_matrix(),
                    completely_distributed_solution.get_psblas_vector(),
                    one,
                    rh,
                    cdh);
        res2 = psb_c_dgenrm2(rh, cdh);

        if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
          {
            fprintf(stdout, "Time: %lf\n", (t2 - t1));
            fprintf(stdout, "Iter: %d\n", iter);
            fprintf(stdout, "Err: %lg\n", err);
            fprintf(stdout, "||r||_2: %lg\n", res2);
          }
      }

    constraints.distribute(completely_distributed_solution);
    locally_relevant_solution = completely_distributed_solution;

#else

    LA::MPI::Vector completely_distributed_solution(locally_owned_dofs,
                                                    mpi_communicator);

    SolverControl solver_control(solver_parameters);
    LA::SolverCG  solver(solver_control);

    LA::MPI::PreconditionAMG::AdditionalData data;
#  ifdef USE_PETSC_LA
    data.symmetric_operator = true;
#  else
    /* Trilinos defaults are good */
#  endif
    LA::MPI::PreconditionAMG preconditioner;
    preconditioner.initialize(system_matrix, data);

    solver.solve(system_matrix,
                 completely_distributed_solution,
                 system_rhs,
                 preconditioner);

#  ifdef USE_PETSC_LA
    std::string solver_name = "PETSc";
#  else
    std::string solver_name = "Trilinos";
#  endif
    pcout << "   Solved with " << solver_name << " in "
          << solver_control.last_step() << " iterations." << std::endl;

    constraints.distribute(completely_distributed_solution);

    locally_relevant_solution = completely_distributed_solution;
#endif
  }



  template <int dim>
  void LaplaceProblem<dim>::refine_grid()
  {
    TimerOutput::Scope t(computing_timer, "refine");
    if (parameters.adaptive_refinement == true)
      {
        Vector<float> estimated_error_per_cell(triangulation.n_active_cells());
        KellyErrorEstimator<dim>::estimate(
          dof_handler,
          QGauss<dim - 1>(fe.degree + 1),
          std::map<types::boundary_id, const Function<dim> *>(),
          locally_relevant_solution,
          estimated_error_per_cell);
        parallel::distributed::GridRefinement::refine_and_coarsen_fixed_number(
          triangulation, estimated_error_per_cell, 0.3, 0.03);
        triangulation.execute_coarsening_and_refinement();
      }
    else
      {
        triangulation.refine_global(1);
      }
  }



  template <int dim>
  void LaplaceProblem<dim>::output_results(const unsigned int cycle)
  {
    TimerOutput::Scope t(computing_timer, "output");

    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(locally_relevant_solution, "u");

    Vector<float> subdomain(triangulation.n_active_cells());
    for (unsigned int i = 0; i < subdomain.size(); ++i)
      subdomain(i) = triangulation.locally_owned_subdomain();
    data_out.add_data_vector(subdomain, "subdomain");

    data_out.build_patches();

    data_out.write_vtu_with_pvtu_record(
      "./", "solution", cycle, mpi_communicator, 2, 8);
  }



  template <int dim>
  void LaplaceProblem<dim>::run()
  {
    pcout << "Running with "
#ifdef USE_PETSC_LA
          << "PETSc"
#elif defined(USE_TRILINOS_LA)
          << "Trilinos"
#else
          << "PSBLAS"
#endif
          << " on " << Utilities::MPI::n_mpi_processes(mpi_communicator)
          << " MPI rank(s)..." << std::endl;

    deallog.depth_console(parameters.verbosity_level);

    for (unsigned int cycle = 0; cycle < parameters.n_refinement_cycles;
         ++cycle)
      {
        pcout << "Cycle " << cycle << ':' << std::endl;

        if (cycle == 0)
          {
            if constexpr (dim == 2)
              GridGenerator::hyper_cube(triangulation, -1, 1, true);
            else if constexpr (dim == 3)
              GridGenerator::hyper_cube(triangulation, 0, 1, true);
            else
              DEAL_II_ASSERT_UNREACHABLE();

            triangulation.refine_global(5);
          }
        else
          refine_grid();

        setup_system();
        assemble_system();
        solve();
        output_results(cycle);

        computing_timer.print_summary();
        computing_timer.reset();

        pcout << std::endl;
      }
  }
} // namespace Benchmark



int main(int argc, char *argv[])
{
  try
    {
      using namespace dealii;
      using namespace Benchmark;

      Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

      LaplaceProblem<2>::Parameters parameters;
      LaplaceProblem<2>             laplace_problem(parameters);

      std::string parameter_file;
      if (argc > 1)
        parameter_file = argv[1];
      else
        parameter_file = "parameters.prm";

      ParameterAcceptor::initialize(parameter_file, "used_parameters.prm");

      laplace_problem.run();
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;
}
