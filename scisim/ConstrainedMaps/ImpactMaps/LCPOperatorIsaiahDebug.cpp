//
// Created by Isaiah Witzke on 2022-10-11.
//

#include <iostream>
#include <chrono>
#include "LCPOperatorIsaiahDebug.h"
#include "LCPOperatorPIv2.h"

using namespace Utilities;

LCPOperatorIsaiahDebug::LCPOperatorIsaiahDebug(
    const std::vector<std::string> &linear_solvers,
    const scalar &tol,
    const unsigned &max_iters)
    : m_linear_solver_order(linear_solvers),
      m_tol(tol),
      max_iters(max_iters)
{
  assert(m_tol > 0.0);

  // Verify that the user provided a valid linear solver option
  if (m_linear_solver_order.empty())
  {
    std::cerr << "No linear solver provided to LCPOperatorIsaiahDebug. Exiting." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (IpoptUtilities::containsDuplicates(m_linear_solver_order))
  {
    std::cerr << "Duplicate linear solvers provided to LCPOperatorIsaiahDebug. Exiting." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  for (const std::string &solver_name : m_linear_solver_order)
  {
    if (!IpoptUtilities::linearSolverSupported(solver_name))
    {
      std::cerr << "Invalid linear solver provided to LCPOperatorIsaiahDebut: " << solver_name << ". Exiting." << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }

  ipopt_solver = std::unique_ptr<LCPOperatorIpopt>(new LCPOperatorIpopt(m_linear_solver_order, m_tol));
  policy_solver = std::unique_ptr<LCPOperatorPI>(new LCPOperatorPI(m_tol, max_iters));
  policy_solver_v2 = std::unique_ptr<LCPOperatorPIv2>(new LCPOperatorPIv2(m_tol, max_iters));
  penalty_solver = std::unique_ptr<LCPOperatorPenalty>(new LCPOperatorPenalty(m_tol, max_iters));
}

LCPOperatorIsaiahDebug::LCPOperatorIsaiahDebug(std::istream &input_stream)
    : m_linear_solver_order(StringUtilities::deserializeVector(input_stream)),
      m_tol(Utilities::deserialize<scalar>(input_stream)),
      max_iters(Utilities::deserialize<unsigned>(input_stream))
{
  assert(!m_linear_solver_order.empty());
  assert(m_tol >= 0.0);
  ipopt_solver = std::unique_ptr<LCPOperatorIpopt>(new LCPOperatorIpopt(m_linear_solver_order, m_tol));
  policy_solver = std::unique_ptr<LCPOperatorPI>(new LCPOperatorPI(m_tol, max_iters));
  policy_solver_v2 = std::unique_ptr<LCPOperatorPIv2>(new LCPOperatorPIv2(m_tol, max_iters));
  penalty_solver = std::unique_ptr<LCPOperatorPenalty>(new LCPOperatorPenalty(m_tol, max_iters));
}

void LCPOperatorIsaiahDebug::flow(
    const std::vector<std::unique_ptr<Constraint>> &cons,
    const SparseMatrixsc &M,
    const SparseMatrixsc &Minv,
    const VectorXs &q0,
    const VectorXs &v0,
    const VectorXs &v0F,
    const SparseMatrixsc &N,
    const SparseMatrixsc &Q,
    const VectorXs &nrel,
    const VectorXs &CoR,
    VectorXs &alpha)
{
  VectorXs b{N.transpose() * (1 + CoR(0)) * v0};

  VectorXs ipopt_sol = alpha;
  VectorXs policy_sol = alpha;
  VectorXs policy_sol_v2 = alpha;

  bool policy_converges = false;
  bool ipopt_converges = false;
  bool policy_v2_converges = false;


  auto ipopt_start = std::chrono::steady_clock::now();
  ipopt_solver->flow(cons, M, Minv, q0, v0, v0F, N, Q, nrel, CoR, ipopt_sol);
  if (getEndError(Q, ipopt_sol, b) <= m_tol)
    ipopt_converges = true;
  auto ipopt_end = std::chrono::steady_clock::now();

  policy_solver->target_soln = ipopt_sol;

  auto pi_start = std::chrono::steady_clock::now();
  policy_solver->flow(cons, M, Minv, q0, v0, v0F, N, Q, nrel, CoR, policy_sol);
  if (getEndError(Q, policy_sol, b) <= m_tol)
    policy_converges = true;
  auto pi_end = std::chrono::steady_clock::now();

  auto pi2_start = std::chrono::steady_clock::now();
  // v2 is in a broken state -- probably not needed anymore though!
  // policy_solver_v2->flow(cons, M, Minv, q0, v0, v0F, N, Q, nrel, CoR, policy_sol_v2);
  if (getEndError(Q, policy_sol_v2, b) <= m_tol)
    policy_v2_converges = true;
  auto pi2_end = std::chrono::steady_clock::now();

  // float K_0 = getKineticEnergy(M, v0);
  // float K_1 = getKineticEnergy(M, Minv * ipopt_sol);
  // printMatrix(M, "M");
  std::cout << "{" << std::endl;
  printSparseMatrixJson(N, "N");
  printSparseMatrixJson(Q, "Q");
  printSparseMatrixJson(Minv, "Minv");
  printVectorJson(v0, "v0");
  printVectorJson(b, "b");
  printVectorJson(ipopt_sol, "ipopt_sol");
  printVectorJson(policy_sol, "policy_sol");
  printBoolJson(ipopt_converges, "ipopt_converges");
  printBoolJson(policy_converges, "policy_converges");
  printBoolJson(policy_v2_converges, "policy_v2_converges");
  
  /*

  currently, without improvements:
  "ipopt_time": 7801312.000000000000,
  "pi_time": 1044528.000000000000,
  "pi_v2_time": 1544688.000000000000,
  */

  // output times in miliseconds
  {
    using namespace std::chrono;
    std::cout.precision(0);
    printDoubleJson(duration_cast<milliseconds>(ipopt_end - ipopt_start).count(), "ipopt_time");
    printDoubleJson(duration_cast<milliseconds>(pi_end - pi_start).count(), "pi_time");
    printDoubleJson(duration_cast<milliseconds>(pi2_end - pi2_start).count(), "pi_v2_time");
  }
  std::cout << "}" << std::endl;

  // we will just always use ipopt solution here
  alpha = ipopt_sol;
  // temp exit for debugging purposes...
  // we only want to output & analyze one collision at a time,
  // so letting the simulation run more would just produce more data that we won't use
  exit(0);
}

std::string LCPOperatorIsaiahDebug::name() const
{
  return "lcp_solver_isaiah_debug";
}

std::unique_ptr<ImpactOperator> LCPOperatorIsaiahDebug::clone() const
{
  return std::unique_ptr<ImpactOperator>(new LCPOperatorIsaiahDebug(m_linear_solver_order, m_tol, max_iters));
}

void LCPOperatorIsaiahDebug::serialize(std::ostream &output_stream) const
{
  Utilities::serialize(m_tol, output_stream);
}
