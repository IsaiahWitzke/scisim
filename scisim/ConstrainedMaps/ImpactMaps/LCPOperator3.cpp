//
// Created by Wen Zhang on 2022-06-27.
//

#include "scisim/StringUtilities.h"
#include "scisim/Utilities.h"
#include "scisim/ConstrainedMaps/IpoptUtilities.h"

#include <iostream>
#include "LCPOperator3.h"
#include "scisim/Utilities.h"
#include <chrono>

LCPOperator3::LCPOperator3(const std::vector<std::string>& linear_solvers, const scalar &tol, const unsigned &max_iters)
: m_linear_solver_order( linear_solvers ), m_tol (tol), max_iters (max_iters)
{
  assert( m_tol > 0.0 );

  // Verify that the user provided a valid linear solver option
  if( m_linear_solver_order.empty() )
  {
    std::cerr << "No linear solver provided to LCPOperator3. Exiting." << std::endl;
    std::exit( EXIT_FAILURE );
  }
  if( IpoptUtilities::containsDuplicates( m_linear_solver_order ) )
  {
    std::cerr << "Duplicate linear solvers provided to LCPOperator3. Exiting." << std::endl;
    std::exit( EXIT_FAILURE );
  }
  for( const std::string& solver_name : m_linear_solver_order )
  {
    if( !IpoptUtilities::linearSolverSupported( solver_name ) )
    {
      std::cerr << "Invalid linear solver provided to LCPOperator3: " << solver_name << ". Exiting." << std::endl;
      std::exit( EXIT_FAILURE );
    }
  }

  ipopt_solver = std::unique_ptr<LCPOperatorIpopt>(new LCPOperatorIpopt(m_linear_solver_order, m_tol));
  policy_solver = std::unique_ptr<LCPOperatorPI>(new LCPOperatorPI(m_tol, max_iters));
  penalty_solver = std::unique_ptr<LCPOperatorPenalty>(new LCPOperatorPenalty(m_tol, max_iters));
}

LCPOperator3::LCPOperator3(std::istream &input_stream)
: m_linear_solver_order( StringUtilities::deserializeVector( input_stream ) )
, m_tol (Utilities::deserialize<scalar>( input_stream ))
, max_iters (Utilities::deserialize<unsigned>(input_stream))
{
  assert( !m_linear_solver_order.empty() );
  assert( m_tol >= 0.0 );
}


static scalar getEndError(const SparseMatrixsc &Q, const VectorXs &x, const VectorXs &b)
{
  scalar err2 = 0;
  VectorXs y = Q * x + b; // resultant vector
  for (int i = 0; i < x.size(); ++i) {
    err2 += fmin(x(i), y(i)) * fmin(x(i), y(i));
  }
  return sqrt(err2);
}

static scalar getAbsDiff(VectorXs a, VectorXs b) {
    scalar err = 0;
    for (int i = 0; i < a.size(); ++i) {
        err += ((a(i) - b(i)) * (a(i) - b(i)));
    }
    return sqrt(err);
}

// runs the actual collision solver - all 3 methods: ipopt, pi, penalty
void LCPOperator3::flow(const std::vector<std::unique_ptr<Constraint>> &cons, const SparseMatrixsc &M,
                         const SparseMatrixsc &Minv, const VectorXs &q0, const VectorXs &v0, const VectorXs &v0F,
                         const SparseMatrixsc &N, const SparseMatrixsc &Q, const VectorXs &nrel, const VectorXs &CoR,
                         VectorXs &alpha) {
    VectorXs b { N.transpose() * (1 + CoR(0)) * v0 };

    VectorXs ipopt_sol = alpha;
    VectorXs policy_sol = alpha;
    VectorXs penalty_sol = alpha;

    bool policy_converges = false;
    bool penalty_converges = false;
    
    ipopt_solver->flow(cons, M, Minv, q0, v0, v0F, N, Q, nrel, CoR, ipopt_sol);
    //TODO: how to check if ipopt doesn't converge, since it crashes automatically

    policy_solver->flow(cons, M, Minv, q0, v0, v0F, N, Q, nrel, CoR, policy_sol);
    if (getEndError(Q, policy_sol, b) <= m_tol) {
        policy_converges = true;
    }
    penalty_solver->flow(cons, M, Minv, q0, v0, v0F, N, Q, nrel, CoR, penalty_sol);
    if (getEndError(Q, penalty_sol, b) <= m_tol) {
        penalty_converges = true;
    }

    //find the differences between solutions
    scalar ipopt_policy_diff = getAbsDiff(ipopt_sol, policy_sol);
    scalar policy_penalty_diff = getAbsDiff(policy_sol, penalty_sol);
    scalar penalty_ipopt_diff = getAbsDiff(penalty_sol, ipopt_sol);

    //print stats
    std::cout << policy_converges << ", " << penalty_converges << ", size," << N.cols() << "," << ipopt_policy_diff << ", " << policy_penalty_diff << ", " << penalty_ipopt_diff;
    std::cout << std::endl;
    
    // pick a next solution: priority: penalty, policy, ipopt
    if (penalty_converges) {
        alpha = penalty_sol;
    } else if (policy_converges) {
        alpha = policy_sol;
    } else {
        alpha = ipopt_sol;
    }
}

std::string LCPOperator3::name() const {
  return "lcp_solver3";
}

std::unique_ptr<ImpactOperator> LCPOperator3::clone() const {
  return std::unique_ptr<ImpactOperator>(new LCPOperator3( m_linear_solver_order, m_tol, max_iters));
}

void LCPOperator3::serialize(std::ostream &output_stream) const {
  Utilities::serialize(m_tol, output_stream);
}

