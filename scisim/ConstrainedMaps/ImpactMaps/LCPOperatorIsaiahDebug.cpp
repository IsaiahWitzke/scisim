//
// Created by Isaiah Witzke on 2022-10-11.
//

#include "scisim/StringUtilities.h"
#include "scisim/Utilities.h"
#include "scisim/ConstrainedMaps/IpoptUtilities.h"

#include <iostream>
#include "LCPOperatorIsaiahDebug.h"
#include "scisim/Utilities.h"
#include <chrono>

LCPOperatorIsaiahDebug::LCPOperatorIsaiahDebug(const std::vector<std::string>& linear_solvers, const scalar &tol, const unsigned &max_iters)
: m_linear_solver_order( linear_solvers ), m_tol (tol), max_iters (max_iters)
{
  assert( m_tol > 0.0 );

  // Verify that the user provided a valid linear solver option
  if( m_linear_solver_order.empty() )
  {
    std::cerr << "No linear solver provided to LCPOperatorIsaiahDebug. Exiting." << std::endl;
    std::exit( EXIT_FAILURE );
  }
  if( IpoptUtilities::containsDuplicates( m_linear_solver_order ) )
  {
    std::cerr << "Duplicate linear solvers provided to LCPOperatorIsaiahDebug. Exiting." << std::endl;
    std::exit( EXIT_FAILURE );
  }
  for( const std::string& solver_name : m_linear_solver_order )
  {
    if( !IpoptUtilities::linearSolverSupported( solver_name ) )
    {
      std::cerr << "Invalid linear solver provided to LCPOperatorIsaiahDebut: " << solver_name << ". Exiting." << std::endl;
      std::exit( EXIT_FAILURE );
    }
  }

  ipopt_solver = std::unique_ptr<LCPOperatorIpopt>(new LCPOperatorIpopt(m_linear_solver_order, m_tol));
  policy_solver = std::unique_ptr<LCPOperatorPI>(new LCPOperatorPI(m_tol, max_iters));
  penalty_solver = std::unique_ptr<LCPOperatorPenalty>(new LCPOperatorPenalty(m_tol, max_iters));
}

LCPOperatorIsaiahDebug::LCPOperatorIsaiahDebug(std::istream &input_stream)
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

void printMatrix(SparseMatrixsc m, std::string name) {
    std::cout << name << ":" << std::endl;
    std::cout << "size: " << m.rows() << " " << m.cols() << std::endl; 

    for (int k=0; k < m.outerSize(); ++k)
    {
      for (SparseMatrixsc::InnerIterator it(m,k); it; ++it)
      {
        std::cout.precision(12);
        std::cout << "(" << it.row() << "," << it.col() << "," << std::fixed << double(it.value()) << ") ";
      }
    }

    std::cout << std::endl;
}
void printVector(VectorXs v, std::string name) {
  std::cout << name << ":" << std::endl;
  std::cout << "size: " << v.rows() << " " << v.cols() << std::endl; 

  for (int row=0; row < v.rows(); ++row) {
    std::cout.precision(12);
    std::cout << "(" << row << "," << 0 << "," << std::fixed << double(v(row)) << ") ";
  }

  std::cout << std::endl;
}

float getKineticEnergy(const SparseMatrixsc &M, const VectorXs &v) {
  return 0.5 * v.transpose() * M * v;
}

void LCPOperatorIsaiahDebug::flow(const std::vector<std::unique_ptr<Constraint>> &cons, const SparseMatrixsc &M,
                         const SparseMatrixsc &Minv, const VectorXs &q0, const VectorXs &v0, const VectorXs &v0F,
                         const SparseMatrixsc &N, const SparseMatrixsc &Q, const VectorXs &nrel, const VectorXs &CoR,
                         VectorXs &alpha) {
    VectorXs b { N.transpose() * (1 + CoR(0)) * v0 };


    VectorXs ipopt_sol = alpha;
    VectorXs policy_sol = alpha;
    VectorXs penalty_sol = alpha;

    bool policy_converges = false;
    bool penalty_converges = false;
    bool ipopt_converges = false;
    
    ipopt_solver->flow(cons, M, Minv, q0, v0, v0F, N, Q, nrel, CoR, ipopt_sol);

    policy_solver->target_soln = ipopt_sol;

    policy_solver->flow(cons, M, Minv, q0, v0, v0F, N, Q, nrel, CoR, policy_sol);
    if (getEndError(Q, policy_sol, b) <= m_tol) {
        policy_converges = true;
    }
    if (getEndError(Q, ipopt_sol, b) <= m_tol) {
        ipopt_converges = true;
    }

    // float K_0 = getKineticEnergy(M, v0);
    // float K_1 = getKineticEnergy(M, Minv * ipopt_sol);
    // printMatrix(M, "M");
    printMatrix(N, "N");
    printMatrix(Q, "Q");
    printMatrix(Minv, "Minv");
    printVector(v0, "v0");
    printVector(ipopt_sol, "ipopt_sol");
    printVector(policy_sol, "policy_sol");
    // printMatrix(v0, "v0");
    // penalty_solver->flow(cons, M, Minv, q0, v0, v0F, N, Q, nrel, CoR, penalty_sol);
    // if (getEndError(Q, penalty_sol, b) <= m_tol) {
    //     penalty_converges = true;
    // }
    SparseMatrixsc N_c(N);
    N_c.makeCompressed();

    std::cout << std::endl;
    std::cout << "policy_converges:\n" << policy_converges << std::endl;
    std::cout << "ipopt_converges:\n" << ipopt_converges << std::endl;
    // std::cerr << "policy_converges:" << policy_converges << std::endl;
    // std::cout << "penalty_converges:\n" << penalty_converges << std::endl;

    // we will just always use ipopt solution here
    alpha = ipopt_sol;
    exit(0);
}

std::string LCPOperatorIsaiahDebug::name() const {
  return "lcp_solver_isaiah_debug";
}

std::unique_ptr<ImpactOperator> LCPOperatorIsaiahDebug::clone() const {
  return std::unique_ptr<ImpactOperator>(new LCPOperatorIsaiahDebug( m_linear_solver_order, m_tol, max_iters));
}

void LCPOperatorIsaiahDebug::serialize(std::ostream &output_stream) const {
  Utilities::serialize(m_tol, output_stream);
}

