//
// Created by Wen Zhang on 2022-06-27.
//

#include <iostream>
#include "LCPOperatorPenalty.h"
#include "scisim/Utilities.h"
#include <chrono>

LCPOperatorPenalty::LCPOperatorPenalty(const scalar &tol, const unsigned &max_iters)
: m_tol (tol), max_iters (max_iters)
{
  assert(m_tol >= 0);
}

LCPOperatorPenalty::LCPOperatorPenalty(std::istream &input_stream)
: m_tol (Utilities::deserialize<scalar>( input_stream ))
, max_iters (Utilities::deserialize<unsigned>(input_stream))
{
  assert(m_tol >= 0);
}

// Returns the error of our solution
scalar getError(const SparseMatrixsc &Q, const VectorXs &x, const VectorXs &b, SparseMatrixsc &policy)
{
  scalar err2 = 0;
  VectorXs y = Q * x + b;
  for (int i = 0; i < x.size(); ++i) {
    scalar choice;
    if (y(i) < x(i))
      choice = 1;
    else
      choice = 0;
    err2 += fmin(x(i), y(i)) * fmin(x(i), y(i));
    policy.coeffRef(i, i) = choice;
  }
  return sqrt(err2);
}

void updateLambdaValue(const SparseMatrixsc &policy, const SparseMatrixsc &Q, const VectorXs &b, VectorXs &x)
{
  Eigen::ConjugateGradient<SparseMatrixsc, Eigen::Lower|Eigen::Upper> cg;
  SparseMatrixsc I {b.size(), b.size()};
  I.setIdentity();
  cg.compute(policy * Q * policy + I - policy);
  x = cg.solve(-policy * b);
}

void reportRuntime(const std::chrono::time_point<std::chrono::system_clock> &start)
{
  std::chrono::duration<double> elapsed_seconds = std::chrono::system_clock::now() - start;
  std::cout << "LCPOperatorPenalty: Time elapsed: " << elapsed_seconds.count() << "s\n";
}

void LCPOperatorPenalty::flow(const std::vector<std::unique_ptr<Constraint>> &cons, const SparseMatrixsc &M,
                         const SparseMatrixsc &Minv, const VectorXs &q0, const VectorXs &v0, const VectorXs &v0F,
                         const SparseMatrixsc &N, const SparseMatrixsc &Q, const VectorXs &nrel, const VectorXs &CoR,
                         VectorXs &alpha) {
  std::cout << "LCPOperatorPenalty: Solving LCP of size " << N.cols() << std::endl;
  // Get initial time
  std::chrono::time_point<std::chrono::system_clock> start = std::chrono::system_clock::now();

  VectorXs b { N.transpose() * (1 + CoR(0)) * v0 };
  // Solve complementarity with x and Qx + b
  VectorXs x { b.size() }; // Initial guess of 0 (maybe change later)
  SparseMatrixsc policy { x.size(), x.size() };
  scalar error = 0;
  for (unsigned n_iter = 0; n_iter <= max_iters; ++n_iter)
  {
    error = getError(Q, x, b, policy);
    if (error <= m_tol) {
      alpha = x;
      std::cout << "LCPOperatorPenalty: Converged in " << n_iter << " iterations." << std::endl;
      reportRuntime(start);
      return;
    }
    if (n_iter == max_iters)
      break;
    updateLambdaValue(policy, Q, b, x);
  }
  std::cout << "LCPOperatorPenalty: Failed to converge in " << max_iters << " iterations." << std::endl;
  reportRuntime(start);
  std::cerr << "LCPOperatorPenalty: Result did not converge" << std::endl;
  std::cerr << "LCPOperatorPenalty: Error is: " << error << std::endl;
  std::cerr << "LCPOperatorPenalty: Failed with size: " << N.cols() << std::endl;
  alpha = x;
}

std::string LCPOperatorPenalty::name() const {
  return "lcp_penalty";
}

std::unique_ptr<ImpactOperator> LCPOperatorPenalty::clone() const {
  return std::unique_ptr<ImpactOperator>(new LCPOperatorPenalty(m_tol, max_iters));
}

void LCPOperatorPenalty::serialize(std::ostream &output_stream) const {
  Utilities::serialize(m_tol, output_stream);
}

