//
// Created by Isaiah Witzke on 2022-10-11.
//

#include "LCPOperatorPIv2.h"

#include <algorithm>
#include <chrono>
#include <iostream>

#include "scisim/Utilities.h"

LCPOperatorPIv2::LCPOperatorPIv2(const scalar& tol, const unsigned& max_iters)
    : m_tol(tol)
    , max_iters(max_iters)
{
    srand(0);
    assert(m_tol > 0.0);
}

LCPOperatorPIv2::LCPOperatorPIv2(std::istream& input_stream)
    : m_tol(Utilities::deserialize<scalar>(input_stream))
    , max_iters(Utilities::deserialize<unsigned>(input_stream))
{
    srand(0);
    assert(m_tol >= 0.0);
}

// TODO: optimize
std::string encodePolicy(const SparseMatrixsc& p)
{
    // std::cout << "converting following into string:" << std::endl;
    unsigned int n_rows = p.rows();
    std::string encoded_str = "";
    for (unsigned int i = 0; i < n_rows; i++) {
        if (p.diagonal()(i) == 1.0) {
            encoded_str += "1";
        } else {
            encoded_str += "0";
        }
    }
    // std::cerr << encoded_str << std::endl;
    return encoded_str;

    // // an empty string "buffer"
    // const int num_bytes = 1 + (n_rows / 8);
    // std::string encoded_policy(num_bytes, (char)0);

    // unsigned int cur_byte = 0;
    // for (unsigned int i = 0; i < n_rows; i++) {
    //     if (i % 8 == 0) {
    //         cur_byte++;
    //     }
    //     if (p.diagonal()(i) == 1.0) {
    //         encoded_policy[cur_byte] += ((char)1) << (i % 8);
    //     }
    // }

    // std::cout << "encoded policy" << std::endl
    //           << encoded_policy << std::endl;
    // return encoded_policy;
}

static bool isPolicyInVector(
    const std::string& p_encoded,
    const std::unordered_set<std::string>& policy_set)
{
    if (policy_set.find(p_encoded) != policy_set.end()) {
        return true;
    } else {
        return false;
    }
    // const Eigen::Diagonal<const SparseMatrixsc> policy_diag =
    // policy.diagonal(); for (int i = 0; i < policy_list.size(); ++i)
    // {
    //   const Eigen::Diagonal<const SparseMatrixsc> diag =
    //   policy_list[i].diagonal(); if (diag.isApprox(policy_diag))
    //   {
    //     return true;
    //   }
    // }
    // return false;
}

// Returns the error of our solution & updates policy
scalar LCPOperatorPIv2::getPolicy(
    const SparseMatrixsc& Q,
    const VectorXs& x,
    const VectorXs& b,
    SparseMatrixsc& policy)
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

    std::chrono::time_point<std::chrono::system_clock> start = std::chrono::system_clock::now();
    std::string p_encoded = encodePolicy(policy);
    while (isPolicyInVector(p_encoded, this->previous_policies)) {
        // if we've already come across this policy, randomize ourselves (effectively trying again)
        for (int i = 0; i < x.size(); ++i) {
            policy.coeffRef(i, i) = rand() % 2;
        }
        p_encoded = encodePolicy(policy);
    }
    // mark this new policy as one we've already come across for future
    this->previous_policies.insert(p_encoded);
    std::chrono::time_point<std::chrono::system_clock> stop = std::chrono::system_clock::now();
    std::cerr << std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() << std::endl;
    return sqrt(err2);
}

void LCPOperatorPIv2::updateValue(const SparseMatrixsc& policy,
    const SparseMatrixsc& Q, const VectorXs& b,
    VectorXs& x)
{
    Eigen::ConjugateGradient<SparseMatrixsc, Eigen::Lower | Eigen::Upper> cg;
    // Eigen::ConjugateGradient<SparseMatrixsc, Eigen::Lower> cg;
    SparseMatrixsc I { b.size(), b.size() };
    I.setIdentity();
    cg.compute(policy * Q * policy + I - policy);
    x = cg.solve(-policy * b);
}

void LCPOperatorPIv2::flow(const std::vector<std::unique_ptr<Constraint>>& cons,
    const SparseMatrixsc& M, const SparseMatrixsc& Minv,
    const VectorXs& q0, const VectorXs& v0,
    const VectorXs& v0F, const SparseMatrixsc& N,
    const SparseMatrixsc& Q, const VectorXs& nrel,
    const VectorXs& CoR, VectorXs& alpha)
{
    int size = N.cols();
    // std::cerr << "LCPOperatorPI: Solving LCP of size " << N.cols() <<
    // std::endl; Get initial time

    VectorXs b { N.transpose() * (1 + CoR(0)) * v0 };
    // Solve complementarity with x and Qx + b
    VectorXs x { b.size() }; // Initial guess of 0 (maybe change later)
    SparseMatrixsc policy { x.size(), x.size() };
    scalar error = 0;

    for (unsigned n_iter = 0; n_iter <= max_iters; ++n_iter) {
        error = getPolicy(Q, x, b, policy);
        // std::cerr << n_iter << ": " << error << std::endl;
        if (error <= m_tol) {
            alpha = x;
            this->converged = true;
            return;
        }
        if (n_iter == max_iters) {
            break;
        }

        updateValue(policy, Q, b, x);
    }
    alpha = x;
}

std::string LCPOperatorPIv2::name() const { return "lcp_solver_pi_v2"; }

std::unique_ptr<ImpactOperator> LCPOperatorPIv2::clone() const
{
    return std::unique_ptr<ImpactOperator>(new LCPOperatorPIv2(m_tol, max_iters));
}

void LCPOperatorPIv2::serialize(std::ostream& output_stream) const
{
    Utilities::serialize(m_tol, output_stream);
}
