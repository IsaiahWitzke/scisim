//
// Created by Wen Zhang on 2022-07-18
//

#ifndef SCISIM_LCP_OPERATOR_3_H
#define SCISIM_LCP_OPERATOR_3_H

// Uses 3 LCP solver methods - IPOPT, PI, and Penalty

#include "ImpactOperator.h"
#include "LCPOperatorIpopt.h"
#include "LCPOperatorPenalty.h"
#include "LCPOperatorPI.h"

class LCPOperator3 final : public ImpactOperator {

public:

    explicit LCPOperator3( const std::vector<std::string>& linear_solvers, const scalar& tol, const unsigned& max_iters );
    explicit LCPOperator3( std::istream& input_stream );

    virtual ~LCPOperator3() override = default;

    virtual void flow( const std::vector<std::unique_ptr<Constraint>>& cons, const SparseMatrixsc& M, const SparseMatrixsc& Minv, const VectorXs& q0, const VectorXs& v0, const VectorXs& v0F, const SparseMatrixsc& N, const SparseMatrixsc& Q, const VectorXs& nrel, const VectorXs& CoR, VectorXs& alpha ) override;

    virtual std::string name() const override;

    virtual std::unique_ptr<ImpactOperator> clone() const override;

    virtual void serialize( std::ostream& output_stream ) const override;

private:
    const std::vector<std::string> m_linear_solver_order;
    const scalar m_tol;
    const unsigned max_iters;
    std::unique_ptr<LCPOperatorIpopt> ipopt_solver;
    std::unique_ptr<LCPOperatorPI> policy_solver;
    std::unique_ptr<LCPOperatorPenalty> penalty_solver;
};


#endif //SCISIM_LCP_OPERATOR_PI_H
