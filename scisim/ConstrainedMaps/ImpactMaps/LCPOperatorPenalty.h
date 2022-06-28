//
// Created by Wen Zhang on 2022-06-27
//

#ifndef SCISIM_LCP_OPERATOR_PENALTY_H
#define SCISIM_LCP_OPERATOR_PENALTY_H

#include "ImpactOperator.h"

class LCPOperatorPenalty final : public ImpactOperator {

public:

    explicit LCPOperatorPenalty( const scalar& tol, const unsigned& max_iters );
    explicit LCPOperatorPenalty( std::istream& input_stream );

    virtual ~LCPOperatorPenalty() override = default;

    virtual void flow( const std::vector<std::unique_ptr<Constraint>>& cons, const SparseMatrixsc& M, const SparseMatrixsc& Minv, const VectorXs& q0, const VectorXs& v0, const VectorXs& v0F, const SparseMatrixsc& N, const SparseMatrixsc& Q, const VectorXs& nrel, const VectorXs& CoR, VectorXs& alpha ) override;

    virtual std::string name() const override;

    virtual std::unique_ptr<ImpactOperator> clone() const override;

    virtual void serialize( std::ostream& output_stream ) const override;

private:
    const scalar m_tol;
    const unsigned max_iters;
};


#endif //SCISIM_LCP_OPERATOR_PI_H
