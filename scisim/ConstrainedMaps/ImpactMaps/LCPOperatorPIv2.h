//
// Created by Isaiah Witzke on 2022-10-11.
//

#ifndef SCISIM_LCP_OPERATOR_PI_V2_H
#define SCISIM_LCP_OPERATOR_PI_V2_H

#include <bits/stdc++.h>
#include <vector>

#include "ImpactOperator.h"
#include "scisim/Utilities.h"

class LCPOperatorPIv2 final : public ImpactOperator {
public:
    explicit LCPOperatorPIv2(const scalar& tol, const unsigned& max_iters);
    explicit LCPOperatorPIv2(std::istream& input_stream);

    virtual ~LCPOperatorPIv2() override = default;

    virtual void flow(
        const std::vector<std::unique_ptr<Constraint>>& cons,
        const SparseMatrixsc& M,
        const SparseMatrixsc& Minv,
        const VectorXs& q0,
        const VectorXs& v0,
        const VectorXs& v0F,
        const SparseMatrixsc& N,
        const SparseMatrixsc& Q,
        const VectorXs& nrel,
        const VectorXs& CoR,
        VectorXs& alpha) override;

    virtual std::string name() const override;

    virtual std::unique_ptr<ImpactOperator> clone() const override;

    virtual void serialize(std::ostream& output_stream) const override;

    bool converged = false;

    std::unordered_set<std::string> previous_policies;

private:
    const scalar m_tol;
    const unsigned max_iters;
    scalar getPolicy(
        const SparseMatrixsc& Q,
        const VectorXs& x,
        const VectorXs& b,
        SparseMatrixsc& policy);
    void updateValue(
        const SparseMatrixsc& policy,
        const SparseMatrixsc& Q,
        const VectorXs& b,
        VectorXs& x);
};

#endif
