import numpy as np
import scipy

def get_policy(Q, x, b):
    """
    updates policy to the next policy,
    returns error
    """
    policy = []
    err2 = 0
    y = Q @ x + b
    for i in range(len(x)):
        if y[i] < x[i]:
            policy.append(i)
        err2 += min(x[i], y[i]) ** 2    # x should be > 0, while y should be as close to 0 as possible
    return err2 ** 0.5, policy

# get the sub-system of linear equations given the policy
def get_sub_sys(policy, Q, b):
    policy_matrix = np.zeros((b.shape[0], b.shape[0]))
    for p in policy:
        policy_matrix[p,p] = 1
    I = np.eye(len(b))
    A = np.dot(
            np.dot(
                policy_matrix,
                Q
            ),
            policy_matrix
        ) + I - policy_matrix

    rhs = - policy_matrix @ b 
    # new_Q = np.zeros((len(policy), len(policy)))
    # new_b = np.zeros(len(policy))

    # for new_row, old_row in enumerate(policy):
    #     new_b[new_row] = b[old_row]
    #     for new_col, old_col in enumerate(policy):
    #         new_Q[new_row, new_col] = Q[old_row, old_col]
    
    return A, rhs
    


def update_value(policy, Q, b, use_cg = True):
    """
    returns the next x given the policy
    """
    I = np.eye(len(b))
    policy_matrix = np.zeros((b.shape[0], b.shape[0]))
    for p in policy:
        policy_matrix[p,p] = 1
    A = policy_matrix @ Q @ policy_matrix + I - policy_matrix
    rhs = - policy_matrix @ b 
    lst_sqr_soln = np.linalg.lstsq(A , rhs)[0]
    cg_soln = scipy.sparse.linalg.cg(A , rhs , tol=1e-12)[0]
    if use_cg:
        return cg_soln
    else:
        return lst_sqr_soln


def flow(Q, N, v0, initial_control, max_itrs = 100, m_tol = 1e-09, CoR = 1, use_cg = True):
    """
    does policy iteration
    returns a bool/vector pair: (converged, x)
    """
    b = np.dot(N.T, (1 + CoR) * v0)
    x = np.zeros(b.shape[0])
    policy = initial_control.copy()
    error = 0.0

    for n_iter in range(max_itrs):
        # have we stumbled across the correct policy/control?
        x = update_value(policy, Q, b, use_cg)
        p = policy.copy()
        error, policy = get_policy(Q, x, b)
        if error <= m_tol:
            return (True, x, p)

    
    return (False, x, None)
