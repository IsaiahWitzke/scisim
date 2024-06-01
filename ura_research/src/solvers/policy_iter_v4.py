import random
from typing import List, Optional
import numpy as np
import scipy
from solver_base import IteratorLcpSolver

"""
Idea with this one:
- when we get into a loop, begin getting rid of linearly dependant row/cols

Our Q matrix isn't "monotone", so this is an attempt to make it so!
"""


class PolicyIterationV4(IteratorLcpSolver):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__( *args, **kwargs )
        self.prev_Q: List[np.ndarray] = []
    
    def objective(self):
        return np.linalg.norm(np.minimum(self.Q @ self.value - self.b, self.value))
    
    def _update_policy(self) -> None:
        self.policy = np.zeros(len(self.b))
        y = self.Q @ self.value - self.b
        for i in range(len(self.b)):
            if y[i] < self.value[i]:
                self.policy[i] = 1

    def _update_value(self) -> None:
        policy_matrix = np.zeros(self.Q.shape)  # a matrix that will be used extract a sub-matrix of Q into A
        for i in range(len(self.policy)):
            if self.policy[i] == 1:
                policy_matrix[i,i] = 1
        I = np.eye(len(self.b))
        # get the submatrix of Q and store in A. A is the same size as Q, but has some rows/columns replaced by identity
        # depending on self.policy
        A = policy_matrix @ self.Q @ policy_matrix + I - policy_matrix
        rhs = policy_matrix @ self.b
        cg_soln = scipy.sparse.linalg.cg(A , rhs , tol=1e-12)[0]
        self.value = cg_soln
    
    def _find_previous_good_itr(self) -> int:
        min_objective = min(self.intermediate_objective)
        min_objective_itr = self.intermediate_objective.index(min_objective)
        return min_objective_itr


    def _simplify_Q(self) -> None:
        """
        Gets rid of a lin dependant row/col of Q (with priority on the rows with most diagonal dominance deviance)

        DDD ~= "row with biggest norm"
        """

        
        indices = list(range(self.Q.shape[0]))
        row_idx_sorted_by_norm = sorted(
            indices,
            # negative so we get in descending order
            key=lambda i: -np.linalg.norm(self.Q[i])
        )

        for idx in row_idx_sorted_by_norm:
            new_Q = self.Q.copy()
            new_Q[idx] = np.zeros(self.Q.shape[0])
            new_Q[:,idx] = np.zeros(self.Q.shape[0])
            new_Q[idx, idx] = 1.0

            tol = 1e-9
            rank_before = np.linalg.matrix_rank(self.Q, tol=tol)
            rank_after = np.linalg.matrix_rank(new_Q, tol=tol)
            print(f"rank before: {rank_before} (of {self.Q.shape[0]})")
            print(f"rank after: {rank_after}")
            if rank_before < rank_after:
                self.Q = new_Q
                return
        
        print("Couldn't simplify Q :(")

    def _is_policy_already_visited(self, p: np.ndarray) -> bool:
        print(p)
        for i in range(len(self.intermediate_policies)):
            if np.allclose(self.intermediate_policies[i], p):
                return True
        return False


    def flow(self):
        # for the first time through, we actually make a guess for the value so that we don't think that we diverge right away
        if len(self.intermediate_values) == 1:
            self._update_value()
            self.intermediate_values[0] = self.value


        # loop to find a "good policy"
        # first, just try to find a policy/value pair as normal... afterwards we check if this policy has already been found
        # we then go to a previous "close" policy, randomize a bit, then try again
        self._update_policy()
        iters = 0
        if self._is_policy_already_visited(self.policy):
            self.pois.append(len(self.intermediate_policies))
        while self._is_policy_already_visited(self.policy):
            # safety stop! if we've completely exhausted the search area around the last best point, we give up
            iters += 1
            if iters == 20:
                print("SAFETY STOP!")
                return False
            self._simplify_Q()
            self._update_policy()

        self._update_value()
        return True