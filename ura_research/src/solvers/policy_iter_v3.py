import random
from typing import Optional
import numpy as np
import scipy
from solver_base import IteratorLcpSolver

class PolicyIterationV3(IteratorLcpSolver):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__( *args, **kwargs )
    
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

    def _gen_semi_random_policy(self, p: np.ndarray) -> np.ndarray:
        """
        returns a slightly-perturbed version of the input p
        """

        def invert_float_1_0(x):
            if x == 1.0:
                return 0.0
            else:
                return 1.0

        # flip 1 <-> 0 with a 10% chance
        return np.array([ invert_float_1_0(x) if random.random() > 0.9 else x for x in p ])

    def _is_policy_already_visited(self, p: np.ndarray) -> bool:
        for i in range(len(self.intermediate_policies)):
            if np.allclose(self.intermediate_policies[i], self.policy):
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
                return False
            prev_good_policy = self.intermediate_policies[self._find_previous_good_itr()]
            self.policy = self._gen_semi_random_policy(prev_good_policy)

        self._update_value()
        return True