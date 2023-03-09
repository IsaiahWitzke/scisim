from typing import Optional
import numpy as np
import scipy
from solver_base import IteratorABC

class PolicyIteration(IteratorABC):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__( *args, **kwargs )
        self.max_iter = 10
    
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

    def flow(self):
        self._update_value()
        self._update_policy()