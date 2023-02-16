from typing import Optional
import numpy as np
from solver_base import IteratorABC

class ValueIteration(IteratorABC):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__( *args, **kwargs )
    
    def objective(self):
        return np.linalg.norm(np.minimum(self.Q @ self.value - self.b, self.value))
    
    def flow(self):
        # hard-coded value iteration for our problem
        num_rows = len(self.b)

        # diag_value = self.Q[0,0]
        # next_values = np.empty((f.b), 2))
        next_value = np.empty(num_rows)

        for i in range(len(self.b)):
            # self.value[i] = -min(
            next_value[i] = -min(
                (sum([
                    self.Q[i,j] * self.value[j]
                    for j in range(num_rows) if j != i
                ]) - self.b[i]) / self.Q[i,i],
                0
            )
            # next_value[i] = -min(
            #     (sum([self.Q[i,j] * self.value[j] for j in range(num_rows)]) - self.b[i]) / 2,
            #     0
            # )
        self.value = next_value
        # # "in"
        # next_values[:,1] = (self.Q @ self.value - diag_value * self.value - self.b) / diag_value
        # # "out"
        # next_values[:,0] = (1/diag_value - 1) * self.value
        # # save the last value for debugging purposes
        # # update value
        # self.value = -np.amin(next_values,axis=1)

        self.intermediate_objective.append(self.objective())
        self.intermediate_values.append(self.value.copy())