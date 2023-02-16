from typing import List, Optional, Type
import numpy as np
import pandas as pd


class IteratorABC(object):
    def __init__(
        self,
        Q: np.ndarray,
        b: np.ndarray,
        init_value: Optional[np.ndarray] = None,
        max_iter = 2000,
        convergence_tol = 1e-05
    ) -> None:
        self.Q = Q
        self.b = b
        if init_value is None:
            init_value = np.zeros(len(b))
        self.value = init_value.copy()
        self.intermediate_values = [init_value.copy()]
        self.intermediate_objective = [self.objective()]
        self.max_iter = max_iter
        self.convergence_tol = convergence_tol

    def flow(self):
        raise NotImplementedError()
    
    def objective(self):
        raise NotImplementedError()
    
    def solve(self):
        """
        Returns True if was able to converge, False otherwise
        """
        for i in range(self.max_iter):
            self.flow()
            if abs(self.objective()) < self.convergence_tol:
                print(f"CONVERGED IN {i} ITERATIONS")
                return True
        print("ERROR: reached max iterations")
        return False

class SolverApplier(object):
    def __init__(self, solver: Type[IteratorABC], data: pd.DataFrame) -> None:
        assert 'Q' in data
        assert 'b' in data
        assert 'value' in data
        assert 'converged' in data

        self.solver = solver
        self.data = data
    
    def apply_solver_to_row(self, row: pd.Series):
        self.solver_inst = self.solver(row['Q'], row['b'])
        row['converged'] = self.solver_inst.solve()
        row['value'] = self.solver_inst.value.copy()
    
    def run(self):
        self.data.apply(lambda r: self.apply_solver_to_row(r), axis=1)

