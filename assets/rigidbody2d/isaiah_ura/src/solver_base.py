from typing import Optional, Type
import numpy as np
import pandas as pd


class IteratorABC(object):
    def __init__(
        self,
        Q: np.ndarray,
        b: np.ndarray,
        init_value: Optional[np.ndarray] = None,
        max_iter = 2000,
        convergence_tol = 1e-05,
        *args,
        **kwargs,
    ) -> None:
        self.Q = Q
        self.b = b
        if init_value is None:
            init_value = np.zeros(len(b))
        self.value = init_value.copy()
        self.max_iter = max_iter
        self.convergence_tol = convergence_tol
        self.name = kwargs.get("name", "Iterator")
        self.policy = kwargs.get('initial_policy', np.ones(len(self.b))).copy()

        self.intermediate_values = [init_value.copy()]
        self.intermediate_policies = [self.policy.copy()]
        self.intermediate_objective = [self.objective()]

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
            self.intermediate_objective.append(self.objective())
            self.intermediate_values.append(self.value.copy())
            self.intermediate_policies.append(self.policy.copy())
            if abs(self.objective()) < self.convergence_tol:
                # print(f"{self.name}: CONVERGED IN {i} ITERATIONS")
                return True
            
            # try to search for a previous policy that we've already come across...
            # if we can find one that means we are in a loop and can just quit now
            for i in range(len(self.intermediate_values) - 1):
                if np.allclose(self.intermediate_values[i], self.intermediate_values[-1]):
                    a = len(self.intermediate_values) - 1
                    print(f"{self.name}: DIVERGING CYCLE FROM {i} to {a} (length: {a - i})")
                    for j in range(i, a):
                        print(self.intermediate_values[j])
                    return False
            
        print(f"{self.name}: reached max iterations")
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

