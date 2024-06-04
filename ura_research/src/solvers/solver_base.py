from typing import Optional, Type
import numpy as np
import pandas as pd
import random

random.seed(123)

def gen_random_policy(size: int) -> np.ndarray:
    return np.array([
        1.0 if random.random() > 0.5 else 0.0 for _ in range(size)
    ])

class IteratorLcpSolver(object):
    """
    A superclass for all solvers that use an iterative method to solve the LCP.

    Child classes just need to implement the `flow` and `objective` methods.

    After running the solver with `iterator_lcp_solver_instance.solve()`,
    `self.intermediate_values[-1]` will contain the solution
    (only if `self.converged == True`).

    The values, policies (if used), and objective function values at each iteration
    are stored in `self.intermediate_values`, `self.intermediate_policies`, and `self.intermediate_objective`,
    respectively so that a history of the solver's progress can be viewed.
    """

    def __init__(
        self,
        Q: np.ndarray,
        b: np.ndarray,
        init_value: Optional[np.ndarray] = None,
        max_iter = 200,
        convergence_tol = 1e-10,
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
        self.pois = []
        self.converged = False

    def flow(self) -> bool:
        """
        A virtual function that performs one iteration of the algorithm.

        Naming this function "flow" was a poor choice since this flow method
        differs quite a bit for from ImpactOperator.h's `flow` method.
        This flow is called once per iteration, whereas ImpactOperator.h's flow
        is only called once per collision.
        """
        raise NotImplementedError()
    
    def objective(self):
        raise NotImplementedError()
    
    def solve(self, debug_logs = ["diverge", "converge"]) -> None:
        """
        Returns True if was able to converge, False otherwise

        debug_logs: an array of values that specify what types of debugging info to output
        """
        for i in range(self.max_iter):
            if not self.flow():
                if "diverge" in debug_logs:
                    print(f"{self.name}: DIVERGED AT ITERATION {i}")
                return
            self.intermediate_objective.append(self.objective())
            self.intermediate_values.append(self.value.copy())
            self.intermediate_policies.append(self.policy.copy())
            if abs(self.objective()) < self.convergence_tol:
                if "converge" in debug_logs:
                    print(f"{self.name}: CONVERGED IN {i} ITERATIONS")
                self.converged = True
                return
            
        print(f"{self.name}: reached max iterations")

class SolverApplier(object):
    def __init__(self, solver: Type[IteratorLcpSolver], data: pd.DataFrame) -> None:
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

