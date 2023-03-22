from typing import Tuple
from policy_iter import PolicyIteration
import pytest
import numpy as np
import random
import _isaiah_stuff.research.util.data_import as data_import


def generate_random_m_matrix(dim: int = 5, rand_range: float = 10.0):
    m = np.empty((dim, dim), dtype=float)
    is_m_matrix = False
    while not is_m_matrix:
        for row in range(m.shape[0]):
            for col in range(m.shape[1]):
                if row == col:
                    m[row, col] = random.uniform(0, rand_range)
                else:
                    m[row, col] = random.uniform(-rand_range, 0)

        # is our matrix non-singular? (dont know if we really need to test this)
        if np.linalg.matrix_rank(m) == dim:
            # are the real parts of the eigenvalues non-negative?
            is_m_matrix = True
            eigenvalues = np.linalg.eigvals(m)
            for e in eigenvalues:
                if type(e) == np.complex128 and e.real < 0:
                    is_m_matrix = False
                    break
                elif e < 0:
                    is_m_matrix = False
                    break
    return m

def generate_mocked_random_simulation_Q(dim: int = 5, rand_range: float = 10.0):
    """
    produces a Q matrix similar to what might be generated with balls of mass 1
    colliding with each other... i.e.: diagonal is 2, all other values are a random
    number in [-1, 1], to try to replicate cos(t) output
    """
    m = np.empty((dim, dim), dtype=float)
    for row in range(m.shape[0]):
        for col in range(m.shape[1]):
            if row == col:
                m[row, col] = 2.0
            else:
                m[row, col] = random.uniform(-1, 1)

    return m


@pytest.fixture(params=list(range(0)))
def solver_mocked_random_simulation(request) -> PolicyIteration:
    random.seed(request.param)
    np.random.seed(request.param)
    d = 3
    lhs = generate_mocked_random_simulation_Q(d)
    rhs = np.random.rand(d)
    return PolicyIteration(lhs, rhs)


def test_policy_iter_mocked_sim_data(solver_mocked_random_simulation: PolicyIteration):
    res = solver_mocked_random_simulation.solve()
    # print(solver_mocked_random_simulation.Q)
    # print(solver_mocked_random_simulation.b)
    # print(solver.intermediate_values)
    print("value:")
    print(solver_mocked_random_simulation.value)
    assert res
    assert solver_mocked_random_simulation.objective() < solver_mocked_random_simulation.convergence_tol

@pytest.fixture(params=list(range(0)))
def solver(request) -> PolicyIteration:
    random.seed(request.param)
    np.random.seed(request.param)
    d = 3
    lhs = generate_random_m_matrix(d)
    rhs = 20 * np.random.rand(d)

    return PolicyIteration(lhs, rhs)

def test_policy_iter_trivial():
    solver = PolicyIteration(
        Q = np.array([
            [3.0, -1.0, 0.0],
            [-1.0, 3.0, -1.0],
            [0.0, -1.0, 3.0],
        ]),
        b = np.array([1.0, 1.0, 1.0,])
    )
    res = solver.solve()
    # print(solver.Q)
    # print(solver.b)
    print("intermediate value:")
    print(solver.intermediate_values)
    print("intermediate obj:")
    print(solver.intermediate_objective)
    print("value:")
    print(solver.value)
    print("Q lambda - b:")
    print(solver.Q @ solver.value)
    assert res
    assert solver.objective() < solver.convergence_tol


def test_policy_iter(solver: PolicyIteration):
    res = solver.solve()
    # print(solver.Q)
    # print(solver.b)
    print("intermediate value:")
    print(solver.intermediate_values)
    print("intermediate obj:")
    print(solver.intermediate_objective)
    print("value:")
    print(solver.value)
    print("Q lambda - b:")
    print(solver.Q @ solver.value)
    assert res
    assert solver.objective() < solver.convergence_tol


#
# getting data that has actually been generated from running our simulations
# the 3x3 grid setup tends to not converge (oscillates between 2 values)
# 2x2 setup works about 93% of the time - oscillations o/w
#

NUM_SIMULATION_GENERATED_TESTS = 0
TESTS =\
    [f"../outs/ok/ok_itr_{i}.xml.out" for i in range(NUM_SIMULATION_GENERATED_TESTS)]
    # ["../K3.out", "../10.out"] +\
    # + [f"../outs/grid/itr_{i}.xml.out" for i in range(NUM_SIMULATION_GENERATED_TESTS)]

@pytest.fixture(scope="module")
def pd_data():
    return data_import.read_files_to_pd_dataframe(TESTS)
    
@pytest.fixture(params=list(range(len(TESTS))))
def solver_ball_data(request, pd_data) -> PolicyIteration:
    vi = PolicyIteration(
        pd_data['Q'][request.param],
        pd_data['b'][request.param],
    )
    vi.max_iter = 300
    return vi

def test_policy_iter_ball_data(solver_ball_data: PolicyIteration):
    res = solver_ball_data.solve()
    print("Q")
    print(solver_ball_data.Q)
    print("b")
    print(solver_ball_data.b)
    print("INTERMEDIATE VALUES")
    for iv in solver_ball_data.intermediate_values:
        print(iv)
    print(solver_ball_data.intermediate_objective)
    assert res
    assert solver_ball_data.objective() < solver_ball_data.convergence_tol


#
# off-diagonal m-matrix
# these all work perfect :)
#

NUM_RANDOM_OFF_DIAG_TESTS = 0
OFF_DIAG_SIZE = 10   # should be > 2
RANDOM_FACTOR = 0.5

CURRENT_RUN = 0

@pytest.fixture(params=list(range(NUM_RANDOM_OFF_DIAG_TESTS)))
def solver_random_off_diag_data() -> PolicyIteration:
    global CURRENT_RUN
    CURRENT_RUN += 1
    random.seed(CURRENT_RUN)
    np.random.seed(CURRENT_RUN)

    Q = 3 * np.eye(OFF_DIAG_SIZE)
    Q[0,1] = -1 + random.uniform(-RANDOM_FACTOR, RANDOM_FACTOR)
    for i in range(1, OFF_DIAG_SIZE - 1):
        Q[i, i-1] = -1 + random.uniform(-RANDOM_FACTOR, RANDOM_FACTOR)
        Q[i, i+1] = -1 + random.uniform(-RANDOM_FACTOR, RANDOM_FACTOR)
        Q[i,i] += random.uniform(-RANDOM_FACTOR, RANDOM_FACTOR)
    Q[OFF_DIAG_SIZE-1,OFF_DIAG_SIZE-2] = -1 + random.uniform(-RANDOM_FACTOR, RANDOM_FACTOR)

    b = 10 * np.random.randn(OFF_DIAG_SIZE)

    return PolicyIteration(Q, b)


def test_policy_iter_random_off_diag_data(solver_random_off_diag_data: PolicyIteration):
    res = solver_random_off_diag_data.solve()
    print("Q")
    print(solver_random_off_diag_data.Q)
    print("b")
    print(solver_random_off_diag_data.b)
    # print("INTERMEDIATE VALUES")
    # for iv in solver_random_off_diag_data.intermediate_values:
    #     print(iv)
    assert res
    assert solver_random_off_diag_data.objective() < solver_random_off_diag_data.convergence_tol

#
# Starting at the IPOPT solution
# This works 96% of the time (4% can be solved by increasing tolerance to 10^-4)
#

NUM_SIMULATION_GENERATED_TESTS_IPOPT_START = 50
TESTS_IPOPT_START = [f"../outs/grid/itr_{i}.xml.out" for i in range(NUM_SIMULATION_GENERATED_TESTS_IPOPT_START)]

@pytest.fixture(scope="module")
def pd_data_tests_ipopt_start():
    return data_import.read_files_to_pd_dataframe(TESTS_IPOPT_START)

@pytest.fixture(params=list(range(len(TESTS_IPOPT_START))))
def solver_ball_data_ipopt_start(request, pd_data_tests_ipopt_start) -> PolicyIteration:
    vi = PolicyIteration(
        pd_data_tests_ipopt_start['Q'][request.param],
        pd_data_tests_ipopt_start['b'][request.param],
        init_value=pd_data_tests_ipopt_start['ipopt_sol'][request.param],
    )
    vi.max_iter = 3
    vi.convergence_tol = 1e-4
    
    return vi

def test_policy_iter_ball_data_ipopt_start(solver_ball_data_ipopt_start: PolicyIteration):
    res = solver_ball_data_ipopt_start.solve()
    print("Q")
    print(solver_ball_data_ipopt_start.Q)
    print("b")
    print(solver_ball_data_ipopt_start.b)
    print("INTERMEDIATE VALUES")
    for iv in solver_ball_data_ipopt_start.intermediate_values:
        print(iv)
    print(solver_ball_data_ipopt_start.intermediate_objective)
    assert res
    assert solver_ball_data_ipopt_start.objective() < solver_ball_data_ipopt_start.convergence_tol


# if __name__ == "__main__":
#     test_policy_iter_trivial()
