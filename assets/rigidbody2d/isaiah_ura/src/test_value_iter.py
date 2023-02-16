from typing import Tuple
from value_iter import ValueIteration
import pytest
import numpy as np
import random
import data_import


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
def solver_mocked_random_simulation(request) -> ValueIteration:
    random.seed(request.param)
    np.random.seed(request.param)
    d = 3
    lhs = generate_mocked_random_simulation_Q(d)
    rhs = np.random.rand(d)
    return ValueIteration(lhs, rhs)


def test_value_iter_mocked_sim_data(solver_mocked_random_simulation: ValueIteration):
    res = solver_mocked_random_simulation.solve()
    # print(solver_mocked_random_simulation.Q)
    # print(solver_mocked_random_simulation.b)
    # print(solver.intermediate_values)
    print("value:")
    print(solver_mocked_random_simulation.value)
    assert res
    assert solver_mocked_random_simulation.objective() < solver_mocked_random_simulation.convergence_tol

@pytest.fixture(params=list(range(0)))
def solver(request) -> ValueIteration:
    random.seed(request.param)
    np.random.seed(request.param)
    d = 3
    lhs = generate_random_m_matrix(d)
    rhs = 20 * np.random.rand(d)

    return ValueIteration(lhs, rhs)

def test_value_iter_trivial():
    solver = ValueIteration(
        Q = np.array([
            [2.0, -1.0, 0.0],
            [-1.0, 2.0, -1.0],
            [0.0, -1.0, 2.0],
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


def test_value_iter(solver: ValueIteration):
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
#

NUM_SIMULATION_GENERATED_TESTS = 10
TESTS = ["../K3.out", "../10.out"]  + [f"../outs/grid/itr_{i}.xml.out" for i in range(NUM_SIMULATION_GENERATED_TESTS)]

@pytest.fixture(scope="module")
def pd_data():
    return data_import.read_files_to_pd_dataframe(TESTS)
    
@pytest.fixture(params=list(range(len(TESTS))))
def solver_ball_data(request, pd_data) -> ValueIteration:
    vi = ValueIteration(
        pd_data['Q'][request.param],
        pd_data['b'][request.param],
    )
    vi.max_iter = 300
    return vi

def test_value_iter_ball_data(solver_ball_data: ValueIteration):
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
def solver_random_off_diag_data() -> ValueIteration:
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

    return ValueIteration(Q, b)


def test_value_iter_random_off_diag_data(solver_random_off_diag_data: ValueIteration):
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