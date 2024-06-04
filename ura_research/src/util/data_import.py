from jsoncomment import JsonComment
json = JsonComment()
import math
import numpy as np
import pandas as pd
import re

def read_sparse_matrix(m_json):
    m_rows = m_json['shape'][0]
    m_cols = m_json['shape'][1]
    m = np.zeros((m_rows, m_cols))
    for nnz_triple_str in m_json['data']:
        i = nnz_triple_str[0]
        j = nnz_triple_str[1]
        val = nnz_triple_str[2]
        m[i,j] = val
    
    # if the number of columns == 1, then this was a vector we just read in... output a 1D numpy array
    if m_cols == 1:
        return m.T[0]
    return m

def read_square_matrix(file):
    """
    this function isnt used... just here incase we might need it in the future
    function that reads in a square matrix from the file if the matrix is printed element by element separated by whitespace like:
    1 0 0
    2 1 0
    3 2 1
    """
    m = []
    cur_row = 1
    max_rows = 1    # placeholder value... since max_rows == # cols, we update this value in the loop
    while cur_row <= max_rows:
        line = file.readline()
        data = [float(x) for x in line.split()]
        m.append(data)
        if cur_row == 1:
            max_rows = len(data)    # number of rows = number of columns
        cur_row += 1
    
    return m


def read_data(f_path):
    with open(f_path) as f:
        f_str = f.read().replace(' ', '')
        f_data = json.loads(f_str)
        Minv = read_sparse_matrix(f_data['Minv'])
        data = {
            'N': [read_sparse_matrix(f_data['N'])],
            'Q': [read_sparse_matrix(f_data['Q'])],
            'v0': [read_sparse_matrix(f_data['v0'])],
            'Minv': [Minv],
            'ipopt_sol': [read_sparse_matrix(f_data['ipopt_sol'])],
            'policy_sol': [read_sparse_matrix(f_data['policy_sol'])],
            'ipopt_converged': [f_data['ipopt_converges']],
            'policy_converged': [f_data['policy_converges']],
            'policy_v2_converged': [f_data['policy_v2_converges']],
            'ipopt_time': [f_data['ipopt_time']],
            'policy_time': [f_data['pi_time']],
            'policy_v2_time': [f_data['pi_v2_time']],
            'f_name':[f_path]
        }
        return data

def inverse_with_default(x, default):
    """
    tries to compute the inverse of x, returns the given default if no good
    """
    try:
        return np.linalg.inv(x)
    except:
        return default

def min_val_with_default(x, default):
    """
    tries to compute the minimum value of x, returns the given default if no good
    """
    try:
        return np.amin(x)
    except:
        return default

def lt_0_with_default(x, default):
    """
    tries to count the number of elements in x that are less than 0,
    returns the given default if no good
    """
    try:
        return np.count_nonzero(x < 0)
    except:
        return default

def f(x):
    # print(x)
    # TODO: sometimes x is a 1x1 matrix... eig breaks on that case!
    return []
    return np.linalg.eig(x)[0]

def get_kinetic_energy_delta(pd_data, i):
    M = pd_data['M'][i]
    v_0 = pd_data['v0'][i]
    Minv = pd_data['Minv'][i]
    N = pd_data['N'][i]
    ipopt_sol = pd_data['ipopt_sol'][i]
    v_1 = v_0 + Minv @ N @ ipopt_sol
    
    ke_0 = 0.5 * v_0.T @ M @ v_0
    ke_1 = 0.5 * v_1.T @ M @ v_1

    return abs(ke_0 - ke_1)

def perform_calcs_on_dataframe(pd_data, tol = 1e-06, print_status = False):
    """
    given a pandas dataframe "pd_data" that only has columns for "Q", "policy_converged", and optionally "f_name",
    computes many extra columns for each row based on (mostly) the Q column so that we can so a little bit of analysisizing
    """
    if print_status:
        print("size...")
    Q_size = [x.shape[0] for x in pd_data['Q']]
    if print_status:
        print("inverse calculations...")
    Q_inv = [inverse_with_default(x, None) for x in pd_data['Q']]
    Q_inv_min_val = [min_val_with_default(x, None) for x in Q_inv]
    Q_inv_num_lt_0 = [lt_0_with_default(x, None) for x in Q_inv]
    if print_status:
        print("eigenvalues...")
    # print(pd_data['Q'])
    eigenvalues = [f(x) for x in pd_data['Q']]
    if print_status:
        print("condition numbers...")
    cond_num = [(np.linalg.cond(x) if x.ndim > 1 else 1) for x in pd_data['Q']]

    # many of the columns are just a scaled colum from the identity matrix (i.e.
    # all rows of col i are 0 except for row i). We care about the interesting columns...
    # i.e. those with more than 1 non-zero
    if print_status:
        print("off-diags...")
    Q_num_off_diag_entries = [(np.count_nonzero(x) - np.shape(x)[0]) for x in pd_data['Q']]

    # angles...
    if print_status:
        print("angles...")
    thetas = []
    min_theta = []
    for x in pd_data['Q']:
        if x.ndim <= 1:
            thetas.append([])
            min_theta.append(0)
            continue
        new_thetas = []
        for row in range(x.shape[0]):
            for col in range(x.shape[1]):
                val = x[row,col]
                
                # skip over the diagonal
                if col == row or val == 0:  # THIS IS WRONG... IF ANGLE IS 90 THEN val == 0 too!!!
                    continue
                
                # the Q matrix is scaled by something called the inverse mass matrix...
                # to reverse this effect (in the case where all the masses of the balls are the same)
                # we can just take a look at the value of the diagonals since
                # each diagonal value = M_inverse * 2
                m_inv_scaling_factor = x[col,col] / 2
                # if cos_theta is < -1 or > 1, then im pretty sure this element corresponds to a wall-ball collision
                # we don't really care about that case
                cos_theta = val / m_inv_scaling_factor
                if -1 > cos_theta or 1 < cos_theta:
                    continue
                new_thetas.append(math.acos(cos_theta) * 180 / math.pi)
        thetas.append(new_thetas.copy())
        min_theta.append(min(new_thetas + [math.inf]))


    pd_data['Q_size'] = Q_size
    pd_data['Q_inv'] = Q_inv
    pd_data['Q_inv_min_val'] = Q_inv_min_val
    pd_data['Q_inv_num_lt_0'] = Q_inv_num_lt_0
    pd_data['Q_det'] = [np.linalg.det(Q) if Q.ndim > 1 else 1 for Q in pd_data['Q']]
    pd_data['eigenvalues'] = eigenvalues
    pd_data['cond_num'] = cond_num

    # 10^18 is pretty close to infinity...
    # if we change all infinities to 10e18, then we can graph our condition numbers
    # without having to fear a math.inf error
    pd_data['cond_num_no_inf'] = [min(10e18, x) for x in pd_data['cond_num']]

    pd_data['Q_num_off_diag_entries'] = Q_num_off_diag_entries
    pd_data['thetas'] = thetas
    pd_data['min_theta'] = min_theta

    if print_status:
        print("rank & nullity")

    pd_data['rank'] = [np.linalg.matrix_rank(x, 1e-05) for x in pd_data.Q]
    pd_data['nullity'] = [
        pd_data['Q_size'][i] - pd_data['rank'][i]
        for i in range(len(pd_data))
    ]

    pd_data['ipopt_delta_KE'] = [
        0 for i in range(len(pd_data))
        # get_kinetic_energy_delta(pd_data, i) for i in range(len(pd_data))
    ]

    policies = []
    for i, solution in enumerate(pd_data['ipopt_sol']):
        correct_policy = [j for j in range(len(solution)) if solution[j] > tol] 
        policies.append(correct_policy.copy())

    pd_data['ipopt_policy'] = policies

    pd_data['b'] = pd_data.apply(lambda r: -2 * r['N'].T @ r['v0'], axis=1)

    if print_status:
        print("done")


def read_file_to_pd_dataframe(f_name, tol = 1e-06, print_status = False):
    if print_status:
        print(f"reading data from: {f_name}")
    file_data = read_data(f_name)
    if file_data == None:
        return None
    pd_data = pd.DataFrame(file_data)
    pd_data['b'] = pd_data.apply(lambda r: -2 * r['N'].T @ r['v0'], axis=1)
    # perform_calcs_on_dataframe(pd_data, tol)
    return pd_data

def read_files_to_pd_dataframe(f_names, tol = 1e-06, print_status = False):
    data_frames = [read_file_to_pd_dataframe(f, tol, print_status) for f in f_names]
    # print(data_frames)
    return pd.concat(data_frames).reset_index()
