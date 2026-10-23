from pathlib import Path

import numpy as np
from scipy.io import loadmat

def repetition_code_XXZZ(number_of_qubit,pauli: str):
    stabs = []
    n=number_of_qubit
    for i in range(n - 1):

        x = [0] * n
        z = [0] * n

        if pauli == 'X':

            x[i] = 1
            x[i + 1] = 1

        elif pauli == 'Z':

            z[i] = 1
            z[i + 1] = 1

        elif pauli == 'Y':

            x[i] = 1
            x[i + 1] = 1
            z[i] = 1
            z[i + 1] = 1

        stabs.append(x + z)

    return np.array(stabs)


def import_gen_matrix(distance: int, folder_prefix: str, drop_last_rows=True, return_logical_parts=False):
    """
    Import a CSS stabilizer generator matrix from a .mat file dynamically.

    The files are expected to be located inside a directory matching the prefix
    (e.g., 'rt_g2') and named '{folder_prefix}_d{distance}.mat'.
    The returned generator matrix has block form:
        [ Hx  0 ]
        [ 0   Hz]

    Args:
        distance: Code distance, for example 6, 8, 10, ..., 30.
        folder_prefix: Directory and file prefix (e.g., "rt_g2" or "st_g3").
        drop_last_rows: If True, remove the final row of Hx and Hz.
        return_logical_parts: If True, return (S, Hx, Hz, Lx, Lz).

    Returns:
        S, Hx, Hz (or more if return_logical_parts=True).
    """
    # Dynamically construct the path: e.g., "rt_g2/rt_g2_d25.mat"
    mat_path = f"stabilizer-matrices" / Path(folder_prefix) / f"{folder_prefix}_d{distance}.mat"

    if not mat_path.exists():
        # Dynamically search the correct folder for available distances
        available = sorted(
            int(path.stem.split("_d")[-1])
            for path in Path(folder_prefix).glob(f"{folder_prefix}_d*.mat")
            if path.stem.split("_d")[-1].isdigit()
        )
        raise FileNotFoundError(
            f"Could not find {mat_path}. Available distances in '{folder_prefix}': {available}"
        )

    data = loadmat(mat_path)
    required_keys = ("Hx", "Hz", "Lx", "Lz")
    missing = [key for key in required_keys if key not in data]
    if missing:
        raise KeyError(f"{mat_path} is missing required matrix keys: {missing}")

    Hx = np.asarray(data["Hx"], dtype=int) % 2
    Hz = np.asarray(data["Hz"], dtype=int) % 2
    Lx = np.asarray(data["Lx"], dtype=int) % 2
    Lz = np.asarray(data["Lz"], dtype=int) % 2

    if Hx.shape[1] != Hz.shape[1]:
        raise ValueError(f"Hx and Hz must have the same number of columns: {Hx.shape}, {Hz.shape}")

    if drop_last_rows:
        Hx = Hx[:-1, :]
        Hz = Hz[:-1, :]

    n = Hx.shape[1]
    zero_x = np.zeros((Hx.shape[0], n), dtype=int)
    zero_z = np.zeros((Hz.shape[0], n), dtype=int)
    S = np.block([[Hx, zero_x], [zero_z, Hz]]) % 2

    if return_logical_parts:
        return S, Hx, Hz, Lx, Lz

    # Matches the unpacking expected by prepare_distance_context() in main.py
    return S, Hx, Hz



shor_code_913=np.array([
        # Phase-flip stabilizers (X-type)
        [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # X1 X2 X3 X4 X5 X6
        [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # X4 X5 X6 X7 X8 X9

        # Bit-flip stabilizers (Z-type)
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],  # Z1 Z2
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],  # Z2 Z3
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],  # Z4 Z5
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],  # Z5 Z6
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],  # Z7 Z8
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]  # Z8 Z9
    ],
        dtype=int,
    )



def steane_code_713():

    G = np.array([

        # X-type stabilizers
        [0,0,0,1,0,1,1,  0,0,0,0,0,0,0],
        [0,0,0,0,1,1,1,  0,0,0,0,0,0,0],
        [0,0,0,1,1,0,1,  0,0,0,0,0,0,0],

        # Z-type stabilizers
        [0,0,0,0,0,0,0,  1,0,0,1,0,1,1],
        [0,0,0,0,0,0,0,  0,1,0,1,1,0,1],
        [0,0,0,0,0,0,0,  0,0,1,0,1,1,1],

    ], dtype=int)

    return G





import numpy as np

def get_rotated_surface_code_matrix(distance: int):
    """
    Generates the binary parity-check matrices for a rotated surface code.

    Args:
        distance (int): The distance of the surface code (must be an odd integer).

    Returns:
        tuple: (H, H_X, H_Z)
            H (np.ndarray): Direct sum [ [H_X, 0], [0, H_Z] ] of shape ((d^2 - 1), 2 * d^2)
            H_X (np.ndarray): Dense X-stabilizer matrix of shape (num_x, d^2)
            H_Z (np.ndarray): Dense Z-stabilizer matrix of shape (num_z, d^2)
    """
    if distance % 2 == 0 or distance < 3:
        raise ValueError("Rotated surface codes typically require an odd distance >= 3.")

    d = distance
    n = d * d  # Total number of data qubits

    # Map (row, col) coordinates to an integer qubit ID (0 to d^2 - 1)
    def get_id(r, c):
        return r * d + c

    x_stabilizers = []
    z_stabilizers = []

    # 1. Bulk Stabilizers (Weight-4)
    for r in range(d - 1):
        for c in range(d - 1):
            qubits = [get_id(r, c), get_id(r, c + 1), get_id(r + 1, c), get_id(r + 1, c + 1)]
            if (r + c) % 2 == 0:
                x_stabilizers.append(qubits)
            else:
                z_stabilizers.append(qubits)

    # 2. Boundary Stabilizers (Weight-2)
    for i in range(d - 1):
        # Left boundary X (column 0)
        if i % 2 != 0:
            x_stabilizers.append([get_id(i, 0), get_id(i + 1, 0)])

        # Right boundary X (column d-1)
        if i % 2 == 0:
            x_stabilizers.append([get_id(i, d - 1), get_id(i + 1, d - 1)])

        # Top boundary Z (row 0)
        if i % 2 == 0:
            z_stabilizers.append([get_id(0, i), get_id(0, i + 1)])

        # Bottom boundary Z (row d-1)
        if i % 2 != 0:
            z_stabilizers.append([get_id(d - 1, i), get_id(d - 1, i + 1)])

    # 3. Construct Binary Sub-Matrices (H_X and H_Z)
    num_x = len(x_stabilizers)
    num_z = len(z_stabilizers)

    H_X = np.zeros((num_x, n), dtype=int)
    H_Z = np.zeros((num_z, n), dtype=int)

    for i, stab in enumerate(x_stabilizers):
        for q in stab:
            H_X[i, q] = 1

    for i, stab in enumerate(z_stabilizers):
        for q in stab:
            H_Z[i, q] = 1

    # 4. Construct the Direct Sum: H = H_X ⊕ H_Z
    # Structure: [ H_X   0  ]
    #            [  0   H_Z ]
    top_block = np.hstack((H_X, np.zeros((num_x, n), dtype=int)))
    bottom_block = np.hstack((np.zeros((num_z, n), dtype=int), H_Z))

    H = np.vstack((top_block, bottom_block))

    return H, H_X, H_Z






