"""Input construction used by the networks in the paper."""
import numpy as np


def standardize_voltage(v):
    """Per-frame z-score of the difference voltages (object minus empty tank)."""
    v = np.asarray(v, dtype=np.float32).ravel()
    return (v - v.mean()) / (v.std() + 1e-6)


def jmap(v_std, A):
    """Back-project standardized voltages with the simulated sensitivity operator A (n_meas x 4096)."""
    return (A.T @ v_std).reshape(64, 64).astype(np.float32)


def network_inputs(dv, z0, A):
    """Return (image input [z0, J-map] of shape (2, 64, 64), standardized voltage vector).

    dv: difference voltages; z0: 64x64 initial linear reconstruction (NaN outside the tank set to 0);
    A: sensitivity operator of the same measurement protocol, computed from the system's simulator.
    """
    v = standardize_voltage(dv)
    z0 = np.nan_to_num(np.asarray(z0, dtype=np.float32), nan=0.0)
    return np.stack([z0, jmap(v, A)], axis=0), v
