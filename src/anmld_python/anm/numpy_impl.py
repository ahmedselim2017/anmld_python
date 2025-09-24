import numpy as np


def build_hessian(
    coords: np.ndarray,
    cutoff: float = 15.0,
    gamma: float = 1.0,
) -> np.ndarray:
    """
    Calculate the Hessian matrix for the given CA coordinates.

    Args:
        coords: (N_nodes, 3) CA coordinates array
        cutoff: ANM cutoff value
        gamma: ANM gamma value

    Returns:
        Hessian matrix with a shape of (3 * N_nodes, 3 * N_nodes)

    """
    N_nodes = coords.shape[0]
    sq_cutoff = cutoff**2

    # (N_nodes, N_nodes, 3)
    dists = coords[None, :, :] - coords[:, None, :]

    # (N_nodes, N_nodes)
    sq_dists = np.sum(dists**2, axis=-1)

    # (N_nodes, N_nodes)
    cutoff_mask = (sq_dists < sq_cutoff) & (sq_dists > 0)

    # (N_nodes, N_nodes). 0 when sq_dist is 0
    inv_sq_dist = np.divide(
        1,
        sq_dists,
        out=np.zeros_like(sq_dists),
        where=cutoff_mask,
    )

    # (N_nodes, N_nodes, 3, 3)
    sup_els = (
        -gamma
        * np.einsum("...i,...j->...ij", dists, dists)
        * inv_sq_dist[..., None, None]
    )

    # (N_nodes,  3, 3)
    diag_sup_els = -np.sum(sup_els, axis=0)

    # (N_nodes, N_nodes, 3, 3)
    diag_idx = np.arange(N_nodes)
    sup_els[diag_idx, diag_idx] = diag_sup_els

    # (N_nodes, N_nodes, 3, 3) -> (N_modes, 3, N_modes, 3) -> (3 * N_nodes, 3 * N_modes)
    hessian = sup_els.transpose(0, 2, 1, 3).reshape(3 * N_nodes, 3 * N_nodes)

    return hessian


def calc_modes(
    hessian: np.ndarray,
    mode_max: int = 30,
) -> tuple[np.ndarray, ...]:
    """
    Calcuate the modes for the given Hessian matrix.

    NOTE: The first 6 trivial modes are removed.

    Args:
        hessian: (3 * N_nodex, 3 * N_nodes) Hessian matrix
        mode_max: Number of non-trivial modes that should be calculated

    Returns:
        (mode_max,) shaped eigenvalues array
        (3 * N_nodes, mode_max) eigenvectors array
        (N_nodes, mode_max) X-eigenvalues array
        (N_nodes, mode_max) Y-eigenvalues array
        (N_nodes, mode_max) Z-eigenvalues array
    """
    W, V = np.linalg.eigh(hessian)

    W = W[6 : mode_max + 6]
    V = V[:, 6 : mode_max + 6]

    Vx = V[np.arange(0, hessian.shape[0], 3), :]
    Vy = V[np.arange(1, hessian.shape[0], 3), :]
    Vz = V[np.arange(2, hessian.shape[0], 3), :]

    return W, V, Vx, Vy, Vz
