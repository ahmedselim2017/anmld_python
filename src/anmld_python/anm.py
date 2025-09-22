from functools import partial

import jax
import jax.numpy as jnp


@jax.jit
def build_hessian(
    coords: jax.Array,
    cutoff: float = 15.0,
    gamma: float = 1.0,
) -> jax.Array:
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
    sq_dists = jnp.sum(dists**2, axis=-1)

    # (N_nodes, N_nodes)
    cutoff_mask = (sq_dists < sq_cutoff) & (sq_dists > 0)

    # (N_nodes, N_nodes). 0 when sq_dist is 0
    inv_sq_dist = jnp.where(cutoff_mask, 1.0 / (sq_dists), 0)

    # (N_nodes, N_nodes, 3, 3)
    sup_els = (
        -gamma
        * jnp.einsum("...i,...j->...ij", dists, dists)
        * inv_sq_dist[..., None, None]
    )

    # (N_nodes,  3, 3)
    diag_sup_els = -jnp.sum(sup_els, axis=0)

    # (N_nodes, N_nodes, 3, 3)
    diag_idx = jnp.arange(N_nodes)
    sup_els = sup_els.at[diag_idx, diag_idx].set(diag_sup_els)

    # (N_nodes, N_nodes, 3, 3) -> (N_modes, 3, N_modes, 3) -> (3 * N_nodes, 3 * N_modes)
    hessian = sup_els.transpose(0, 2, 1, 3).reshape(3 * N_nodes, 3 * N_nodes)

    return hessian


@partial(jax.jit, static_argnames=["mode_max"])
def calc_modes(
    hessian: jax.Array,
    mode_max: int = 30,
) -> tuple[jax.Array, ...]:
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
    W, V = jnp.linalg.eigh(hessian)

    W = W[6 : mode_max + 6]
    V = V[:, 6 : mode_max + 6]

    Vx = V[jnp.arange(0, hessian.shape[0], 3), :]
    Vy = V[jnp.arange(1, hessian.shape[0], 3), :]
    Vz = V[jnp.arange(2, hessian.shape[0], 3), :]

    return W, V, Vx, Vy, Vz
