from functools import partial

import jax
import jax.numpy as jnp


@jax.jit
def build_hessian(coords: jax.Array, cutoff: float = 15.0, gamma: float = 1.0) -> jax.Array:
    N_nodes = coords.shape[0]
    sq_cutoff = cutoff**2

    # (N_nodes, N_nodes, 3)
    dists = coords[None, :, :] - coords[:, None, :]

    # (N_nodes, N_nodes)
    sq_dists = jnp.sum(dists**2, axis=-1)

    # (N_nodes, N_nodes)
    cutoff_mask = (sq_dists < sq_cutoff) & (sq_dists > 0)

    # (N_nodes, N_nodes). 0 when sq_dist is 0
    inv_sq_dist = jnp.where(cutoff_mask, 1.0 / sq_dists, 0)

    # (N_nodes, N_nodes, 3, 3)
    superelements = -gamma * jnp.einsum("...i,...j->...ij", dists, dists) * inv_sq_dist[..., None, None]

    # (N_nodes,  3, 3)
    diag_superelements = -jnp.sum(superelements, axis=0)

    # (N_nodes, N_nodes, 3, 3)
    superelements = superelements.at[jnp.arange(N_nodes), jnp.arange(N_nodes)].set(diag_superelements)

    # (N_nodes, N_nodes, 3, 3) -> (N_modes, 3, N_modes, 3) -> (3 * N_nodes, 3 * N_modes)
    hessian = superelements.transpose(0, 2, 1, 3).reshape(3 * N_nodes, 3 * N_nodes)

    return hessian


def calc_modes(
    hessian: jax.Array,
    remove_trivial_modes: bool = True,
    trivial_cutoff: float = 1e-7,
) -> tuple[jax.Array, jax.Array]:
    L, Q = jnp.linalg.eigh(hessian)

    if remove_trivial_modes:
        zero_mask = L > trivial_cutoff
        L = L[zero_mask]
        Q = Q[:, zero_mask]

    return L, Q
