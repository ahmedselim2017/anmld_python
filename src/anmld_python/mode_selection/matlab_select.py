from __future__ import annotations
from typing import cast

from biotite.structure import AtomArray
import jax
import jax.numpy as jnp
import loguru
import numpy as np

from anmld_python.settings import AppSettings


@jax.jit
def matlab_select(
    ca_coords_init: jax.Array,
    ca_coords_target: jax.Array,
    Vx_init: jax.Array,
    Vy_init: jax.Array,
    Vz_init: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    N_nodes, N_modes = Vx_init.shape

    # (N_nodes, 3)
    diff_vector = ca_coords_target - ca_coords_init

    # (3 * N_nodes,)
    # [Xdiff1, Ydiff1, Zdiff1, Xdiff2, Ydiff2, Zdiff2, ...]
    diff_vector = diff_vector.reshape((3 * N_nodes,))

    # (3 * N_nodes, N_modes)
    # Each column is [Xdiff1, Ydiff1, Zdiff1, Xdiff2, Ydiff2, Zdiff2, ...]
    # for one mode
    mode_vectors = jnp.stack((Vx_init, Vy_init, Vz_init), axis=1)
    mode_vectors = jnp.reshape(mode_vectors, (3 * N_nodes, N_modes))

    norm_diff = jnp.linalg.norm(diff_vector)
    norm_modes = jnp.linalg.norm(mode_vectors, axis=0)

    # (N_modes, )
    norms = norm_diff * norm_modes

    # (N_modes, )
    dots = jnp.dot(diff_vector, mode_vectors)

    # (N_modes, )
    abs_cos_sims = dots / norms

    sel_mode_idx = jnp.argmax(jnp.abs(abs_cos_sims))

    return sel_mode_idx, abs_cos_sims[sel_mode_idx]


def generate_structures(
    aa_init: AtomArray,
    aa_target: AtomArray,
    Vx_init: jax.Array,
    Vy_init: jax.Array,
    Vz_init: jax.Array,
    step_logger: loguru.Logger,
    app_settings: AppSettings,
) -> AtomArray:
    N_nodes = Vx_init.shape[0]

    # (N_nodes, mode_max)
    eig_mag = Vx_init**2 + Vy_init**2 + Vz_init**2

    # NOTE: Isn't this always have 1s at all elements as the eigecs are normalized?
    # (mode_max, )
    eig_mag_sum = eig_mag.sum(axis=0)

    # (mode_max, )
    rescale = app_settings.anmld_settings.DF / jnp.sqrt(eig_mag_sum / N_nodes)
    rescale_SC = app_settings.anmld_settings.DF_SC_ratio * rescale

    ca_init = aa_init[(aa_init.atom_name == "CA") & (aa_init.element == "C")]
    ca_target = aa_target[
        (aa_target.atom_name == "CA") & (aa_target.element == "C")
    ]

    sel_mode_idx, sel_mode_cos_sim = matlab_select(
        ca_coords_init=ca_init.coord,
        ca_coords_target=ca_target.coord,
        Vx_init=Vx_init,
        Vy_init=Vy_init,
        Vz_init=Vz_init,
    )
    step_logger.info(f"Selected mode: {sel_mode_idx + 1}")
    step_logger.info(f"Selected mode cosine sim: {sel_mode_cos_sim}")

    sel_mode_sign = jnp.sign(sel_mode_cos_sim)

    aa_pred = aa_init.copy()

    aa_nonSC_mask = np.isin(
        cast(np.ndarray, aa_pred.atom_name), ["CA", "N", "O", "C"]
    )

    mvmt_X = Vx_init[:, sel_mode_idx] * sel_mode_sign
    mvmt_Y = Vy_init[:, sel_mode_idx] * sel_mode_sign
    mvmt_Z = Vz_init[:, sel_mode_idx] * sel_mode_sign

    # TODO: vectorization?
    for i, res_id in enumerate(cast(np.ndarray, aa_pred.res_id)):
        res_mask = aa_pred.res_id == res_id

        aa_pred.coord[(res_mask) & (aa_nonSC_mask)][:, 0] += (  # type: ignore
            mvmt_X[i] * rescale[sel_mode_idx]
        )
        aa_pred.coord[(res_mask) & (aa_nonSC_mask)][:, 1] += (  # type: ignore
            mvmt_Y[i] * rescale[sel_mode_idx]
        )
        aa_pred.coord[(res_mask) & (aa_nonSC_mask)][:, 2] += (  # type: ignore
            mvmt_Z[i] * rescale[sel_mode_idx]
        )

        aa_pred.coord[(res_mask) & (~aa_nonSC_mask)][:, 0] += (  # type: ignore
            mvmt_X[i] * rescale_SC[sel_mode_idx]
        )
        aa_pred.coord[(res_mask) & (~aa_nonSC_mask)][:, 1] += (  # type: ignore
            mvmt_Y[i] * rescale_SC[sel_mode_idx]
        )
        aa_pred.coord[(res_mask) & (~aa_nonSC_mask)][:, 2] += (  # type: ignore
            mvmt_Z[i] * rescale_SC[sel_mode_idx]
        )

    return aa_pred
