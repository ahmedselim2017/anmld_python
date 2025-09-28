from __future__ import annotations
from typing import cast

from biotite.structure import AtomArray
import loguru
import numpy as np

from anmld_python.settings import AppSettings
from anmld_python.tools import get_CAs


def select_modes(
    ca_coords_step: np.ndarray,
    ca_coords_target: np.ndarray,
    Vx_step: np.ndarray,
    Vy_step: np.ndarray,
    Vz_step: np.ndarray,
) -> tuple[int, float]:
    N_nodes, N_modes = Vx_step.shape

    # (N_nodes, 3)
    diff_vector = ca_coords_target - ca_coords_step

    # (3 * N_nodes,)
    # [Xdiff1, Ydiff1, Zdiff1, Xdiff2, Ydiff2, Zdiff2, ...]
    diff_vector = diff_vector.reshape((3 * N_nodes,))

    # (3 * N_nodes, N_modes)
    # Each column is [Xdiff1, Ydiff1, Zdiff1, Xdiff2, Ydiff2, Zdiff2, ...]
    # for one mode
    mode_vectors = np.stack((Vx_step, Vy_step, Vz_step), axis=1)
    mode_vectors = np.reshape(mode_vectors, (3 * N_nodes, N_modes))

    norm_diff = np.linalg.norm(diff_vector)
    norm_modes = np.linalg.norm(mode_vectors, axis=0)

    # (N_modes, )
    norms = norm_diff * norm_modes

    # (N_modes, )
    dots = np.dot(diff_vector, mode_vectors)

    # (N_modes, )
    abs_cos_sims = dots / norms

    sel_mode_idx = np.argmax(np.abs(abs_cos_sims))

    return sel_mode_idx, abs_cos_sims[sel_mode_idx]


def generate_structures(
    aa_step: AtomArray,
    aa_target: AtomArray,
    Vx_step: np.ndarray,
    Vy_step: np.ndarray,
    Vz_step: np.ndarray,
    step_logger: loguru.Logger,
    app_settings: AppSettings,
) -> tuple[AtomArray, dict]:
    N_nodes = Vx_step.shape[0]

    # (N_nodes, mode_max)
    eig_mag = Vx_step**2 + Vy_step**2 + Vz_step**2

    # NOTE: Isn't this always have 1s at all elements as the eigecs are normalized?
    # (mode_max, )
    eig_mag_sum = eig_mag.sum(axis=0)

    # (mode_max, )
    rescale = app_settings.anmld_settings.DF / np.sqrt(eig_mag_sum / N_nodes)
    rescale_SC = app_settings.anmld_settings.DF_SC_ratio * rescale

    ca_step = get_CAs(aa_step)
    ca_target = get_CAs(aa_target)

    sel_mode_idx, sel_mode_cos_sim = select_modes(
        ca_coords_step=ca_step.coord,
        ca_coords_target=ca_target.coord,
        Vx_step=Vx_step,
        Vy_step=Vy_step,
        Vz_step=Vz_step,
    )
    step_logger.info(f"Selected mode number: {sel_mode_idx + 1}")
    step_logger.debug(f"Selected mode cosine sim: {sel_mode_cos_sim}")

    sel_mode_sign = np.sign(sel_mode_cos_sim)

    aa_pred = aa_step.copy()

    aa_nonSC_mask = np.isin(cast(np.ndarray, aa_pred.atom_name), ["CA", "N", "O", "C"])

    # TODO: needs to be imporved
    mvmt_X = np.asarray(Vx_step[:, sel_mode_idx] * sel_mode_sign)
    mvmt_Y = np.asarray(Vy_step[:, sel_mode_idx] * sel_mode_sign)
    mvmt_Z = np.asarray(Vz_step[:, sel_mode_idx] * sel_mode_sign)

    res_ids = np.unique(aa_pred.res_id)
    for i in range(len(res_ids)):
        res_id = res_ids[i]
        res_mask = aa_pred.res_id == res_id

        aa_pred.coord[(res_mask) & (aa_nonSC_mask), 0] += (  # type: ignore
            mvmt_X[i] * rescale[sel_mode_idx]
        )
        aa_pred.coord[(res_mask) & (aa_nonSC_mask), 1] += (  # type: ignore
            mvmt_Y[i] * rescale[sel_mode_idx]
        )
        aa_pred.coord[(res_mask) & (aa_nonSC_mask), 2] += (  # type: ignore
            mvmt_Z[i] * rescale[sel_mode_idx]
        )

        aa_pred.coord[(res_mask) & (~aa_nonSC_mask), 0] += (  # type: ignore
            mvmt_X[i] * rescale_SC[sel_mode_idx]
        )
        aa_pred.coord[(res_mask) & (~aa_nonSC_mask), 1] += (  # type: ignore
            mvmt_Y[i] * rescale_SC[sel_mode_idx]
        )
        aa_pred.coord[(res_mask) & (~aa_nonSC_mask), 2] += (  # type: ignore
            mvmt_Z[i] * rescale_SC[sel_mode_idx]
        )

    return aa_pred, {
        "mode_number": int(sel_mode_idx) + 1,
        "cos_sim": float(sel_mode_cos_sim),
    }
