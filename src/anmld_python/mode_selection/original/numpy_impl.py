from __future__ import annotations
from typing import cast

from biotite.structure import AtomArray
import biotite.structure as b_structure
import loguru
import numpy as np

from anmld_python.settings import AppSettings
from anmld_python.tools import get_CAs


def select_modes(
    ca_coords_step: np.ndarray,
    ca_coords_target: np.ndarray,
    V: np.ndarray,
) -> tuple[int, float]:
    N_nodes = V.shape[0] // 3

    # (N_nodes, 3)
    diff_vector = ca_coords_target - ca_coords_step

    # (3 * N_nodes,)
    # [Xdiff1, Ydiff1, Zdiff1, Xdiff2, Ydiff2, Zdiff2, ...]
    diff_vector = diff_vector.reshape((3 * N_nodes,))

    norm_diff = np.linalg.norm(diff_vector)
    norm_modes = np.linalg.norm(V, axis=0)

    # (N_modes, )
    norms = norm_diff * norm_modes

    # (N_modes, )
    dots = np.dot(diff_vector, V)

    # (N_modes, )
    abs_cos_sims = dots / norms

    sel_mode_idx = np.argmax(np.abs(abs_cos_sims))

    return sel_mode_idx, abs_cos_sims[sel_mode_idx]


def generate_structures(
    aa_step: AtomArray,
    aa_target: AtomArray,
    V: np.ndarray,
    step_logger: loguru.Logger,
    app_settings: AppSettings,
) -> tuple[AtomArray, dict]:
    N_nodes = V.shape[0] // 3

    ca_step = get_CAs(aa_step)
    ca_target = get_CAs(aa_target)

    mode_idx, sel_mode_cos_sim = select_modes(
        ca_coords_step=ca_step.coord,
        ca_coords_target=ca_target.coord,
        V=V,
    )
    step_logger.info(f"Selected mode number: {mode_idx + 1}")
    step_logger.debug(f"Selected mode cosine sim: {sel_mode_cos_sim}")

    mode_sign = np.sign(sel_mode_cos_sim)

    aa_pred = aa_step.copy()

    rescale = app_settings.anmld_settings._step_DF / np.sqrt(1 / N_nodes)

    # Each column of V is [Xdiff1, Ydiff1, Zdiff1, Xdiff2, Ydiff2, Zdiff2, ...]
    res_mvmt_X = V[0::3, mode_idx] * mode_sign * rescale
    res_mvmt_Y = V[1::3, mode_idx] * mode_sign * rescale
    res_mvmt_Z = V[2::3, mode_idx] * mode_sign * rescale

    atom_res_mvmt_X = b_structure.spread_residue_wise(aa_pred, res_mvmt_X)
    atom_res_mvmt_Y = b_structure.spread_residue_wise(aa_pred, res_mvmt_Y)
    atom_res_mvmt_Z = b_structure.spread_residue_wise(aa_pred, res_mvmt_Z)

    aa_pred.coord[:, 0] += atom_res_mvmt_X
    aa_pred.coord[:, 1] += atom_res_mvmt_Y
    aa_pred.coord[:, 2] += atom_res_mvmt_Z

    return aa_pred, {
        "mode_number": int(mode_idx) + 1,
        "cos_sim": float(sel_mode_cos_sim),
    }
