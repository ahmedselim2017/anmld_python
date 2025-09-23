from __future__ import annotations
from typing import Any, Optional

from biotite.structure import AtomArray
import fastpdb
import loguru
import numpy as np

from anmld_python.settings import AppSettings, StepPathSettings
from anmld_python.tools import LDError, NonConnectedStructureError, get_CAs
import anmld_python.anm as ANM


def run_step(
    aa_step: AtomArray,
    aa_target: AtomArray,
    step_logger: loguru.Logger,
    ld_logger: loguru.Logger,
    step_paths: StepPathSettings,
    app_settings: AppSettings,
    mm_min_sim: Optional[Any] = None,
    mm_ld_sim: Optional[Any] = None,
) -> dict:
    PS = app_settings.path_settings

    ca_step = get_CAs(aa_step)

    hessian_step = ANM.build_hessian(
        coords=ca_step.coord,
        cutoff=app_settings.anmld_settings.rcut_ANM,
        gamma=app_settings.anmld_settings.gamma_ANM,
    )
    step_logger.debug("Calculated the Hessian matrix")
    W, _, Vx_step, Vy_step, Vz_step = ANM.calc_modes(
        hessian=hessian_step,
        mode_max=app_settings.anmld_settings.max_mode,
    )
    step_logger.debug("Calculated ANM modes")

    if np.any(np.allclose(W, 0)):
        raise NonConnectedStructureError(
            "The given initial structure is not fully connected"
        )

    mode_selection = app_settings.mode_selection

    step_logger.debug(
        "Using {mode_selection} method to select modes",
        mode_selection=mode_selection,
    )
    match mode_selection:
        case "ORIGINAL":
            from anmld_python.mode_selection.original_select import (
                generate_structures,
            )

            pred_aa, sel_info = generate_structures(
                aa_step=aa_step,
                aa_target=aa_target,
                Vx_step=Vx_step,
                Vy_step=Vy_step,
                Vz_step=Vz_step,
                step_logger=step_logger,
                app_settings=app_settings,
            )

    pred_abs_path = PS.out_dir / step_paths.step_anm_pdb

    pred_file = fastpdb.PDBFile()
    pred_file.set_structure(pred_aa)
    pred_file.write(pred_abs_path)
    step_logger.debug(f"Wrote raw step coordinates to {pred_abs_path}")

    match app_settings.LD_method:
        case "OpenMM":
            from anmld_python.ld.openmm import run_ld_step
            from openmm import OpenMMException
            from scipy.linalg import LinAlgError

            if mm_min_sim is None or mm_ld_sim is None:
                raise ValueError("mm_min_sim and mm_ld_sim must not be None.")

            try:
                rmsd = run_ld_step(
                    aa_anm=pred_aa,
                    aa_target=aa_target,
                    pred_abs_path=pred_abs_path,
                    min_sim=mm_min_sim,
                    ld_sim=mm_ld_sim,
                    ld_logger=ld_logger,
                    app_settings=app_settings,
                    step_paths=step_paths,
                )
            except (OpenMMException, LinAlgError):
                raise LDError
        case "AMBER":
            from anmld_python.ld.amber import run_ld_step
            from subprocess import CalledProcessError

            resnum: int = np.unique(aa_step.res_id).size  # type: ignore
            try:
                rmsd = run_ld_step(
                    pred_abs_path=pred_abs_path,
                    resnum=resnum,
                    ld_logger=ld_logger,
                    app_settings=app_settings,
                    SP=step_paths,
                )
            except CalledProcessError:
                raise LDError
    return {"rmsd": rmsd, "selection": sel_info}
