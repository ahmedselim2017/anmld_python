from __future__ import annotations

from biotite.structure import AtomArray
import fastpdb
import loguru
import numpy as np

from anmld_python.settings import AppSettings
from anmld_python.tools import get_CAs
import anmld_python.anm as ANM

def run_step(
    aa_step: AtomArray,
    aa_target: AtomArray,
    step: int,
    step_logger: loguru.Logger,
    ld_logger: loguru.Logger,
    app_settings: AppSettings,
):
    PS = app_settings.path_settings
    SP = app_settings.path_settings.step_path_settings.format_step(step)

    ca_step = get_CAs(aa_step)

    hessian_step = ANM.build_hessian(
        coords=ca_step.coord,
        cutoff=app_settings.anmld_settings.rcut_ANM,
        gamma=app_settings.anmld_settings.gamma_ANM,
    )
    step_logger.debug("Calculated the Hessian matrix")
    _, _, Vx_step, Vy_step, Vz_step = ANM.calc_modes(
        hessian=hessian_step,
        mode_max=app_settings.anmld_settings.max_mode,
    )
    step_logger.debug("Calculated ANM modes")

    mode_selection = app_settings.mode_selection

    step_logger.info(
        "Using {mode_selection} method to select modes",
        mode_selection=mode_selection,
    )
    match mode_selection:
        case "MATLAB":
            import anmld_python.mode_selection.matlab_select as matlab_select

            pred_aa = matlab_select.generate_structures(
                aa_step=aa_step,
                aa_target=aa_target,
                Vx_step=Vx_step,
                Vy_step=Vy_step,
                Vz_step=Vz_step,
                step_logger=step_logger,
                app_settings=app_settings,
            )

    pred_abs_path = PS.out_dir / SP.step_anm_pdb

    pred_file = fastpdb.PDBFile()
    pred_file.set_structure(pred_aa)
    pred_file.write(pred_abs_path)
    step_logger.info(f"Wrote raw step coordinates to {pred_abs_path}")

    match app_settings.LD_method:
        case "OpenMM":
            from anmld_python.ld.openmm import run_ld_step
            run_ld_step(
                aa_anm=pred_aa,
                aa_target=aa_target,
                pred_abs_path=pred_abs_path,
                ld_logger=ld_logger,
                app_settings=app_settings,
                step_paths=SP,
            )
        case "AMBER":
            from anmld_python.ld.amber import run_ld_step
            resnum: int = np.unique(aa_step.res_id).size  # type: ignore
            run_ld_step(
                pred_abs_path=pred_abs_path,
                resnum=resnum,
                ld_logger=ld_logger,
                app_settings=app_settings,
                SP=SP,
            )
