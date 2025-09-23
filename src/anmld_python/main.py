from __future__ import annotations
from pathlib import Path
import importlib.metadata
from typing import Optional
from loguru import logger
import tomllib

from tqdm import tqdm
import numpy as np
import typer
import pandas as pd

from anmld_python.runner import run_step
from anmld_python.settings import AppSettings
from anmld_python.tools import (
    LDError,
    NonConnectedStructureError,
    get_atomarray,
    sanitize_pdb,
)


def process_inputs(
    path_abs_structure_init: Path,
    path_abs_structure_target: Path,
    chain_init: Optional[list[str]],
    chain_target: Optional[list[str]],
    app_settings: AppSettings,
):
    PS = app_settings.path_settings
    PS.out_dir = PS.out_dir.absolute()

    app_settings.subprocess_settings.cwd = PS.out_dir

    PS.out_dir.mkdir(parents=True)
    logger.trace(f"Created output directory at {PS.out_dir}")

    logger.info("Sanitizing the initial and target structures")
    aa_step = sanitize_pdb(
        in_path=path_abs_structure_init.absolute(),
        out_path=PS.out_dir / PS.sanitized_init_structure,
        app_settings=app_settings,
        sel_chains=chain_init,
        include_bonds=True,
    )
    aa_target = sanitize_pdb(
        in_path=path_abs_structure_target.absolute(),
        out_path=PS.out_dir / PS.sanitized_target_structure,
        app_settings=app_settings,
        sel_chains=chain_target,
        include_bonds=True,
    )

    if aa_step.bonds.as_set() != aa_target.bonds.as_set():  # type: ignore
        raise ValueError(
            "The initial and target structures must have the same topology"
        )


def run_cycle(app_settings: AppSettings):
    PS = app_settings.path_settings

    pbar = tqdm(
        total=app_settings.anmld_settings.n_steps,
        desc="Running ANM-LD",
    )
    mm_min_sim = mm_ld_sim = None
    step = 0

    cycle_info = []
    while True:
        step_logger = logger.bind(step=step)
        ld_logger = step_logger.bind(LD=True)
        if step >= app_settings.anmld_settings.n_steps:
            break

        step_logger.debug("Starting step")

        if step == 0:
            match app_settings.LD_method:
                case "OpenMM":
                    from anmld_python.ld.openmm import run_setup, setup_sims
                    import openmm.app as mm_app

                    anm_pdb = mm_app.PDBFile(
                        str(PS.out_dir / PS.sanitized_init_structure)
                    )  # INFO: fastpdb can't load bondlist so load file with openmm

                    mm_min_sim, mm_ld_sim = setup_sims(
                        topology=anm_pdb.topology,
                        ld_logger=ld_logger,
                        app_settings=app_settings,
                    )

                    run_setup(
                        path_init=PS.out_dir / PS.sanitized_init_structure,
                        path_target=PS.out_dir / PS.sanitized_target_structure,
                        min_sim=mm_min_sim,
                        ld_logger=ld_logger,
                        app_settings=app_settings,
                    )

                    aa_step = get_atomarray(
                        PS.out_dir / PS.openmm_min_aligned_init_pdb
                    )
                    aa_target = get_atomarray(
                        PS.out_dir / PS.openmm_min_target_pdb
                    )
                case "AMBER":
                    from anmld_python.ld.amber import run_setup

                    aa_step = get_atomarray(
                        PS.out_dir / PS.sanitized_init_structure
                    )

                    resnum = np.unique(aa_step.res_id).size  # type: ignore
                    run_setup(
                        path_abs_init=PS.out_dir / PS.sanitized_init_structure,
                        path_abs_target=PS.out_dir
                        / PS.sanitized_target_structure,
                        resnum=resnum,
                        ld_logger=ld_logger,
                        app_settings=app_settings,
                    )

                    aa_step = get_atomarray(
                        PS.out_dir / PS.amber_pdb_initial_min_pdb
                    )
                    aa_target = get_atomarray(
                        PS.out_dir / PS.amber_pdb_target_min_pdb
                    )

        try:
            step_info = run_step(
                aa_step=aa_step,  # type: ignore
                aa_target=aa_target,  # type: ignore
                step=step,
                step_logger=step_logger,
                ld_logger=ld_logger,
                app_settings=app_settings,
                mm_min_sim=mm_min_sim,
                mm_ld_sim=mm_ld_sim,
            )
        except NonConnectedStructureError:
            if step == 0:
                raise ValueError(
                    "The given initial structure is not fully connected"
                )
            else:
                ld_logger.warning(
                    f"ANM deformation produced a non-connected structure. Halving the DF value to {app_settings.anmld_settings.DF / 2} and restarting the step"
                )
                app_settings.anmld_settings.DF /= 2
                continue

        except LDError:
            ld_logger.warning(
                f"The LD simulation returned an error. Halving the DF value to {app_settings.anmld_settings.DF / 2} and restarting the step"
            )
            app_settings.anmld_settings.DF /= 2
            continue

        new_name = PS.step_path_settings.step_anmld_pdb.format(step=step)
        aa_step = get_atomarray(PS.out_dir / new_name)

        step_info["step"] = step

        if step_info["rmsd"] < app_settings.anmld_settings.early_stopping_rmsd:
            step_logger.success(
                (
                    f"Early stopping with {float(step_info['rmsd'])} RMSD ",
                    "which is below the given threshold ",
                    f"{app_settings.anmld_settings.early_stopping_rmsd}",
                )
            )
            break

        cycle_info.append(step_info)
        step += 1
        pbar.update(1)

    pbar.close()

    cycle_df = pd.json_normalize(cycle_info)
    cycle_df.to_csv(PS.out_dir / PS.info_csv, index=False)

    return cycle_info


@logger.catch(reraise=True)
def main(
    settings_path: Path,
    path_abs_structure_init: Path,
    path_abs_structure_target: Path,
    chain_init: Optional[list[str]] = None,
    chain_target: Optional[list[str]] = None,
):
    with open(settings_path, "rb") as settings_f:
        app_settings = AppSettings(**tomllib.load(settings_f))

    logger.remove()
    logger_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<level>STEP {extra[step]: <4}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level> | "
        "<level>{extra}</level>"
    )
    logger.configure(extra={"step": -1})
    logger.add(
        lambda msg: tqdm.write(msg, end=""),
        colorize=True,
        level=app_settings.logging_level,
        format=logger_format,
    )

    logger.info("Starting ANM-LD python.")
    logger.info(f"Version: {importlib.metadata.version('anmld_python')}")

    process_inputs(
        path_abs_structure_init=path_abs_structure_init,
        path_abs_structure_target=path_abs_structure_target,
        chain_init=chain_init,
        chain_target=chain_target,
        app_settings=app_settings,
    )

    run_cycle(app_settings=app_settings)


if __name__ == "__main__":
    typer.run(main)
