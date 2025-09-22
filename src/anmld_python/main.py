from pathlib import Path
from textwrap import dedent
import importlib.metadata
from typing import Optional
from loguru import logger
import tomllib

from tqdm import tqdm
import numpy as np
import typer

from anmld_python.runner import run_step
from anmld_python.settings import AppSettings
from anmld_python.tools import LDError, get_atomarray, sanitize_pdb


@logger.catch(reraise=True)
def main(
    settings_path: Path,
    path_abs_structure_init: Path,
    path_abs_structure_target: Path,
    chain_init: Optional[str] = None,
    chain_target: Optional[str] = None,
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
    logger.add(
        lambda msg: tqdm.write(msg, end=""),
        colorize=True,
        level=app_settings.logging_level,
        format=logger_format,
    )

    step_logger = logger.bind(step=-1)

    step_logger.info("Starting ANM-LD python.")
    step_logger.info(f"Version: {importlib.metadata.version('anmld_python')}")

    PS = app_settings.path_settings
    PS.out_dir = PS.out_dir.absolute()

    if (
        app_settings.LD_method == "AMBER"
        and app_settings.anmld_settings.early_stopping_rmsd
    ):
        logger.warning("Early stopping is only availible for OpenMM")

    app_settings.subprocess_settings.cwd = PS.out_dir

    PS.out_dir.mkdir(parents=True)
    step_logger.trace(f"Created output directory at {PS.out_dir}")

    step_logger.add(PS.out_dir / "anmld.log", serialize=True)

    step_logger.info("Sanitizing the initial and target structures")
    aa_step = sanitize_pdb(
        in_path=path_abs_structure_init.absolute(),
        out_path=PS.out_dir / PS.sanitized_init_structure,
        app_settings=app_settings,
        chain_id=chain_init,
    )
    aa_target = sanitize_pdb(
        in_path=path_abs_structure_target.absolute(),
        out_path=PS.out_dir / PS.sanitized_target_structure,
        app_settings=app_settings,
        chain_id=chain_target,
    )

    # TODO: A better way?
    if not (aa_step.res_id == aa_target.res_id).all():
        raise ValueError("The initial and target structures must have the same residues")

    pbar = tqdm(
        total=app_settings.anmld_settings.n_steps,
        desc="Running ANM-LD",
    )
    mm_min_sim = mm_ld_sim = rmsd = None
    step = 0
    while True:
        if step >= app_settings.anmld_settings.n_steps:
            break
        elif rmsd and app_settings.anmld_settings.early_stopping_rmsd:
            if rmsd < app_settings.anmld_settings.early_stopping_rmsd:
                step_logger.success(
                    (
                        f"Early stopping with {rmsd=}, which is below the given"
                        f"threshold {app_settings.anmld_settings.early_stopping_rmsd}"
                    )
                )
                break

        step_logger = logger.bind(step=step)
        ld_logger = step_logger.bind(LD=True)

        step_logger.info("Starting step")

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
            rmsd = run_step(
                aa_step=aa_step,  # type: ignore
                aa_target=aa_target,  # type: ignore
                step=step,
                step_logger=step_logger,
                ld_logger=ld_logger,
                app_settings=app_settings,
                mm_min_sim=mm_min_sim,
                mm_ld_sim=mm_ld_sim,
            )
        except LDError:
            ld_logger.warning(
                f"The LD simulation returned an error. Halving the DF value to {app_settings.anmld_settings.DF / 2} and retrying."
            )
            app_settings.anmld_settings.DF /= 2
            continue

        new_structure_name = PS.step_path_settings.step_anmld_pdb.format(
            step=step
        )
        aa_step = get_atomarray(PS.out_dir / new_structure_name)

        step += 1
        pbar.update(1)

    pbar.close()


if __name__ == "__main__":
    typer.run(main)
