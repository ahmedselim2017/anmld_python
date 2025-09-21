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
from anmld_python.tools import get_atomarray, sanitize_pdb


@logger.catch(reraise=True)
def main(
    settings_path: Path,
    path_abs_structure_init: Path,
    path_abs_structure_target: Path,
    chain_init: Optional[str] = None,
    chain_target: Optional[str] = None
):
    with open(settings_path, "rb") as settings_f:
        app_settings = AppSettings(**tomllib.load(settings_f))

    logger.remove()
    logger_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>STEP {extra[step]}</level> | "
        "<level>{message}</level> | "
        "<level>{extra}</level>"
    )
    logger.add(
        lambda msg: tqdm.write(msg, end=""),
        colorize=True,
        format=logger_format,
    )

    step_logger = logger.bind(step=-1)

    step_logger.info("Starting ANM-LD python.")
    step_logger.info(f"Version: {importlib.metadata.version('anmld_python')}")

    PS = app_settings.path_settings
    PS.out_dir = PS.out_dir.absolute()

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

    mm_min_sim = mm_ld_sim = None
    aa_step = aa_target = None
    for step in tqdm(
        range(app_settings.anmld_settings.n_steps),
        desc="Running ANM-LD",
    ):
        step_logger = logger.bind(step=step)
        ld_logger = step_logger.bind(LD=True)

        step_logger.info("Starting step")

        if step == 0:
            match app_settings.LD_method:
                case "OpenMM":  # TODO: Convert AMBER setup to openMM
                    from anmld_python.ld.openmm import run_setup, setup_sims
                    import openmm.app as mm_app

                    anm_pdb = mm_app.PDBFile(
                        str(PS.out_dir / PS.sanitized_init_structure)
                    )  # TODO: get it from biotite

                    mm_min_sim, mm_ld_sim = setup_sims(
                        topology=anm_pdb.topology,
                        app_settings=app_settings,
                    )

                    # TODO: paths
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

                    aa_step = get_atomarray(PS.out_dir / PS.sanitized_init_structure)

                    resnum = np.unique(aa_step.res_id).size  # type: ignore
                    # TODO: Paths
                    run_setup(
                        path_abs_init=PS.out_dir / PS.sanitized_init_structure,
                        path_abs_target=PS.out_dir / PS.sanitized_target_structure,
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


        run_step(
            aa_step=aa_step,
            aa_target=aa_target,
            step=step,
            step_logger=step_logger,
            ld_logger=ld_logger,
            app_settings=app_settings,
            mm_min_sim=mm_min_sim,
            mm_ld_sim=mm_ld_sim,
        )
        new_structure_name = PS.step_path_settings.step_anmld_pdb.format(
            step=step
        )
        aa_step = get_atomarray(PS.out_dir / new_structure_name)


if __name__ == "__main__":
    typer.run(main)
