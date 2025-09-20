from pathlib import Path
from textwrap import dedent
import importlib.metadata
from loguru import logger
import subprocess
import tomllib

from tqdm import tqdm
import numpy as np
import typer

from anmld_python.runner import run_step
from anmld_python.settings import AppSettings
from anmld_python.tools import get_atomarray, sanitize_pdb


@logger.catch(reraise=True)
def main(settings_path: Path, structure_init: Path, structure_target: Path):
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

    structure_init = structure_init.absolute()
    structure_target = structure_target.absolute()
    PS.out_dir = PS.out_dir.absolute()

    app_settings.subprocess_settings.cwd = PS.out_dir

    PS.out_dir.mkdir(parents=True)
    step_logger.trace(f"Created output directory at {PS.out_dir}")

    step_logger.add(PS.out_dir / "anmld.log", serialize=True)

    aa_step = sanitize_pdb(
        structure_init, PS.out_dir / PS.sanitized_init_pdb_path
    )
    aa_target = sanitize_pdb(
        structure_target, PS.out_dir / PS.sanitized_target_pdb_path
    )
    step_logger.info("Sanitized initial and target structures")

    for step in tqdm(
        range(app_settings.anmld_settings.n_steps),
        desc="Running ANM-LD",
    ):
        step_logger = logger.bind(step=step)
        ld_logger = step_logger.bind(LD=True)

        step_logger.info("Starting step")

        if step == 0:
            match app_settings.LD_method:
                case _:  # TODO: add openmm setup
                    from anmld_python.ld.amber import run_setup

                    resnum = np.unique(aa_step.res_id).size  # type: ignore
                    run_setup(
                        resnum=resnum,
                        ld_logger=ld_logger,
                        app_settings=app_settings,
                    )

            aa_step = get_atomarray(PS.out_dir / PS.amber_pdb_initial_min_pdb)
            aa_target = get_atomarray(PS.out_dir / PS.amber_pdb_target_min_pdb)

            # TODO
            if True:
                aa_step = get_atomarray(Path("deneme.pdb"))

        run_step(
            aa_step,
            aa_target,
            step,
            step_logger,
            ld_logger,
            app_settings,
        )
        new_structure_name = PS.step_path_settings.step_anmld_pdb.format(
            step=step
        )
        aa_step = get_atomarray(PS.out_dir / new_structure_name)


if __name__ == "__main__":
    typer.run(main)
