from __future__ import annotations
from loguru import logger
from pathlib import Path
from typing import Optional
import importlib.metadata
import shutil
import tomllib

from tqdm.auto import tqdm
import biotite.structure as b_structure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import typer

from anmld_python.runner import run_step
from anmld_python.settings import AppSettings
from anmld_python.tools import (
    LDError,
    NonConnectedStructureError,
    get_atomarray,
    sanitize_structure,
)

try:
    import jax

    jax.config.update("jax_enable_x64", True)
except ModuleNotFoundError:
    logger.warning("JAX was not found. Using NumPy as fallback.")
    pass


def setup_logging(app_settings: AppSettings):
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
    logger.add(
        app_settings.path_settings._out_dir / "trace.log",
        colorize=False,
        level="TRACE",
        format=logger_format,
    )

    logger.info("Starting ANM-LD python")
    logger.info(f"Version: {importlib.metadata.version('anmld_python')}")


def process_inputs(
    path_abs_structure_init: Path,
    path_abs_structure_target: Path,
    chain_init: Optional[list[str]],
    chain_target: Optional[list[str]],
    app_settings: AppSettings,
):
    logger.trace("Processing inputs")
    PS = app_settings.path_settings
    PS._out_dir = PS._out_dir.absolute()

    shutil.copy(path_abs_structure_init, PS._out_dir / PS.init_structure)
    shutil.copy(path_abs_structure_target, PS._out_dir / PS.target_structure)

    app_settings.subprocess_settings.cwd = PS._out_dir

    if app_settings.LD_method == "OpenMM":
        import openmm as mm

        MS = app_settings.openmm_settings
        try:
            MS.platform_obj = mm.Platform.getPlatformByName(MS.platform_name)
            logger.info(f"Set OpenMM platform to {MS.platform_name}")
        except mm.OpenMMException:
            logger.warning(
                f"The given platform {MS.platform_name} is not found."
                " Using the dafault platform."
            )
            MS.platform_obj = None

    logger.info("Sanitizing the initial and target structures")
    aa_step = sanitize_structure(
        in_path=path_abs_structure_init.absolute(),
        out_path=PS._out_dir / PS.sanitized_init_structure,
        app_settings=app_settings,
        sel_chains=chain_init,
        include_bonds=True,
    )
    aa_target = sanitize_structure(
        in_path=path_abs_structure_target.absolute(),
        out_path=PS._out_dir / PS.sanitized_target_structure,
        app_settings=app_settings,
        sel_chains=chain_target,
        include_bonds=True,
    )

    if b_structure.get_residue_count(aa_step) != b_structure.get_residue_count(
        aa_target
    ):
        raise ValueError(
            "The initial and target structures must have the same residue count"
        )

    elif aa_step.bonds.as_set() != aa_target.bonds.as_set():  # type: ignore
        logger.warning(
            "The initial and target structures do not have the same topology. Only C-alpha RMSDs will be calculated."
        )
        app_settings.different_topologies = True


def run_cycle(app_settings: AppSettings) -> list[dict]:
    PS = app_settings.path_settings

    pbar = tqdm(
        total=app_settings.anmld_settings.n_steps,
        desc="Running ANM-LD",
    )
    mm_min_sim = mm_ld_sim = None
    step = 0

    cycle_info = []
    while True:
        if step >= app_settings.anmld_settings.n_steps:
            break

        step_logger = logger.bind(step=step)
        ld_logger = step_logger.bind(LD=True)
        step_logger.debug("Starting step")

        step_paths = PS.step_path_settings.format_step(
            step=step,
            n_maxdigit=len(str(app_settings.anmld_settings.n_steps)),
        )

        if step == 0:
            step_logger.trace("Starting ANM-LD setup")
            match app_settings.LD_method:
                case "OpenMM":
                    from anmld_python.ld.openmm import run_setup, setup_sims
                    import openmm.app as mm_app
                    import openmm as mm

                    anm_pdb = mm_app.PDBFile(
                        str(PS._out_dir / PS.sanitized_init_structure)
                    )  # INFO: fastpdb can't load bondlist so load file with openmm

                    mm_min_sim, mm_ld_sim = setup_sims(
                        topology=anm_pdb.topology,
                        ld_logger=ld_logger,
                        app_settings=app_settings,
                    )

                    try:
                        run_setup(
                            path_init=PS._out_dir / PS.sanitized_init_structure,
                            path_target=PS._out_dir / PS.sanitized_target_structure,
                            init_min_sim=mm_min_sim,
                            ld_logger=ld_logger,
                            app_settings=app_settings,
                        )
                    except mm.OpenMMException as e:
                        emsg = "The OpenMM step of the ANM-LD setup returned an error."
                        ld_logger.critical(emsg, err=e)
                        raise e from None

                    aa_step = get_atomarray(
                        PS._out_dir / PS.openmm_min_aligned_init_pdb
                    )
                    aa_target = get_atomarray(PS._out_dir / PS.openmm_min_target_pdb)
                case "AMBER":
                    from anmld_python.ld.amber import run_setup

                    aa_step = get_atomarray(PS._out_dir / PS.sanitized_init_structure)

                    resnum = b_structure.get_residue_count(aa_step)
                    try:
                        run_setup(
                            path_abs_init=PS._out_dir / PS.sanitized_init_structure,
                            path_abs_target=PS._out_dir / PS.sanitized_target_structure,
                            resnum=resnum,
                            ld_logger=ld_logger,
                            app_settings=app_settings,
                        )
                    except (LDError, ValueError) as e:
                        ld_logger.critical(
                            "The AMBER step of the ANM-LD setup returned"
                            " an error. For details, see AMBER outputs.",
                            err=e,
                        )
                        raise e from None

                    try:
                        aa_step = get_atomarray(
                            PS._out_dir / PS.amber_pdb_initial_min_pdb
                        )
                        aa_target = get_atomarray(
                            PS._out_dir / PS.amber_pdb_target_min_pdb
                        )
                    except Exception as e:
                        ld_logger.critical(
                            "Could not load the AMBER minimized structures"
                            f" {PS._out_dir / PS.amber_pdb_initial_min_pdb} and"
                            f" {PS._out_dir / PS.amber_pdb_target_min_pdb}"
                        )
                        raise e
            step_logger.info("Finished ANM-LD setup")

        try:
            step_info = run_step(
                aa_step=aa_step,  # type: ignore
                aa_target=aa_target,  # type: ignore
                step_logger=step_logger,
                ld_logger=ld_logger,
                step_paths=step_paths,
                app_settings=app_settings,
                mm_min_sim=mm_min_sim,
                mm_ld_sim=mm_ld_sim,
            )

        except NonConnectedStructureError:
            if step == 0:
                raise ValueError("The given initial structure is not fully connected")
            else:
                ld_logger.warning("ANM deformation produced a non-connected structure.")
                continue

        except (LDError, ValueError):
            ld_logger.warning(
                f"The LD step returned an error. Halving the DF value to"
                f" {app_settings.anmld_settings.DF / 2} and restarting the step"
            )
            app_settings.anmld_settings.DF /= 2
            continue
        aa_step = get_atomarray(PS._out_dir / step_paths.step_anmld_pdb)

        step_info["step"] = step

        if (not app_settings.different_topologies) and (
            step_info["aa_rmsd_target"]
            < app_settings.anmld_settings.early_stopping_aa_rmsd
        ):
            step_logger.success(
                f"Early stopping with {float(step_info['aa_rmsd_target'])}"
                " all-atom RMSD  which is below the given threshold"
                f" {app_settings.anmld_settings.early_stopping_aa_rmsd}"
            )
            break
        elif (
            step_info["ca_rmsd_target"]
            < app_settings.anmld_settings.early_stopping_ca_rmsd
        ):
            step_logger.success(
                f"Early stopping with {float(step_info['ca_rmsd_target'])}"
                " C-alpha RMSD  which is below the given threshold"
                f" {app_settings.anmld_settings.early_stopping_ca_rmsd}"
            )
            break

        cycle_info.append(step_info)
        step += 1
        pbar.update(1)

    pbar.close()

    return cycle_info


def analyze(cycle_info: list[dict], app_settings: AppSettings):
    PS = app_settings.path_settings

    cycle_df = pd.json_normalize(cycle_info)
    cycle_df.to_csv(PS._out_dir / PS.info_csv, index=False)
    logger.info(f"Wrote cycle results to {PS._out_dir / PS.info_csv}")

    sns.set_theme()

    fig, ax = plt.subplots()
    sns.lineplot(
        cycle_df["aa_rmsd_target"],
        label="All atom RMSD to the target",
        color=sns.color_palette()[0],
        ax=ax,
    )
    sns.lineplot(
        cycle_df["ca_rmsd_target"],
        label="C-alpha RMSD to the target",
        color=sns.color_palette()[0],
        linestyle="dashed",
        ax=ax,
    )
    sns.lineplot(
        cycle_df["aa_rmsd_init"],
        label="All atom RMSD to the initial",
        color=sns.color_palette()[1],
        ax=ax,
    )
    sns.lineplot(
        cycle_df["ca_rmsd_init"],
        label="C-alpha RMSD to the initial",
        color=sns.color_palette()[1],
        linestyle="dashed",
        ax=ax,
    )
    ax.set_xlabel("Step")
    ax.set_ylabel("RMSD (Å)")
    ax.set_title("RMSD Values With The Minimized Target and Initial Structures")
    fig.tight_layout()
    fig.savefig(PS._out_dir / PS.info_rmsd_fig)

    fig, ax = plt.subplots()
    sns.scatterplot(
        cycle_df["selection.mode_number"],
        label="Selected mode",
        legend=None,
        color=sns.color_palette()[0],
        ax=ax,
    )
    ax.set_xlabel("Step")
    ax.set_ylabel("Mode")
    ax.set_title("Selected Modes")
    fig.tight_layout()
    fig.savefig(PS._out_dir / PS.info_sel_modes_fig)

    logger.info(f"Plotted cycle results.")


def cleanup(app_settings: AppSettings):
    PS = app_settings.path_settings

    if app_settings.LD_method == "AMBER":
        Path(PS._out_dir / PS.amber_min_in).unlink(missing_ok=True)
        Path(PS._out_dir / PS.amber_sim_in).unlink(missing_ok=True)
        Path(PS._out_dir / PS.amber_pdb_init_top).unlink(missing_ok=True)
        Path(PS._out_dir / PS.amber_pdb_init_coord).unlink(missing_ok=True)
        Path(PS._out_dir / PS.amber_pdb_target_top).unlink(missing_ok=True)
        Path(PS._out_dir / PS.amber_pdb_target_coord).unlink(missing_ok=True)
        Path(PS._out_dir / PS.amber_tleap_init_in).unlink(missing_ok=True)
        Path(PS._out_dir / PS.amber_pdb_init_min_out).unlink(missing_ok=True)
        Path(PS._out_dir / PS.amber_pdb_init_min_coord).unlink(missing_ok=True)
        Path(PS._out_dir / PS.amber_pdb_init_min_rst).unlink(missing_ok=True)
        Path(PS._out_dir / PS.amber_pdb_target_min_out).unlink(missing_ok=True)
        Path(PS._out_dir / PS.amber_pdb_target_min_coord).unlink(missing_ok=True)
        Path(PS._out_dir / PS.amber_pdb_target_min_rst).unlink(missing_ok=True)
        Path(PS._out_dir / PS.amber_ptraj_rewrite_init_in).unlink(missing_ok=True)
        Path(PS._out_dir / PS.amber_pdb_rewrite_init_min_rst).unlink(missing_ok=True)
        Path(PS._out_dir / PS.amber_ptraj_align_target2initial_in).unlink(
            missing_ok=True
        )
        Path(PS._out_dir / PS.amber_rms_target_align_dat).unlink(missing_ok=True)
        Path(PS._out_dir / PS.amber_target_min_algn).unlink(missing_ok=True)
        Path(PS._out_dir / PS.amber_pdb_initial_min_pdb).unlink(missing_ok=True)
        Path(PS._out_dir / PS.amber_pdb_target_min_pdb).unlink(missing_ok=True)

    for step in range(app_settings.anmld_settings.n_steps):
        step_paths = PS.step_path_settings.format_step(
            step=step,
            n_maxdigit=len(str(app_settings.anmld_settings.n_steps)),
        )

        if app_settings.LD_method == "OpenMM":
            Path(PS._out_dir / step_paths.step_anm_pdb).unlink(missing_ok=True)
        elif app_settings.LD_method == "AMBER":
            Path(PS._out_dir / step_paths.step_amber_anm_pdb).unlink(missing_ok=True)
            Path(PS._out_dir / step_paths.step_amber_tleap_anm_pdb).unlink(
                missing_ok=True
            )
            Path(PS._out_dir / step_paths.step_amber_top).unlink(missing_ok=True)
            Path(PS._out_dir / step_paths.step_amber_coord).unlink(missing_ok=True)
            Path(PS._out_dir / step_paths.step_amber_min_out).unlink(missing_ok=True)
            Path(PS._out_dir / step_paths.step_amber_min_coord).unlink(missing_ok=True)
            Path(PS._out_dir / step_paths.step_amber_min_rst).unlink(missing_ok=True)
            Path(PS._out_dir / step_paths.step_amber_sim_out).unlink(missing_ok=True)
            Path(PS._out_dir / step_paths.step_amber_sim_coord).unlink(missing_ok=True)
            Path(PS._out_dir / step_paths.step_amber_sim_ener).unlink(missing_ok=True)
            Path(PS._out_dir / step_paths.step_amber_sim_restart).unlink(
                missing_ok=True
            )
            Path(PS._out_dir / step_paths.step_amber_ptraj_align_in).unlink(
                missing_ok=True
            )
            Path(PS._out_dir / step_paths.step_amber_ptraj_rms_align_dat).unlink(
                missing_ok=True
            )
            Path(PS._out_dir / step_paths.step_amber_ptraj_algn_restart).unlink(
                missing_ok=True
            )


def main(
    settings_path: Path,
    path_abs_structure_init: Path,
    path_abs_structure_target: Path,
    out_dir: Path = Path("anmld_out"),
    chain_init: Optional[list[str]] = None,
    chain_target: Optional[list[str]] = None,
):
    with open(settings_path, "rb") as settings_f:
        app_settings = AppSettings(**tomllib.load(settings_f))

    app_settings.path_settings._out_dir = out_dir
    app_settings.path_settings._out_dir.mkdir(parents=True)
    logger.trace(f"Created output directory at {app_settings.path_settings._out_dir}")

    setup_logging(app_settings=app_settings)

    process_inputs(
        path_abs_structure_init=path_abs_structure_init,
        path_abs_structure_target=path_abs_structure_target,
        chain_init=chain_init,
        chain_target=chain_target,
        app_settings=app_settings,
    )

    cycle_info = run_cycle(app_settings=app_settings)

    analyze(cycle_info=cycle_info, app_settings=app_settings)

    if app_settings.cleanup:
        cleanup(app_settings)


def cli():
    app = typer.Typer(pretty_exceptions_show_locals=False, add_completion=False)
    app.command()(main)
    app(standalone_mode=True)


if __name__ == "__main__":
    cli()
