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
from anmld_python.tools import sanitize_pdb


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

    AS = app_settings.amber_settings
    PS = app_settings.path_settings

    structure_init = structure_init.absolute()
    structure_target = structure_target.absolute()
    PS.out_dir = PS.out_dir.absolute()

    app_settings.subprocess_settings.cwd = PS.out_dir

    PS.out_dir.mkdir(parents=True)
    step_logger.trace(f"Created output directory at {PS.out_dir}")

    step_logger.add(PS.out_dir / "anmld.log", serialize=True)

    aa_init = sanitize_pdb(
        structure_init, PS.out_dir / PS.sanitized_init_pdb_path
    )
    aa_target = sanitize_pdb(
        structure_target, PS.out_dir / PS.sanitized_target_pdb_path
    )
    resnum: int = np.unique(aa_init.res_id).size
    step_logger.info("Sanitized initial and target structures")

    for step in tqdm(
        range(app_settings.anmld_settings.n_steps), desc="Running ANM-LD"
    ):
        step_logger = logger.bind(step=step)
        amber_logger = step_logger.bind(isAMBER=True)

        step_logger.info("Starting step")

        if step == 0:
            with open(PS.out_dir / PS.amber_min_in, "w") as AMBER_min_in_f:
                AMBER_min_in_f.write(
                    dedent(f"""\
                    {app_settings.run_name} minimization, implicit solvent
                     &cntrl
                      imin = 1,
                      maxcyc = {AS.min_step},
                      ncyc = 50,
                      ntmin = 1,
                      drms = 0.01,
                      cut = 1000.0,
                      ntb = 0,
                      saltcon = 0.1,
                      igb = 1,
                     &end
                           """)
                )
                amber_logger.trace(
                    "Wrote file at {path}",
                    path=PS.out_dir / PS.amber_min_in,
                )

            with open(PS.out_dir / PS.amber_sim_in, "w") as AMBER_sim_in_f:
                AMBER_sim_in_f.write(
                    dedent(f"""\
                    {app_settings.run_name} targeted Langevin dynamics simulation
                     &cntrl
                      imin = 0,
                      irest = 0,
                      ntx = 1,
                      ntt = 3,
                      gamma_ln = 5.0,
                      ig = -1,
                      tempi = {AS.temp},
                      temp0 = {AS.temp},
                      nstlim = {AS.sim_step},
                      dt = 0.002,
                      ntc = 2,
                      ntf = 2,
                      ntwr = 1,
                      ntpr = 1,
                      ntwx = 1,
                      ntwe = 1,
                      igb = 1,
                      saltcon = 0.1,
                      ntb = 0,
                      cut = 1000.0,
                     &end
                           """)
                )
                amber_logger.trace(
                    "Wrote file at {path}",
                    path=PS.out_dir / PS.amber_sim_in,
                )

            with open(
                PS.out_dir / PS.amber_tleap_init_in, "w"
            ) as AMBER_tleap_initial_f:
                AMBER_tleap_initial_f.write(
                    dedent(f"""\
                    source {AS.forcefield}
                    x=loadpdb {PS.out_dir / PS.sanitized_init_pdb_path}
                    saveamberparm x {PS.out_dir / PS.amber_pdb_init_top} {PS.out_dir / PS.amber_pdb_init_coord}
                    y=loadpdb {PS.out_dir / PS.sanitized_target_pdb_path}
                    saveamberparm y {PS.out_dir / PS.amber_pdb_target_top} {PS.out_dir / PS.amber_pdb_target_coord}
                    quit
                           """)
                )
                amber_logger.trace(
                    "Wrote file at {path}",
                    path=PS.out_dir / PS.amber_tleap_init_in,
                )

            cmd_tleap = f"tleap -f {PS.out_dir / PS.amber_tleap_init_in}"
            amber_logger.info("Running tleap")
            amber_logger.debug("Running {cmd}", cmd=AS.cmd_prefix + cmd_tleap)
            subprocess.run(
                AS.cmd_prefix + cmd_tleap,
                **app_settings.subprocess_settings.__dict__,
            )

            cmd_amber_initial = dedent(f"""\
                                    $AMBERHOME/bin/pmemd.cuda -O                        \\
                                        -i {PS.out_dir / PS.amber_min_in}               \\
                                        -p {PS.out_dir / PS.amber_pdb_init_top}         \\
                                        -c {PS.out_dir / PS.amber_pdb_init_coord}       \\
                                        -o {PS.out_dir / PS.amber_pdb_init_min_out}     \\
                                        -x {PS.out_dir / PS.amber_pdb_init_min_coord}   \\
                                        -r {PS.out_dir / PS.amber_pdb_init_min_rst}     \\
                                        </dev/null""")
            cmd_amber_target = dedent(f"""\
                                    $AMBERHOME/bin/pmemd.cuda -O                        \\
                                        -i {PS.out_dir / PS.amber_min_in}               \\
                                        -p {PS.out_dir / PS.amber_pdb_target_top}       \\
                                        -c {PS.out_dir / PS.amber_pdb_target_coord}     \\
                                        -o {PS.out_dir / PS.amber_pdb_target_min_out}   \\
                                        -x {PS.out_dir / PS.amber_pdb_target_min_coord} \\
                                        -r {PS.out_dir / PS.amber_pdb_target_min_rst}   \\
                                        </dev/null""")
            amber_logger.info("Running pmemd min for the initial structure")
            amber_logger.debug(
                "Running {cmd}", cmd=AS.cmd_prefix + cmd_amber_initial
            )
            subprocess.run(
                AS.cmd_prefix + cmd_amber_initial,
                **app_settings.subprocess_settings.__dict__,
            )

            amber_logger.info("Running pmemd min for the target structure")
            amber_logger.debug(
                "Running {cmd}", cmd=AS.cmd_prefix + cmd_amber_target
            )
            subprocess.run(
                AS.cmd_prefix + cmd_amber_target,
                **app_settings.subprocess_settings.__dict__,
            )

            # create initial_min.rst (rewrite to be able to read by ambmsk,
            # version problem)
            with open(
                PS.out_dir / PS.amber_ptraj_rewrite_init_in,
                "w",
            ) as amber_ptraj_rewrite_initial_f:
                amber_ptraj_rewrite_initial_f.write(
                    dedent(f"""\
                            trajin {PS.out_dir / PS.amber_pdb_init_min_rst}
                            trajout {PS.out_dir / PS.amber_pdb_rewrite_init_min_rst} restart
                           """)
                )
                amber_logger.trace(
                    "Wrote file at {path}",
                    path=PS.out_dir / PS.amber_ptraj_rewrite_init_in,
                )

            cmd_rewrite = dedent(f"""cpptraj                                                \\
                                        {PS.out_dir / PS.amber_pdb_init_top}    \\
                                        {PS.out_dir / PS.amber_ptraj_rewrite_init_in}""")
            amber_logger.info("Running cpptraj")
            amber_logger.debug("Ran {cmd}", cmd=AS.cmd_prefix + cmd_rewrite)
            subprocess.run(
                AS.cmd_prefix + cmd_rewrite,
                **app_settings.subprocess_settings.__dict__,
            )

            # Create AMBER_target_min_algn.rst
            with open(
                PS.out_dir / PS.amber_ptraj_align_target2initial_in,
                "w",
            ) as amber_ptraj_align_target2initial_f:
                amber_ptraj_align_target2initial_f.write(
                    dedent(f"""\
                            parm {PS.out_dir / PS.amber_pdb_init_top} [initial-top]
                            parm {PS.out_dir / PS.amber_pdb_target_top} [target-top]
                            trajin {PS.out_dir / PS.amber_pdb_target_min_rst} parm [target-top]
                            reference {PS.out_dir / PS.amber_pdb_init_min_rst} parm [initial-top] [initial-ref]
                            rms ref [initial-ref]  :1-{resnum}@CA out {PS.out_dir / PS.amber_rms_target_align_dat} 
                            trajout {PS.out_dir / PS.amber_target_min_algn} restart parm [target-top]
                           """)
                )
                amber_logger.trace(
                    "Wrote file at {path}",
                    path=PS.out_dir / PS.amber_ptraj_align_target2initial_in,
                )

            cmd_align = dedent(f"""cpptraj                                  \\
                                    {PS.out_dir / PS.amber_pdb_target_top}  \\
                                    {PS.out_dir / PS.amber_ptraj_align_target2initial_in}""")
            amber_logger.info("Running cpptraj")
            amber_logger.debug("Running {cmd}", cmd=AS.cmd_prefix + cmd_align)
            subprocess.run(
                AS.cmd_prefix + cmd_align,
                **app_settings.subprocess_settings.__dict__,
            )

            # create initial all-atom and alpha C pdbs
            cmd_AA_init = dedent(f"""\
                    ambmask -p {PS.out_dir / PS.amber_pdb_init_top} \\
                        -c {PS.out_dir / PS.amber_pdb_rewrite_init_min_rst}     \\
                        -prnlev 1 -out pdb""")
            amber_logger.info("Running ambmask (initial AA)")
            amber_logger.debug("Running {cmd}", cmd=AS.cmd_prefix + cmd_AA_init)
            with open(PS.out_dir / PS.amber_pdb_initial_min_pdb, "w") as out_f:
                subprocess.run(
                    AS.cmd_prefix + cmd_AA_init,
                    stdout=out_f,
                    **app_settings.subprocess_settings.__dict__,
                )

            cmd_CA_init = cmd_AA_init + " -find @CA"
            amber_logger.info("Running ambmask (initial CA)")
            amber_logger.debug("Running {cmd}", cmd=AS.cmd_prefix + cmd_CA_init)
            with open(
                PS.out_dir / PS.amber_pdb_initial_min_c_pdb, "w"
            ) as out_f:
                subprocess.run(
                    AS.cmd_prefix + cmd_CA_init,
                    stdout=out_f,
                    **app_settings.subprocess_settings.__dict__,
                )

            cmd_AA_target = dedent(f"""\
                    ambmask -p {PS.out_dir / PS.amber_pdb_target_top}  \\
                        -c {PS.out_dir / PS.amber_target_min_algn} \\
                        -prnlev 1 -out pdb""")
            amber_logger.info("Running ambmask (target AA)")
            amber_logger.debug(
                "Running {cmd}", cmd=AS.cmd_prefix + cmd_AA_target
            )
            with open(PS.out_dir / PS.amber_pdb_target_min_pdb, "w") as out_f:
                subprocess.run(
                    AS.cmd_prefix + cmd_AA_target,
                    stdout=out_f,
                    **app_settings.subprocess_settings.__dict__,
                )

            cmd_CA_target = cmd_AA_target + " -find @CA"
            amber_logger.info("Running ambmask (target CA)")
            amber_logger.debug("Running {cmd}", cmd=AS.cmd_prefix + cmd_CA_target)
            with open(PS.out_dir / PS.amber_pdb_target_min_c_pdb, "w") as out_f:
                subprocess.run(
                    AS.cmd_prefix + cmd_CA_target,
                    stdout=out_f,
                    **app_settings.subprocess_settings.__dict__,
                )

        if app_settings.mode_selection == "MATLAB":
            aa_step = aa_init
        else:
            raise NotImplementedError()

        run_step(
            aa_step,
            aa_target,
            step,
            step_logger,
            amber_logger,
            app_settings,
        )


if __name__ == "__main__":
    typer.run(main)
