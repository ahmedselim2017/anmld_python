from pathlib import Path
from textwrap import dedent
import importlib.metadata
from loguru import logger
import subprocess
import tomllib

from tqdm import tqdm
import typer

from anmld_python.settings import AppSettings


@logger.catch(reraise=True)
def main(settings_path: Path):
    with open(settings_path, "rb") as settings_f:
        AS = AppSettings(**tomllib.load(settings_f))

    logger.remove()
    logger_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>STEP {extra['step']}</level> | "
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

    PS = AS.path_settings
    AS.structure_init = AS.structure_init.absolute()
    AS.structure_target = AS.structure_target.absolute()
    AS.out_dir = AS.out_dir.absolute()

    AS.out_dir.mkdir(parents=True)
    step_logger.trace(f"Created output directory at {AS.out_dir}")

    step_logger.add(AS.out_dir / "anmld.log", serialize=True)

    cmd_load = f"module load {AS.amber_settings.module_name} && "

    for step in tqdm(range(AS.anmld_settings.n_steps), desc="Running ANM-LD"):
        step_logger = logger.bind(step=step)
        amber_logger = step_logger.bind(isAMBER=True)

        step_logger.info("Starting step")

        if step == 0:
            with open(AS.out_dir / PS.amber_min_in, "w") as AMBER_min_in_f:
                AMBER_min_in_f.write(
                    dedent(f"""\
                    {AS.run_name} minimization, implicit solvent
                     &cntrl
                      imin = 1,
                      maxcyc = {AS.amber_settings.min_step},
                      ncyc = 50,
                      ntmin = 1,
                      drms = 0.01,
                      cut = 1000.0,
                      ntb = 0,
                      saltcon = 0.1,
                      igb = 1,
                     &end""")
                )
                amber_logger.trace(
                    "Wrote file at {path}", path=AS.out_dir / PS.amber_min_in
                )

            with open(AS.out_dir / PS.amber_sim_in, "w") as AMBER_sim_in_f:
                AMBER_sim_in_f.write(
                    dedent(f"""\
                    {AS.run_name} targeted Langevin dynamics simulation
                     &cntrl
                      imin = 0,
                      irest = 0,
                      ntx = 1,
                      ntt = 3,
                      gamma_ln = 5.0,
                      ig = -1,
                      tempi = {AS.amber_settings.temp},
                      temp0 = {AS.amber_settings.temp},
                      nstlim = {AS.amber_settings.sim_step},
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
                     &end""")
                )
                amber_logger.trace(
                    "Wrote file at {path}", path=AS.out_dir / PS.amber_sim_in
                )

            with open(
                AS.out_dir / PS.amber_tleap_init_in, "w"
            ) as AMBER_tleap_initial_f:
                AMBER_tleap_initial_f.write(
                    dedent(f"""\
                    source {AS.amber_settings.forcefield}
                    x=loadpdb {AS.structure_init}
                    saveamberparm x {AS.out_dir / PS.amber_pdb_init_top} {AS.out_dir / PS.amber_pdb_init_coord}
                    y=loadpdb {AS.structure_target}
                    saveamberparm y {AS.out_dir / PS.amber_pdb_target_top} {AS.out_dir / PS.amber_pdb_target_coord}
                    quit""")
                )
                amber_logger.trace(
                    "Wrote file at {path}",
                    path=AS.out_dir / PS.amber_tleap_init_in,
                )

            cmd_tleap = f"tleap -f {AS.out_dir / PS.amber_tleap_init_in}"
            subprocess.run(
                cmd_load + cmd_tleap,
                shell=True,
                cwd=AS.out_dir,
                check=True,
            )
            amber_logger.debug("Ran {cmd}", cmd=cmd_load + cmd_tleap)

            cmd_amber_initial = dedent(f"""\
                                    $AMBERHOME/bin/pmemd.cuda -O                                    \\
                                        -i {AS.out_dir / PS.amber_min_in}               \\
                                        -p {AS.out_dir / PS.amber_pdb_init_top}         \\
                                        -c {AS.out_dir / PS.amber_pdb_init_coord}       \\
                                        -o {AS.out_dir / PS.amber_pdb_init_min_out}     \\
                                        -x {AS.out_dir / PS.amber_pdb_init_min_coord}   \\
                                        -r {AS.out_dir / PS.amber_pdb_init_min_rst}     \\
                                        </dev/null""")
            cmd_amber_target = dedent(f"""\
                                    $AMBERHOME/bin/pmemd.cuda -O                                    \\
                                        -i {AS.out_dir / PS.amber_min_in}               \\
                                        -p {AS.out_dir / PS.amber_pdb_target_top}       \\
                                        -c {AS.out_dir / PS.amber_pdb_target_coord}     \\
                                        -o {AS.out_dir / PS.amber_pdb_target_min_out}   \\
                                        -x {AS.out_dir / PS.amber_pdb_target_min_coord} \\
                                        -r {AS.out_dir / PS.amber_pdb_target_min_rst}   \\
                                        </dev/null""")
            subprocess.run(
                cmd_load + cmd_amber_initial,
                shell=True,
                cwd=AS.out_dir,
                check=True,
            )
            amber_logger.debug("Ran {cmd}", cmd=cmd_load + cmd_amber_initial)
            subprocess.run(
                cmd_load + cmd_amber_target,
                shell=True,
                cwd=AS.out_dir,
                check=True,
            )
            amber_logger.debug("Ran {cmd}", cmd=cmd_load + cmd_amber_target)

            # create initial_min.rst (rewrite to be able to read by ambmsk,
            # version problem)
            with open(
                AS.out_dir / PS.amber_ptraj_rewrite_init_in,
                "w",
            ) as amber_ptraj_rewrite_initial_f:
                amber_ptraj_rewrite_initial_f.write(
                    dedent(f"""\
                            trajin {AS.out_dir / PS.amber_pdb_init_min_rst}
                            trajout {AS.out_dir / PS.amber_pdb_rewrite_init_min_rst} restart""")
                )
                amber_logger.trace(
                    "Wrote file at {path}",
                    path=AS.out_dir / PS.amber_ptraj_rewrite_init_in,
                )

            cmd_rewrite = dedent(f"""cpptraj                                                \\
                                        {AS.out_dir / PS.amber_pdb_init_top}    \\
                                        {AS.out_dir / PS.amber_ptraj_rewrite_init_in}""")
            subprocess.run(
                cmd_load + cmd_rewrite,
                shell=True,
                cwd=AS.out_dir,
                check=True,
            )
            amber_logger.debug("Ran {cmd}", cmd=cmd_load + cmd_rewrite)

            # Create AMBER_target_min_algn.rst
            with open(
                AS.out_dir / PS.amber_ptraj_align_target2initial_in,
                "w",
            ) as amber_ptraj_align_target2initial_f:
                amber_ptraj_align_target2initial_f.write(
                    dedent(f"""\
                            parm {AS.out_dir / PS.amber_pdb_init_top} [initial-top]
                            parm {AS.out_dir / PS.amber_pdb_target_top} [target-top]
                            trajin {AS.out_dir / PS.amber_pdb_target_min_rst} parm [target-top]
                            reference {AS.out_dir / PS.amber_pdb_init_min_rst} parm [initial-top] [initial-ref]
                            rms ref [initial-ref]  :1-{RESNUM}@CA out {AS.out_dir / PS.amber_rms_target_align_dat} 
                            trajout {AS.out_dir / PS.amber_target_min_algn} restart parm [target-top]""")  # TODO: RESNUM
                )
                amber_logger.trace(
                    "Wrote file at {path}",
                    path=AS.out_dir / PS.amber_ptraj_align_target2initial_in,
                )

            cmd_align = dedent(f"""cpptraj                                  \\
                                    {AS.out_dir / PS.amber_pdb_target_top}  \\
                                    {AS.out_dir / PS.amber_ptraj_align_target2initial_in}""")
            subprocess.run(
                cmd_load + cmd_align,
                shell=True,
                cwd=AS.out_dir,
                check=True,
            )
            amber_logger.debug("Ran {cmd}", cmd=cmd_load + cmd_align)

            # create initial all-atom and alpha C pdbs
            cmd_AA_init = dedent(f"""\
                    ambmask -p {AS.out_dir / PS.amber_pdb_init_top} \\
                        -c {AS.out_dir / PS.amber_pdb_rewrite_init_min_rst}     \\
                        -prnlev 1 -out pdb""")
            with open(AS.out_dir / PS.amber_pdb_initial_min_pdb, "w") as out_f:
                subprocess.run(
                    cmd_load + cmd_AA_init,
                    shell=True,
                    cwd=AS.out_dir,
                    stdout=out_f,
                    check=True,
                )
            amber_logger.debug("Ran {cmd}", cmd=cmd_load + cmd_AA_init)

            cmd_CA_init = cmd_AA_init + " -find @CA"
            with open(
                AS.out_dir / PS.amber_pdb_initial_min_c_pdb, "w"
            ) as out_f:
                subprocess.run(
                    cmd_load + cmd_CA_init,
                    shell=True,
                    cwd=AS.out_dir,
                    stdout=out_f,
                    check=True,
                )
            amber_logger.debug("Ran {cmd}", cmd=cmd_load + cmd_CA_init)

            cmd_AA_target = dedent(f"""\
                    ambmask -p {AS.out_dir / PS.amber_pdb_target_top}  \\
                        -c {AS.out_dir / PS.amber_target_min_algn} \\
                        -prnlev 1 -out pdb""")
            with open(AS.out_dir / PS.amber_pdb_target_min_pdb, "w") as out_f:
                subprocess.run(
                    cmd_load + cmd_AA_target,
                    shell=True,
                    cwd=AS.out_dir,
                    stdout=out_f,
                    check=True,
                )
            amber_logger.debug("Ran {cmd}", cmd=cmd_load + cmd_AA_target)

            cmd_CA_target = cmd_AA_target + " -find @CA"
            with open(AS.out_dir / PS.amber_pdb_target_min_c_pdb, "w") as out_f:
                subprocess.run(
                    cmd_load + cmd_CA_target,
                    shell=True,
                    cwd=AS.out_dir,
                    stdout=out_f,
                    check=True,
                )
            amber_logger.debug("Ran {cmd}", cmd=cmd_load + cmd_CA_target)


if __name__ == "__main__":
    typer.run(main)
