from pathlib import Path
from textwrap import dedent
import importlib.metadata
import logging
import subprocess
import tomllib

from tqdm import tqdm
import typer

from anmld_python.settings import AppSettings


def main(settings_path: Path):
    with open(settings_path, "rb") as settings_f:
        app_settings = AppSettings(**tomllib.load(settings_f))

    logging.basicConfig(
        level=app_settings.logging_level,
        format="%(asctime)s | %s(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logging.info("Starting ANM-LD python.")
    logging.info(f"Version: {importlib.metadata.version('anmld_python')}")

    app_settings.structure_init = app_settings.structure_init.absolute()
    app_settings.structure_target = app_settings.structure_target.absolute()
    app_settings.out_dir = app_settings.out_dir.absolute()

    logging.debug(f"structure_init: {app_settings.structure_init}")
    logging.debug(f"structure_target: {app_settings.structure_target}")
    logging.debug(f"out_dir: {app_settings.out_dir}")

    app_settings.out_dir.mkdir(parents=True)
    logging.debug(f"Created output directory at {app_settings.out_dir}")

    cmd_load = f"module load {app_settings.amber_settings.module_name} && "

    for cycle in tqdm(range(app_settings.anmld_settings.n_cycle)):
        logging.info(f"Starting cycle: {cycle}")

        if cycle == 0:
            with open(app_settings.out_dir / "AMBER_min.in", "w") as AMBER_min_in_f:
                AMBER_min_in_f.write(
                    dedent(f"""\
                    {app_settings.run_name} minimization, implicit solvent
                     &cntrl
                      imin = 1,
                      maxcyc = {app_settings.amber_settings.min_step},
                      ncyc = 50,
                      ntmin = 1,
                      drms = 0.01,
                      cut = 1000.0,
                      ntb = 0,
                      saltcon = 0.1,
                      igb = 1,
                     &end""")
                )
            logging.debug(f"Wrote {app_settings.out_dir / 'AMBER_min.in'}")

            with open(app_settings.out_dir / "AMBER_sim.in", "w") as AMBER_sim_in_f:
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
                      tempi = {app_settings.amber_settings.temp},
                      temp0 = {app_settings.amber_settings.temp},
                      nstlim = {app_settings.amber_settings.sim_step},
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

            logging.debug(f"Wrote {app_settings.out_dir / 'AMBER_sim.in'}")

            path_topo_init = app_settings.out_dir / "AMBER_pdb_initial.top"
            path_coord_init = app_settings.out_dir / "AMBER_pdb_initial.coord"

            path_topo_target = app_settings.out_dir / "AMBER_pdb_target.top"
            path_coord_target = app_settings.out_dir / "AMBER_pdb_target.coord"

            with open(
                app_settings.out_dir / "AMBER_tleap_initial.in", "w"
            ) as AMBER_tleap_initial_f:
                AMBER_tleap_initial_f.write(
                    dedent(f"""\
                    source {app_settings.amber_settings.forcefield}
                    x=loadpdb {app_settings.structure_init}
                    saveamberparm x {app_settings.out_dir / "AMBER_pdb_initial.top"} {app_settings.out_dir / "AMBER_pdb_initial.coord"}
                    y=loadpdb {app_settings.structure_target}
                    saveamberparm y {app_settings.out_dir / "AMBER_pdb_target.top"} {app_settings.out_dir / "AMBER_pdb_target.coord"}
                    quit""")
                )
            logging.info(f"Wrote {app_settings.out_dir / 'AMBER_tleap_initial.in'}")

            cmd_tleap = f"tleap -f {app_settings.out_dir / 'AMBER_tleap_initial.in'}"
            subprocess.run(
                cmd_load + cmd_tleap,
                shell=True,
                cwd=app_settings.out_dir,
                check=True,
            )
            logging.debug("Ran tleap")

            cmd_amber_initial = dedent(f"""\
                                    $AMBERHOME/bin/pmemd.cuda -O                                    \\
                                        -i {app_settings.out_dir / "AMBER_min.in"}                  \\
                                        -p {app_settings.out_dir / "AMBER_pdb_initial.top"}         \\
                                        -c {app_settings.out_dir / "AMBER_pdb_initial.coord"}       \\
                                        -o {app_settings.out_dir / "AMBER_pdb_initial_min.out"}     \\
                                        -x {app_settings.out_dir / "AMBER_pdb_initial_min.coord"}   \\
                                        -r {app_settings.out_dir / "AMBER_pdb_initial_min.rst"}     \\
                                        </dev/null""")
            cmd_amber_target = dedent(f"""\
                                    $AMBERHOME/bin/pmemd.cuda -O                                    \\
                                        -i {app_settings.out_dir / "AMBER_min.in"}                  \\
                                        -p {app_settings.out_dir / "AMBER_pdb_target.top"}          \\
                                        -c {app_settings.out_dir / "AMBER_pdb_target.coord"}        \\
                                        -o {app_settings.out_dir / "AMBER_pdb_target_min.out"}      \\
                                        -x {app_settings.out_dir / "AMBER_pdb_target_min.coord"}    \\
                                        -r {app_settings.out_dir / "AMBER_pdb_target_min.rst"}      \\
                                        </dev/null""")
            subprocess.run(
                cmd_load + cmd_amber_initial,
                shell=True,
                cwd=app_settings.out_dir,
                check=True,
            )
            logging.debug("Ran AMBER initial")
            subprocess.run(
                cmd_load + cmd_amber_target,
                shell=True,
                cwd=app_settings.out_dir,
                check=True,
            )
            logging.debug("Ran AMBER target")

            # create initial_min.rst (rewrite to be able to read by ambmsk,
            # version problem)
            with open(
                app_settings.out_dir / "AMBER_ptraj_rewrite_initial.in", "w"
            ) as amber_ptraj_rewrite_initial_f:
                amber_ptraj_rewrite_initial_f.write(
                    dedent(f"""\
                            trajin {app_settings.out_dir / "AMBER_pdb_initial_min.rst"}
                            trajout {app_settings.out_dir / "AMBER_initial_min.rst"} restart""")
                )

            cmd_rewrite = dedent(f"""cpptraj                                                \\
                                        {app_settings.out_dir / "AMBER_pdb_initial.top"}    \\
                                        {app_settings.out_dir / "AMBER_ptraj_rewrite_initial.in"}""")
            subprocess.run(
                cmd_load + cmd_rewrite,
                shell=True,
                cwd=app_settings.out_dir,
                check=True,
            )
            logging.debug("Ran rewrite")

            # Create AMBER_target_min_algn.rst
            with open(
                app_settings.out_dir / "AMBER_ptraj_align_target2initial.in", "w"
            ) as amber_ptraj_align_target2initial_f:
                amber_ptraj_align_target2initial_f.write(
                    dedent(f"""\
                            parm {app_settings.out_dir / "AMBER_pdb_initial.top"} [initial-top]
                            parm {app_settings.out_dir / "AMBER_pdb_target.top"} [target-top]
                            trajin {app_settings.out_dir / "AMBER_pdb_target_min.rst"} parm [target-top]
                            reference {app_settings.out_dir / "AMBER_pdb_initial_min.rst"} parm [initial-top] [initial-ref]
                            rms ref [initial-ref]  :1-{RESNUM}@CA out {app_settings.out_dir / "AMBER_rms_target_align.dat"} 
                            trajout {app_settings.out_dir / "AMBER_target_min_algn.rst"} restart parm [target-top]""")  # TODO: RESNUM
                )

            cmd_align = dedent(f"""cpptraj                                          \\
                                    {app_settings.out_dir / "AMBER_pdb_target.top"} \\
                                    {app_settings.out_dir / "AMBER_ptraj_align_target2initial.in"}""")
            subprocess.run(
                cmd_load + cmd_align,
                shell=True,
                cwd=app_settings.out_dir,
                check=True,
            )
            logging.debug("Ran alignment")

            # create initial all-atom and alpha C pdbs
            cmd_AA_init = dedent(f"""\
                    ambmask -p {app_settings.out_dir / "AMBER_pdb_initial.top"} \\
                        -c {app_settings.out_dir / "AMBER_initial_min.rst"}     \\
                        -prnlev 1 -out pdb""")
            with open(app_settings.out_dir / "AMBER_pdb_initial_min.pdb", "w") as out_f:
                subprocess.run(
                    cmd_load + cmd_AA_init,
                    shell=True,
                    cwd=app_settings.out_dir,
                    stdout=out_f,
                    check=True,
                )
            logging.debug("Ran ambmask for init AAs")

            cmd_CA_init = cmd_AA_init + " -find @CA"
            with open(
                app_settings.out_dir / "AMBER_pdb_initial_min_C.pdb", "w"
            ) as out_f:
                subprocess.run(
                    cmd_load + cmd_CA_init,
                    shell=True,
                    cwd=app_settings.out_dir,
                    stdout=out_f,
                    check=True,
                )
            logging.debug("Ran ambmask for init CAs")

            cmd_AA_target = dedent(f"""\
                    ambmask -p {app_settings.out_dir / "AMBER_pdb_target.top"}  \\
                        -c {app_settings.out_dir / "AMBER_target_min_algn.rst"} \\
                        -prnlev 1 -out pdb""")
            with open(app_settings.out_dir / "AMBER_pdb_target_min.pdb", "w") as out_f:
                subprocess.run(
                    cmd_load + cmd_AA_target,
                    shell=True,
                    cwd=app_settings.out_dir,
                    stdout=out_f,
                    check=True,
                )
            logging.debug("Ran ambmask for target AAs")

            cmd_CA_target = cmd_AA_target + " -find @CA"
            with open(
                app_settings.out_dir / "AMBER_pdb_target_min_C.pdb", "w"
            ) as out_f:
                subprocess.run(
                    cmd_load + cmd_CA_target,
                    shell=True,
                    cwd=app_settings.out_dir,
                    stdout=out_f,
                    check=True,
                )
            logging.debug("Ran ambmask for target AAs")


if __name__ == "__main__":
    typer.run(main)
