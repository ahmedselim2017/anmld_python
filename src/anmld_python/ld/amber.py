from __future__ import annotations
from pathlib import Path
from textwrap import dedent
import subprocess

from biotite.structure.io import load_structure
import loguru

from anmld_python.settings import AppSettings, StepPathSettings
from anmld_python.tools import LDError, calc_aa_ca_rmsd


class tleapError(Exception):
    pass


def run_pdb4amber(
    in_path: Path,
    out_path: Path,
    logger: loguru.Logger,
    app_settings: AppSettings,
):
    logger.trace("run_pdb4amber")

    cmd = (
        app_settings.amber_settings.pdb4amber_prefix
        + app_settings.amber_settings.pdb4amber_cmd
    )

    cmd += f" -i {in_path} -o {out_path} --reduce --dry"
    logger.info("Running pdb4amber")
    logger.debug(f"Running {app_settings.amber_settings.ambertools_prefix + cmd}")
    subprocess.run(
        app_settings.amber_settings.ambertools_prefix + cmd,
        **app_settings.subprocess_settings.__dict__,
    )


def check_pmemd_out(out_path: Path, logger: loguru.Logger):
    """
    As pmemd can fail with a successful exit code, its output must be checked
    additionally.
    """
    logger.trace("Checking pmemd output", out_path=out_path)
    with open(out_path, "r") as f:
        out_content = f.read()
    results = out_content.split("                    FINAL RESULTS")[-1]

    # TODO: A less hacky way to check failiure?
    keywords = ["*************", "FAILURE", "Indef", "NaN", "Infinity"]
    for keyword in keywords:
        if keyword in results:
            return False
    return True


def run_setup(
    path_abs_init: Path,
    path_abs_target: Path,
    resnum: int,
    ld_logger: loguru.Logger,
    app_settings: AppSettings,
):
    AS = app_settings.amber_settings
    PS = app_settings.path_settings

    with open(PS.out_dir / PS.amber_min_in, "w") as AMBER_min_in_f:
        AMBER_min_in_f.write(
            dedent(f"""\
            ANMLD minimization, implicit solvent
             &cntrl
              imin = {AS.min.imin},
              maxcyc = {AS.min_step},
              ncyc = {AS.min.ncyc},
              ntmin = {AS.min.ntmin},
              drms = {AS.min.drms},
              cut = {AS.min.cut},
              ntb = {AS.min.ntb},
              saltcon = {AS.min.saltcon},
              igb = {AS.min.igb},
             &end
                   """)
        )
        ld_logger.trace(
            "Wrote file at {path}",
            path=PS.out_dir / PS.amber_min_in,
        )

    with open(PS.out_dir / PS.amber_sim_in, "w") as AMBER_sim_in_f:
        AMBER_sim_in_f.write(
            dedent(f"""\
            ANMLD targeted Langevin dynamics simulation
             &cntrl
              imin = {AS.ld.imin},
              irest = {AS.ld.irest},
              ntx = {AS.ld.ntx},
              ntt = {AS.ld.ntt},
              gamma_ln = {AS.ld.gamma_ln},
              ig = {AS.ld.ig},
              tempi = {AS.ld_temp},
              temp0 = {AS.ld_temp},
              nstlim = {AS.ld_step},
              dt = {AS.ld.dt},
              ntc = {AS.ld.ntc},
              ntf = {AS.ld.ntf},
              ntwr = {AS.ld.ntwr},
              ntpr = {AS.ld.ntpr},
              ntwx = {AS.ld.ntwx},
              ntwe = {AS.ld.ntwe},
              igb = {AS.ld.igb},
              saltcon = {AS.ld.saltcon},
              ntb = {AS.ld.ntb},
              cut = {AS.ld.cut},
             &end
                   """)
        )
        ld_logger.trace(
            "Wrote file at {path}",
            path=PS.out_dir / PS.amber_sim_in,
        )

    with open(PS.out_dir / PS.amber_tleap_init_in, "w") as AMBER_tleap_initial_f:
        AMBER_tleap_initial_f.write(
            dedent(f"""\
            source {AS.forcefield}
            x=loadpdb "{path_abs_init}"
            saveamberparm x "{PS.out_dir / PS.amber_pdb_init_top}" "{PS.out_dir / PS.amber_pdb_init_coord}"
            y=loadpdb "{path_abs_target}"
            saveamberparm y "{PS.out_dir / PS.amber_pdb_target_top}" "{PS.out_dir / PS.amber_pdb_target_coord}"
            quit
                   """)
        )
        ld_logger.trace(
            "Wrote file at {path}",
            path=PS.out_dir / PS.amber_tleap_init_in,
        )

    cmd_tleap = f'tleap -f "{PS.out_dir / PS.amber_tleap_init_in}"'
    ld_logger.info("Running tleap")
    ld_logger.debug("Running {cmd}", cmd=AS.ambertools_prefix + cmd_tleap)
    subprocess.run(
        AS.ambertools_prefix + cmd_tleap,
        **app_settings.subprocess_settings.__dict__,
    )

    if (
        not (PS.out_dir / PS.amber_pdb_init_top).is_file()
        or not (PS.out_dir / PS.amber_pdb_init_coord).is_file()
        or not (PS.out_dir / PS.amber_pdb_target_top).is_file()
        or not (PS.out_dir / PS.amber_pdb_target_coord).is_file()
    ):
        emsg = (
            f"tleap command {cmd_tleap} failed to create output files."
            f" {PS.amber_pdb_init_top}, {PS.amber_pdb_init_coord}"
            f" {PS.amber_pdb_target_top}, {PS.amber_pdb_target_coord}"
        )
        ld_logger.error(emsg)
        raise tleapError().with_traceback(None) from None

    cmd_amber_initial = dedent(f"""\
                            $AMBERHOME/bin/{AS.pmemd_cmd} -O                        \\
                                -i "{PS.out_dir / PS.amber_min_in}"             \\
                                -p "{PS.out_dir / PS.amber_pdb_init_top}"       \\
                                -c "{PS.out_dir / PS.amber_pdb_init_coord}"     \\
                                -o "{PS.out_dir / PS.amber_pdb_init_min_out}"   \\
                                -x "{PS.out_dir / PS.amber_pdb_init_min_coord}" \\
                                -r "{PS.out_dir / PS.amber_pdb_init_min_rst}"   \\
                                </dev/null""")
    cmd_amber_target = dedent(f"""\
                            $AMBERHOME/bin/{AS.pmemd_cmd} -O                            \\
                                -i "{PS.out_dir / PS.amber_min_in}"                 \\
                                -p "{PS.out_dir / PS.amber_pdb_target_top}"         \\
                                -c "{PS.out_dir / PS.amber_pdb_target_coord}"       \\
                                -o "{PS.out_dir / PS.amber_pdb_target_min_out}"     \\
                                -x "{PS.out_dir / PS.amber_pdb_target_min_coord}"   \\
                                -r "{PS.out_dir / PS.amber_pdb_target_min_rst}"     \\
                                </dev/null""")
    ld_logger.info("Running pmemd min for the initial structure")
    ld_logger.debug("Running {cmd}", cmd=AS.pmemd_prefix + cmd_amber_initial)
    subprocess.run(
        AS.pmemd_prefix + cmd_amber_initial,
        **app_settings.subprocess_settings.__dict__,
    )
    if not check_pmemd_out(PS.out_dir / PS.amber_pdb_init_min_out, ld_logger):
        emsg = "The pmemd minimization of the initial structure failed."
        ld_logger.error(emsg)
        raise LDError(emsg) from None

    ld_logger.info("Running pmemd min for the target structure")
    ld_logger.debug("Running {cmd}", cmd=AS.pmemd_prefix + cmd_amber_target)
    subprocess.run(
        AS.pmemd_prefix + cmd_amber_target,
        **app_settings.subprocess_settings.__dict__,
    )
    if not check_pmemd_out(PS.out_dir / PS.amber_pdb_target_min_out, ld_logger):
        emsg = "The pmemd minimization of the target structure failed."
        ld_logger.error(emsg)
        raise LDError(emsg) from None

    # create initial_min.rst (rewrite to be able to read by ambmsk,
    # version problem)
    with open(
        PS.out_dir / PS.amber_ptraj_rewrite_init_in,
        "w",
    ) as amber_ptraj_rewrite_initial_f:
        amber_ptraj_rewrite_initial_f.write(
            dedent(f"""\
                    trajin "{PS.out_dir / PS.amber_pdb_init_min_rst}"
                    trajout "{PS.out_dir / PS.amber_pdb_rewrite_init_min_rst}" restart
                   """)
        )
        ld_logger.trace(
            "Wrote file at {path}",
            path=PS.out_dir / PS.amber_ptraj_rewrite_init_in,
        )

    cmd_rewrite = dedent(f"""cpptraj                                    \\
                                "{PS.out_dir / PS.amber_pdb_init_top}"  \\
                                "{PS.out_dir / PS.amber_ptraj_rewrite_init_in}\"""")
    ld_logger.info("Running cpptraj")
    ld_logger.debug("Ran {cmd}", cmd=AS.ambertools_prefix + cmd_rewrite)
    subprocess.run(
        AS.ambertools_prefix + cmd_rewrite,
        **app_settings.subprocess_settings.__dict__,
    )

    # Create AMBER_target_min_algn.rst
    with open(
        PS.out_dir / PS.amber_ptraj_align_target2initial_in,
        "w",
    ) as amber_ptraj_align_target2initial_f:
        amber_ptraj_align_target2initial_f.write(
            dedent(f"""\
                    parm "{PS.out_dir / PS.amber_pdb_init_top}" [initial-top]
                    parm "{PS.out_dir / PS.amber_pdb_target_top}" [target-top]
                    trajin "{PS.out_dir / PS.amber_pdb_target_min_rst}" parm [target-top]
                    reference "{PS.out_dir / PS.amber_pdb_init_min_rst}" parm [initial-top] [initial-ref]
                    rms ref [initial-ref]  :1-{resnum}@CA out "{PS.out_dir / PS.amber_rms_target_align_dat}"
                    trajout "{PS.out_dir / PS.amber_target_min_algn}" restart parm [target-top]
                   """)
        )
        ld_logger.trace(
            "Wrote file at {path}",
            path=PS.out_dir / PS.amber_ptraj_align_target2initial_in,
        )

    cmd_align = dedent(f"""cpptraj                                      \\
                            "{PS.out_dir / PS.amber_pdb_target_top}"    \\
                            "{PS.out_dir / PS.amber_ptraj_align_target2initial_in}\"""")
    ld_logger.info("Running cpptraj")
    ld_logger.debug("Running {cmd}", cmd=AS.ambertools_prefix + cmd_align)
    subprocess.run(
        AS.ambertools_prefix + cmd_align,
        **app_settings.subprocess_settings.__dict__,
    )

    # create initial all-atom and alpha C pdbs
    cmd_AA_init = dedent(f"""\
            ambmask -p "{PS.amber_pdb_init_top}"            \\
                -c "{PS.amber_pdb_rewrite_init_min_rst}"    \\
                -prnlev 1 -out pdb""")
    ld_logger.info("Running ambmask (initial AA)")
    ld_logger.debug("Running {cmd}", cmd=AS.ambertools_prefix + cmd_AA_init)
    with open(PS.out_dir / PS.amber_pdb_initial_min_pdb, "w") as out_f:
        subprocess.run(
            AS.ambertools_prefix + cmd_AA_init,
            stdout=out_f,
            **app_settings.subprocess_settings.__dict__,
        )

    cmd_CA_init = cmd_AA_init + " -find @CA"
    ld_logger.info("Running ambmask (initial CA)")
    ld_logger.debug("Running {cmd}", cmd=AS.ambertools_prefix + cmd_CA_init)
    with open(PS.out_dir / PS.amber_pdb_initial_min_c_pdb, "w") as out_f:
        subprocess.run(
            AS.ambertools_prefix + cmd_CA_init,
            stdout=out_f,
            **app_settings.subprocess_settings.__dict__,
        )

    cmd_AA_target = dedent(f"""\
            ambmask -p "{PS.amber_pdb_target_top}"  \\
                -c "{PS.amber_target_min_algn}"     \\
                -prnlev 1 -out pdb""")
    ld_logger.info("Running ambmask (target AA)")
    ld_logger.debug("Running {cmd}", cmd=AS.ambertools_prefix + cmd_AA_target)
    with open(PS.out_dir / PS.amber_pdb_target_min_pdb, "w") as out_f:
        subprocess.run(
            AS.ambertools_prefix + cmd_AA_target,
            stdout=out_f,
            **app_settings.subprocess_settings.__dict__,
        )

    cmd_CA_target = cmd_AA_target + " -find @CA"
    ld_logger.info("Running ambmask (target CA)")
    ld_logger.debug("Running {cmd}", cmd=AS.ambertools_prefix + cmd_CA_target)
    with open(PS.out_dir / PS.amber_pdb_target_min_c_pdb, "w") as out_f:
        subprocess.run(
            AS.ambertools_prefix + cmd_CA_target,
            stdout=out_f,
            **app_settings.subprocess_settings.__dict__,
        )


def run_ld_step(
    pred_abs_path: Path,
    resnum: int,
    ld_logger: loguru.Logger,
    app_settings: AppSettings,
    SP: StepPathSettings,
) -> dict:
    AS = app_settings.amber_settings
    PS = app_settings.path_settings

    amber_tleap_step_in_path = PS.out_dir / SP.step_amber_tleap_anm_pdb

    with open(amber_tleap_step_in_path, "w") as amber_tleap_step_in_f:
        amber_tleap_step_in_f.write(
            dedent(f"""\
                    source {AS.forcefield}
                    x=loadpdb "{pred_abs_path}"
                    saveamberparm x "{PS.out_dir / SP.step_amber_top}" "{PS.out_dir / SP.step_amber_coord}"
                    quit
                       """)
        )
    ld_logger.trace("Wrote file at {path}", path=amber_tleap_step_in_path)

    cmd_tleap = f'tleap -f "{amber_tleap_step_in_path}"'
    ld_logger.info("Running tleap")
    ld_logger.debug("Running {cmd}", cmd=AS.ambertools_prefix + cmd_tleap)
    subprocess.run(
        AS.ambertools_prefix + cmd_tleap,
        **app_settings.subprocess_settings.__dict__,
    )

    if (
        not (PS.out_dir / SP.step_amber_top).is_file()
        or not (PS.out_dir / SP.step_amber_coord).is_file()
    ):
        emsg = (
            f"tleap command {cmd_tleap} failed to create output files."
            f" {SP.step_amber_top} and {SP.step_amber_coord}"
        )
        ld_logger.error(emsg)
        raise tleapError().with_traceback(None) from None

    cmd_min = dedent(f"""\
                    $AMBERHOME/bin/{AS.pmemd_cmd} -O                    \\
                        -i "{PS.out_dir / PS.amber_min_in}"         \\
                        -p "{PS.out_dir / SP.step_amber_top}"       \\
                        -c "{PS.out_dir / SP.step_amber_coord}"     \\
                        -o "{PS.out_dir / SP.step_amber_min_out}"   \\
                        -x "{PS.out_dir / SP.step_amber_min_coord}" \\
                        -r "{PS.out_dir / SP.step_amber_min_rst}"   \\
                        </dev/null""")
    ld_logger.info("Running pmemd min")
    ld_logger.debug("Running {cmd}", cmd=AS.pmemd_prefix + cmd_min)
    subprocess.run(
        AS.pmemd_prefix + cmd_min,
        **app_settings.subprocess_settings.__dict__,
    )
    if not check_pmemd_out(PS.out_dir / SP.step_amber_min_out, ld_logger):
        emsg = "The pmemd minimization of the structure failed."
        ld_logger.error(emsg)
        raise LDError(emsg) from None

    cmd_sim = dedent(f"""\
                    $AMBERHOME/bin/{AS.pmemd_cmd} -O                        \\
                        -i "{PS.out_dir / PS.amber_sim_in}"             \\
                        -p "{PS.out_dir / SP.step_amber_top}"           \\
                        -c "{PS.out_dir / SP.step_amber_min_rst}"       \\
                        -o "{PS.out_dir / SP.step_amber_sim_out}"       \\
                        -x "{PS.out_dir / SP.step_amber_sim_coord}"     \\
                        -e "{PS.out_dir / SP.step_amber_sim_ener}"      \\
                        -r "{PS.out_dir / SP.step_amber_sim_restart}"   \\
                        </dev/null""")
    ld_logger.info("Running pmemd sim")
    ld_logger.debug("Running {cmd}", cmd=AS.pmemd_prefix + cmd_sim)
    subprocess.run(
        AS.pmemd_prefix + cmd_sim,
        **app_settings.subprocess_settings.__dict__,
    )
    if not check_pmemd_out(PS.out_dir / SP.step_amber_sim_out, ld_logger):
        emsg = "The pmemd simulation of the structure failed."
        ld_logger.error(emsg)
        raise LDError(emsg) from None

    with open(
        PS.out_dir / SP.step_amber_ptraj_align_in, "w"
    ) as step_amber_ptraj_align_in_file:
        step_amber_ptraj_align_in_file.write(
            dedent(f"""\
                    parm "{PS.out_dir / SP.step_amber_top}" [initial-top]
                    parm "{PS.out_dir / PS.amber_pdb_target_top}" [target-top]
                    trajin "{PS.out_dir / SP.step_amber_sim_restart}" parm [initial-top]
                    reference "{PS.out_dir / PS.amber_target_min_algn}" parm [target-top] [target-ref]
                    rms ref [target-ref] :1-{resnum}@CA out {PS.out_dir / SP.step_amber_ptraj_rms_align_dat}
                    trajout "{PS.out_dir / SP.step_amber_ptraj_algn_restart}" restart parm [initial-top]
                       """)
        )
    ld_logger.trace(
        "Wrote file at {path}",
        path=PS.out_dir / SP.step_amber_ptraj_align_in,
    )

    cmd_align = f'cpptraj "{PS.out_dir / SP.step_amber_top}" "{PS.out_dir / SP.step_amber_ptraj_align_in}"'
    ld_logger.info("Running cpptraj")
    ld_logger.debug("Running {cmd}", cmd=AS.ambertools_prefix + cmd_align)
    subprocess.run(
        AS.ambertools_prefix + cmd_align,
        **app_settings.subprocess_settings.__dict__,
    )

    cmd_ambmask_AA = dedent(f"""\
                ambmask -p "{SP.step_amber_top}"                 \\
                        -c "{SP.step_amber_ptraj_algn_restart}"  \\
                        -prnlev 1 -out pdb""")

    ld_logger.info("Running ambmask AA")
    ld_logger.debug("Running {cmd}", cmd=AS.ambertools_prefix + cmd_ambmask_AA)
    with open(PS.out_dir / SP.step_anmld_pdb, "w") as step_ambmask_AA_pdb_f:
        subprocess.run(
            AS.ambertools_prefix + cmd_ambmask_AA,
            stdout=step_ambmask_AA_pdb_f,
            **app_settings.subprocess_settings.__dict__,
        )

    cmd_ambmask_CA = cmd_ambmask_AA + " -find @CA"
    ld_logger.info("Running ambmask CA")
    ld_logger.debug("Running {cmd}", cmd=AS.ambertools_prefix + cmd_ambmask_CA)
    with open(PS.out_dir / SP.step_anmld_CA_pdb, "w") as step_ambmask_CA_pdb_f:
        subprocess.run(
            AS.ambertools_prefix + cmd_ambmask_CA,
            stdout=step_ambmask_CA_pdb_f,
            **app_settings.subprocess_settings.__dict__,
        )

    ld_logger.debug("Aligning the LD result to the target")
    ld_aa = load_structure(str(PS.out_dir / SP.step_anm_pdb))
    aa_target = load_structure(str(PS.out_dir / PS.amber_pdb_target_min_pdb))
    aa_init = load_structure(str(PS.out_dir / PS.amber_pdb_initial_min_pdb))

    step_info = {}
    step_info["aa_rmsd_target"], step_info["ca_rmsd_target"] = calc_aa_ca_rmsd(
        aa_fixed=aa_target, aa_mobile=ld_aa, app_settings=app_settings
    )
    step_info["aa_rmsd_init"], step_info["ca_rmsd_init"] = calc_aa_ca_rmsd(
        aa_fixed=aa_init, aa_mobile=ld_aa, app_settings=app_settings
    )

    msg = (
        "Finished LD step with ",
        f"target AA RMSD: {step_info['aa_rmsd_target']} "
        f"target C-alpha RMSD: {step_info['ca_rmsd_target']} "
        f"initial AA RMSD: {step_info['aa_rmsd_init']} "
        f"initial C-alpha RMSD: {step_info['ca_rmsd_init']} ",
    )

    ld_logger.info(msg)

    return step_info
