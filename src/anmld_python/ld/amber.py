from __future__ import annotations
from pathlib import Path
from textwrap import dedent
import subprocess

import loguru
import parmed

from anmld_python.settings import AppSettings, StepPathSettings
from anmld_python.tools import LDError, calc_aa_ca_rmsd, get_atomarray


class tleapError(Exception):
    pass


def run_pdb4amber(
    in_path: Path,
    out_path: Path,
    stdout_stderr_redirection_path: Path,
    logger: loguru.Logger,
    app_settings: AppSettings,
    reduce: bool = True,
):
    logger.trace("run_pdb4amber")

    cmd = (
        app_settings.amber_settings.pdb4amber_prefix
        + app_settings.amber_settings.pdb4amber_cmd
    )

    cmd += f" -i {in_path} -o {out_path} --dry"
    if reduce:
        cmd += " --reduce"

    logger.info("Running pdb4amber")
    logger.debug(f"Running {app_settings.amber_settings.ambertools_prefix + cmd}")

    with open(stdout_stderr_redirection_path, "w") as f:
        subprocess.run(
            app_settings.amber_settings.ambertools_prefix + cmd,
            **app_settings.subprocess_settings.__dict__,
            stdout=f,
            stderr=subprocess.STDOUT,
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
            logger.trace(f"Found {keyword=} in pmemd output {out_path=}")
            return False
    return True


def safe_pmemd(
    overwrite: bool,
    mdin: Path,
    prmtop: Path,
    inpcrd: Path,
    mdout: Path,
    mdcrd: Path,
    restrt: Path,
    stdout_path: Path,
    ld_logger: loguru.Logger,
    app_settings: AppSettings,
    mden: Path | None = None,
):
    cmd_opts = [
        f"-i '{mdin}'",
        f"-p '{prmtop}'",
        f"-c '{inpcrd}'",
        f"-o '{mdout}'",
        f"-x '{mdcrd}'",
        f"-r '{restrt}'",
    ]
    if overwrite:
        cmd_opts.insert(0, "-O")
    if mden:
        cmd_opts.append(f"-e '{mden}'")

    AS = app_settings.amber_settings

    cmd = AS.pmemd_prefix + AS.pmemd_cmd + " " + " ".join(cmd_opts)
    ld_logger.info(f"Running pmemd min for {inpcrd}")
    ld_logger.debug("Running {cmd}", cmd=cmd)

    with open(stdout_path, "w") as f:
        subprocess.run(
            cmd,
            **app_settings.subprocess_settings.__dict__,
            stdout=f,
        )

    if check_pmemd_out(mdout, ld_logger):
        return True

    ld_logger.warning(
        f"pmemd command for {inpcrd} has failed. Retrying with double precision."
    )

    cmd = AS.pmemd_prefix + AS.pmemd_dpfp_cmd + " " + " ".join(cmd_opts)
    ld_logger.info(f"Running pmemd min for {inpcrd}")
    ld_logger.debug("Running {cmd}", cmd=cmd)

    with open(stdout_path, "w") as f:
        subprocess.run(
            cmd,
            **app_settings.subprocess_settings.__dict__,
            stdout=f,
        )

    return check_pmemd_out(mdout, ld_logger)


def run_setup(
    path_abs_init: Path,
    path_abs_target: Path,
    resnum: int,
    ld_logger: loguru.Logger,
    app_settings: AppSettings,
):
    AS = app_settings.amber_settings
    PS = app_settings.path_settings

    with open(PS._out_dir / PS.amber_min_in, "w") as AMBER_min_in_f:
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
              dx0 = {AS.min.dx0}
             &end
                   """)
        )
        ld_logger.trace(
            "Wrote file at {path}",
            path=PS._out_dir / PS.amber_min_in,
        )

    with open(PS._out_dir / PS.amber_sim_in, "w") as AMBER_sim_in_f:
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
            path=PS._out_dir / PS.amber_sim_in,
        )

    with open(PS._out_dir / PS.amber_tleap_init_in, "w") as AMBER_tleap_initial_f:
        AMBER_tleap_initial_f.write(
            dedent(f"""\
            source {AS.forcefield}
            x=loadpdb "{path_abs_init}"
            saveamberparm x "{PS._out_dir / PS.amber_pdb_init_top}" "{PS._out_dir / PS.amber_pdb_init_coord}"
            y=loadpdb "{path_abs_target}"
            saveamberparm y "{PS._out_dir / PS.amber_pdb_target_top}" "{PS._out_dir / PS.amber_pdb_target_coord}"
            quit
                   """)
        )
        ld_logger.trace(
            "Wrote file at {path}",
            path=PS._out_dir / PS.amber_tleap_init_in,
        )

    cmd_tleap = f'tleap -f "{PS._out_dir / PS.amber_tleap_init_in}"'
    ld_logger.info("Running tleap")
    ld_logger.debug("Running {cmd}", cmd=AS.ambertools_prefix + cmd_tleap)

    with open(PS._out_dir / PS.amber_tleap_init_stdout, "w") as f:
        subprocess.run(
            AS.ambertools_prefix + cmd_tleap,
            **app_settings.subprocess_settings.__dict__,
            stdout=f,
        )

    if (
        not (PS._out_dir / PS.amber_pdb_init_top).is_file()
        or not (PS._out_dir / PS.amber_pdb_init_coord).is_file()
        or not (PS._out_dir / PS.amber_pdb_target_top).is_file()
        or not (PS._out_dir / PS.amber_pdb_target_coord).is_file()
    ):
        emsg = (
            f"tleap command {cmd_tleap} failed to create output files."
            f" {PS.amber_pdb_init_top}, {PS.amber_pdb_init_coord}"
            f" {PS.amber_pdb_target_top}, {PS.amber_pdb_target_coord}"
        )
        ld_logger.error(emsg)
        raise tleapError().with_traceback(None) from None

    if not safe_pmemd(
        overwrite=True,
        mdin=PS._out_dir / PS.amber_min_in,
        prmtop=PS._out_dir / PS.amber_pdb_init_top,
        inpcrd=PS._out_dir / PS.amber_pdb_init_coord,
        mdout=PS._out_dir / PS.amber_pdb_init_min_out,
        mdcrd=PS._out_dir / PS.amber_pdb_init_min_coord,
        restrt=PS._out_dir / PS.amber_pdb_init_min_rst,
        stdout_path=PS._out_dir / PS.amber_pmemd_min_init_stdout,
        ld_logger=ld_logger,
        app_settings=app_settings,
    ):
        emsg = "The pmemd minimization of the initial structure failed."
        ld_logger.error(emsg)
        raise LDError(emsg) from None

    if not safe_pmemd(
        overwrite=True,
        mdin=PS._out_dir / PS.amber_min_in,
        prmtop=PS._out_dir / PS.amber_pdb_target_top,
        inpcrd=PS._out_dir / PS.amber_pdb_target_coord,
        mdout=PS._out_dir / PS.amber_pdb_target_min_out,
        mdcrd=PS._out_dir / PS.amber_pdb_target_min_coord,
        restrt=PS._out_dir / PS.amber_pdb_target_min_rst,
        stdout_path=PS._out_dir / PS.amber_pmemd_min_target_stdout,
        ld_logger=ld_logger,
        app_settings=app_settings,
    ):
        emsg = "The pmemd minimization of the target structure failed."
        ld_logger.error(emsg)
        raise LDError(emsg) from None

    # create initial_min.rst (rewrite to be able to read by ambmsk,
    # version problem as noted by the deprecated MATLAB version of ANMLD)
    with open(
        PS._out_dir / PS.amber_cpptraj_rewrite_init_in,
        "w",
    ) as f:
        f.write(
            dedent(f"""\
                    trajin "{PS._out_dir / PS.amber_pdb_init_min_rst}"
                    trajout "{PS._out_dir / PS.amber_pdb_rewrite_init_min_rst}" restart
                   """)
        )
        ld_logger.trace(
            "Wrote file at {path}",
            path=PS._out_dir / PS.amber_cpptraj_rewrite_init_in,
        )

    cmd_rewrite = dedent(f"""cpptraj                                    \\
                                "{PS._out_dir / PS.amber_pdb_init_top}"  \\
                                "{PS._out_dir / PS.amber_cpptraj_rewrite_init_in}\"""")
    ld_logger.info("Running cpptraj")
    ld_logger.debug("Ran {cmd}", cmd=AS.ambertools_prefix + cmd_rewrite)
    with open(PS._out_dir / PS.amber_cpptraj_rewrite_stdout, "w") as f:
        subprocess.run(
            AS.ambertools_prefix + cmd_rewrite,
            **app_settings.subprocess_settings.__dict__,
            stdout=f,
        )

    # Create AMBER_target_min_algn.rst
    with open(
        PS._out_dir / PS.amber_cpptraj_align_target2initial_in,
        "w",
    ) as f:
        f.write(
            dedent(f"""\
                    parm "{PS._out_dir / PS.amber_pdb_init_top}" [initial-top]
                    parm "{PS._out_dir / PS.amber_pdb_target_top}" [target-top]
                    trajin "{PS._out_dir / PS.amber_pdb_target_min_rst}" parm [target-top]
                    reference "{PS._out_dir / PS.amber_pdb_init_min_rst}" parm [initial-top] [initial-ref]
                    rms ref [initial-ref]  :1-{resnum}@CA out "{PS._out_dir / PS.amber_rms_target_align_dat}"
                    trajout "{PS._out_dir / PS.amber_target_min_algn}" restart parm [target-top]
                   """)
        )
        ld_logger.trace(
            "Wrote file at {path}",
            path=PS._out_dir / PS.amber_cpptraj_align_target2initial_in,
        )

    cmd_align = dedent(f"""cpptraj                                      \\
                            "{PS._out_dir / PS.amber_pdb_target_top}"    \\
                            "{PS._out_dir / PS.amber_cpptraj_align_target2initial_in}\"""")
    ld_logger.info("Running cpptraj")
    ld_logger.debug("Running {cmd}", cmd=AS.ambertools_prefix + cmd_align)
    with open(PS._out_dir / PS.amber_cpptraj_align_stdout, "w") as f:
        subprocess.run(
            AS.ambertools_prefix + cmd_align,
            **app_settings.subprocess_settings.__dict__,
            stdout=f,
        )

    parmed_init_parm = parmed.amber.AmberParm(str(PS._out_dir / PS.amber_pdb_init_top))
    parmed_init_parm.load_rst7(str(PS._out_dir / PS.amber_pdb_rewrite_init_min_rst))
    parmed.tools.actions.addPDB(
        parmed_init_parm,
        str(PS._out_dir / PS.filtered_init_structure),
    ).execute()
    parmed_init_parm.save(
        str(PS._out_dir / PS.amber_pdb_initial_min_pdb),
        renumber=False,
    )

    ld_logger.info(
        "Saved minimized initial structure to {path}",
        path=PS._out_dir / PS.amber_pdb_initial_min_pdb,
    )

    parmed_target_parm = parmed.amber.AmberParm(
        str(PS._out_dir / PS.amber_pdb_target_top)
    )
    parmed.tools.actions.addPDB(
        parmed_target_parm, str(PS._out_dir / PS.filtered_target_structure)
    ).execute()
    parmed_target_parm.load_rst7(str(PS._out_dir / PS.amber_target_min_algn))
    parmed_target_parm.save(
        str(PS._out_dir / PS.amber_pdb_target_min_pdb),
        renumber=False,
    )
    ld_logger.info(
        "Saved minimized target structure to {path}",
        path=PS._out_dir / PS.amber_pdb_target_min_pdb,
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

    amber_tleap_step_in_path = PS._out_dir / SP.step_amber_tleap_anm_pdb

    run_pdb4amber(
        in_path=pred_abs_path,
        out_path=PS._out_dir / SP.step_amber_anm_pdb,
        stdout_stderr_redirection_path=PS._out_dir
        / SP.step_amber_pdb4amber_stdout_stderr,
        logger=ld_logger,
        app_settings=app_settings,
        reduce=False,
    )

    with open(amber_tleap_step_in_path, "w") as amber_tleap_step_in_f:
        amber_tleap_step_in_f.write(
            dedent(f"""\
                    source {AS.forcefield}
                    x=loadpdb "{PS._out_dir / SP.step_amber_anm_pdb}"
                    saveamberparm x "{PS._out_dir / SP.step_amber_top}" "{PS._out_dir / SP.step_amber_coord}"
                    quit
                       """)
        )
    ld_logger.trace("Wrote file at {path}", path=amber_tleap_step_in_path)

    cmd_tleap = f'tleap -f "{amber_tleap_step_in_path}"'
    ld_logger.info("Running tleap")
    ld_logger.debug("Running {cmd}", cmd=AS.ambertools_prefix + cmd_tleap)

    with open(PS._out_dir / SP.step_amber_tleap_stdout, "w") as f:
        subprocess.run(
            AS.ambertools_prefix + cmd_tleap,
            **app_settings.subprocess_settings.__dict__,
            stdout=f,
        )

    if (
        not (PS._out_dir / SP.step_amber_top).is_file()
        or not (PS._out_dir / SP.step_amber_coord).is_file()
    ):
        emsg = (
            f"tleap command {cmd_tleap} failed to create output files."
            f" {SP.step_amber_top} and {SP.step_amber_coord}"
        )
        ld_logger.error(emsg)
        raise tleapError().with_traceback(None) from None

    if not safe_pmemd(
        overwrite=True,
        mdin=PS._out_dir / PS.amber_min_in,
        prmtop=PS._out_dir / SP.step_amber_top,
        inpcrd=PS._out_dir / SP.step_amber_coord,
        mdout=PS._out_dir / SP.step_amber_min_out,
        mdcrd=PS._out_dir / SP.step_amber_min_coord,
        restrt=PS._out_dir / SP.step_amber_min_rst,
        stdout_path=PS._out_dir / SP.step_amber_pmemd_min_stdout,
        ld_logger=ld_logger,
        app_settings=app_settings,
    ):
        emsg = "The pmemd minimization of the structure failed."
        ld_logger.error(emsg)
        raise LDError(emsg) from None

    if not safe_pmemd(
        overwrite=True,
        mdin=PS._out_dir / PS.amber_sim_in,
        prmtop=PS._out_dir / SP.step_amber_top,
        inpcrd=PS._out_dir / SP.step_amber_min_rst,
        mdout=PS._out_dir / SP.step_amber_sim_out,
        mdcrd=PS._out_dir / SP.step_amber_sim_coord,
        restrt=PS._out_dir / SP.step_amber_sim_restart,
        stdout_path=PS._out_dir / SP.step_amber_pmemd_sim_stdout,
        ld_logger=ld_logger,
        app_settings=app_settings,
        mden=PS._out_dir / SP.step_amber_sim_ener,
    ):
        emsg = "The pmemd LD simulation of the structure failed."
        ld_logger.error(emsg)
        raise LDError(emsg) from None

    with open(
        PS._out_dir / SP.step_amber_cpptraj_align_in, "w"
    ) as step_amber_cpptraj_align_in_file:
        step_amber_cpptraj_align_in_file.write(
            dedent(f"""\
                    parm "{PS._out_dir / SP.step_amber_top}" [initial-top]
                    parm "{PS._out_dir / PS.amber_pdb_target_top}" [target-top]
                    trajin "{PS._out_dir / SP.step_amber_sim_restart}" parm [initial-top]
                    reference "{PS._out_dir / PS.amber_target_min_algn}" parm [target-top] [target-ref]
                    rms ref [target-ref] :1-{resnum}@CA out {PS._out_dir / SP.step_amber_cpptraj_rms_align_dat}
                    trajout "{PS._out_dir / SP.step_amber_cpptraj_algn_restart}" restart parm [initial-top]
                       """)
        )
    ld_logger.trace(
        "Wrote file at {path}",
        path=PS._out_dir / SP.step_amber_cpptraj_align_in,
    )

    cmd_align = f'cpptraj "{PS._out_dir / SP.step_amber_top}" "{PS._out_dir / SP.step_amber_cpptraj_align_in}"'
    ld_logger.info("Running cpptraj")
    ld_logger.debug("Running {cmd}", cmd=AS.ambertools_prefix + cmd_align)
    with open(PS._out_dir / SP.step_amber_cpptraj_align_stdout, "w") as f:
        subprocess.run(
            AS.ambertools_prefix + cmd_align,
            **app_settings.subprocess_settings.__dict__,
            stdout=f,
        )

    parmed_parm = parmed.amber.AmberParm(str(PS._out_dir / SP.step_amber_top))
    parmed.tools.actions.addPDB(
        parmed_parm, PS._out_dir / PS.filtered_init_structure
    ).execute()
    parmed_parm.load_rst7(str(PS._out_dir / SP.step_amber_cpptraj_algn_restart))
    parmed_parm.save(
        str(PS._out_dir / SP.step_anmld_pdb),
        renumber=False,
    )
    ld_logger.info(
        "Saved LD structure to {path}",
        path=PS._out_dir / SP.step_anmld_pdb,
    )

    ld_logger.debug("Aligning the LD result to the target")
    ld_aa = get_atomarray(PS._out_dir / SP.step_anmld_pdb)
    aa_target = get_atomarray(PS._out_dir / PS.amber_pdb_target_min_pdb)
    aa_init = get_atomarray(PS._out_dir / PS.amber_pdb_initial_min_pdb)

    step_info = {}
    step_info["aa_rmsd_target"], step_info["ca_rmsd_target"] = calc_aa_ca_rmsd(
        aa_fixed=aa_target, aa_mobile=ld_aa, app_settings=app_settings
    )
    step_info["aa_rmsd_init"], step_info["ca_rmsd_init"] = calc_aa_ca_rmsd(
        aa_fixed=aa_init, aa_mobile=ld_aa, app_settings=app_settings
    )

    msg = (
        "Finished LD step with "
        f"target AA RMSD: {step_info['aa_rmsd_target']} "
        f"target C-alpha RMSD: {step_info['ca_rmsd_target']} "
        f"initial AA RMSD: {step_info['aa_rmsd_init']} "
        f"initial C-alpha RMSD: {step_info['ca_rmsd_init']} "
    )

    ld_logger.info(msg)

    return step_info
