from textwrap import dedent
import subprocess

from loguru import logger
from biotite.structure import AtomArray
import fastpdb
import loguru

from anmld_python.settings import AppSettings
import anmld_python.anm as ANM


def run_step(
    aa_init: AtomArray,
    aa_target: AtomArray,
    step: int,
    step_logger: loguru.Logger,
    amber_logger: loguru.Logger,
    app_settings: AppSettings,
):
    AS = app_settings.amber_settings
    PS = app_settings.path_settings
    SP = app_settings.path_settings.step_path_settings.format_step(step)

    ca_init = aa_init[(aa_init.atom_name == "CA") & (aa_init.element == "C")]
    hessian_init = ANM.build_hessian(
        coords=ca_init,
        cutoff=AppSettings.anmld_settings.rcut_ANM,
        gamma=app_settings.anmld_settings.gamma_ANM,
    )
    step_logger.debug("Calculated the Hessian matrix")
    W_init, V_init, Vx_init, Vy_init, Vz_init = ANM.calc_modes(
        hessian=hessian_init,
        mode_max=app_settings.anmld_settings.max_mode,
    )
    step_logger.debug("Calculated ANM modes")

    mode_selection = app_settings.mode_selection

    step_logger.info(
        "Using {mode_selection} method to select modes",
        mode_selection=mode_selection,
    )
    match mode_selection:
        case "MATLAB":
            import anmld_python.mode_selection.matlab_select as matlab_select

            pred_aa = matlab_select.generate_structures(
                aa_init=aa_init,
                aa_target=aa_target,
                Vx_init=Vx_init,
                Vy_init=Vy_init,
                Vz_init=Vz_init,
                step_logger=step_logger,
                app_settings=app_settings,
            )

    pred_path = PS.out_dir / SP.step_raw_pdb_format

    pred_file = fastpdb.PDBFile()
    pred_file.set_structure(pred_aa)
    pred_file.write(pred_path)
    step_logger.info(f"Wrote raw step coordinates to {pred_path}")

    amber_tleap_step_in_path = PS.out_dir / SP.step_amber_tleap_anm_pdb_format

    with open(amber_tleap_step_in_path) as amber_tleap_step_in_f:
        amber_tleap_step_in_f.write(
            dedent(f"""\
                source {AS.forcefield}
                x=loadpdb {pred_path}
                saveamberparm x {PS.out_dir / SP.step_amber_coord}
                quit""")
        )
    amber_logger.trace("Wrote file at {path}", path=amber_tleap_step_in_path)

    cmd_tleap = f"tleap -f {amber_tleap_step_in_path}"
    subprocess.run(
        AS.cmd_prefix + cmd_tleap,
        shell=True,
        cwd=PS.out_dir,
        check=True,
    )
    amber_logger.debug("Ran {cmd}", cmd=AS.cmd_prefix + cmd_tleap)

    cmd_min = dedent(f"""\
                $AMBERHOME/bin/pmemd.cuda -O                    \\
                    -i {PS.out_dir / PS.amber_min_in}           \\
                    -p {PS.out_dir / SP.step_amber_top}         \\
                    -c {PS.out_dir / SP.step_amber_coord}       \\
                    -o {PS.out_dir / SP.step_amber_min_out}     \\
                    -x {PS.out_dir / SP.step_amber_min_coord}   \\
                    -r {PS.out_dir / SP.step_amber_min_rst}     \\
                    </dev/null""")
    subprocess.run(
        AS.cmd_prefix + cmd_min,
        shell=True,
        cwd=PS.out_dir,
        check=True,
    )
    amber_logger.debug("Ran {cmd}", cmd=AS.cmd_prefix + cmd_min)

    cmd_sim = dedent(f"""\
                $AMBERHOME/bin/pmemd.cuda -O                    \\
                    -i {PS.out_dir / PS.amber_sim_in}           \\
                    -p {PS.out_dir / SP.step_amber_top}         \\
                    -c {PS.out_dir / SP.step_amber_min_rst}     \\
                    -o {PS.out_dir / SP.step_amber_sim_out}     \\
                    -x {PS.out_dir / SP.step_amber_sim_coord}   \\
                    -e {PS.out_dir / SP.step_amber_sim_ener}    \\
                    -r {PS.out_dir / SP.step_amber_sim_restart} \\
                    </dev/null""")
    subprocess.run(
        AS.cmd_prefix + cmd_sim,
        shell=True,
        cwd=PS.out_dir,
        check=True,
    )
    amber_logger.debug("Ran {cmd}", cmd=AS.cmd_prefix + cmd_sim)

    with open(
        PS.out_dir / SP.step_amber_ptraj_align_in
    ) as step_amber_ptraj_align_in_file:
        step_amber_ptraj_align_in_file.write(
            dedent(f"""\
                parm {PS.out_dir / SP.step_amber_top} [initial-top]
                parm {PS.out_dir / PS.amber_pdb_target_top} [target-top]
                trajin {PS.out_dir / SP.step_amber_sim_restart} parm [initial-top]
                reference {PS.out_dir / PS.amber_target_min_algn} [target-top] [target-ref]
                rms ref [target-ref] :1-%d@CA out {PS.out_dir / SP.step_amber_ptraj_rms_align_dat}
                trajout {PS.out_dir / SP.step_amber_ptraj_algn_restart} restart parm [initial-top]
            """)
        )
    amber_logger.trace(
        "Wrote file at {path}", path=PS.out_dir / SP.step_amber_ptraj_align_in
    )

    cmd_align = f"cpptraj {PS.out_dir / SP.step_amber_top} {PS.out_dir / SP.step_amber_ptraj_align_in}"
    subprocess.run(
        AS.cmd_prefix + cmd_align,
        shell=True,
        cwd=PS.out_dir,
        check=True,
    )
    amber_logger.debug("Ran {cmd}", cmd=AS.cmd_prefix + cmd_align)

    cmd_ambmask_AA = dedent(f"""\
            ambmask -p {PS.out_dir / SP.step_amber_top}                 \\
                    -c {PS.out_dir / SP.step_amber_ptraj_algn_restart}  \\
                    -prnlev 1 -out pdb""")

    with open(PS.out_dir / SP.step_ambmask_AA_pdb) as step_ambmask_AA_pdb_f:
        subprocess.run(
            AS.cmd_prefix + cmd_ambmask_AA,
            shell=True,
            cwd=PS.out_dir,
            stdout=step_ambmask_AA_pdb_f,
            check=True,
        )
        amber_logger.debug("Ran {cmd}", cmd=AS.cmd_prefix + cmd_ambmask_AA)

    cmd_ambmask_CA = cmd_ambmask_AA + " -find @CA"
    with open(PS.out_dir / SP.step_ambmask_AA_pdb) as step_ambmask_CA_pdb_f:
        subprocess.run(
            AS.cmd_prefix + cmd_ambmask_CA,
            shell=True,
            cwd=PS.out_dir,
            stdout=step_ambmask_CA_pdb_f,
            check=True,
        )
        amber_logger.debug("Ran {cmd}", cmd=AS.cmd_prefix + cmd_ambmask_CA)
