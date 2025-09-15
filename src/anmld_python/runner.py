from __future__ import annotations
from textwrap import dedent
import subprocess

from biotite.structure import AtomArray
import fastpdb
import loguru
import numpy as np

from anmld_python.settings import AppSettings
import anmld_python.anm as ANM
from anmld_python.tools import get_CAs


def run_step(
    aa_step: AtomArray,
    aa_target: AtomArray,
    step: int,
    step_logger: loguru.Logger,
    amber_logger: loguru.Logger,
    app_settings: AppSettings,
):
    AS = app_settings.amber_settings
    PS = app_settings.path_settings
    SP = app_settings.path_settings.step_path_settings.format_step(step)

    ca_step = get_CAs(aa_step)

    hessian_step = ANM.build_hessian(
        coords=ca_step.coord,
        cutoff=app_settings.anmld_settings.rcut_ANM,
        gamma=app_settings.anmld_settings.gamma_ANM,
    )
    step_logger.debug("Calculated the Hessian matrix")
    _, _, Vx_step, Vy_step, Vz_step = ANM.calc_modes(
        hessian=hessian_step,
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
                aa_step=aa_step,
                aa_target=aa_target,
                Vx_step=Vx_step,
                Vy_step=Vy_step,
                Vz_step=Vz_step,
                step_logger=step_logger,
                app_settings=app_settings,
            )

    pred_abs_path = PS.out_dir / SP.step_raw_pdb

    pred_file = fastpdb.PDBFile()
    pred_file.set_structure(pred_aa)
    pred_file.write(pred_abs_path)
    step_logger.info(f"Wrote raw step coordinates to {pred_abs_path}")

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
    amber_logger.trace("Wrote file at {path}", path=amber_tleap_step_in_path)

    cmd_tleap = f"tleap -f \"{amber_tleap_step_in_path}\""
    amber_logger.info("Running tleap")
    amber_logger.debug("Running {cmd}", cmd=AS.cmd_prefix + cmd_tleap)
    subprocess.run(
        AS.cmd_prefix + cmd_tleap,
        **app_settings.subprocess_settings.__dict__,
    )

    cmd_min = dedent(f"""\
                $AMBERHOME/bin/pmemd.cuda -O                    \\
                    -i "{PS.out_dir / PS.amber_min_in}"         \\
                    -p "{PS.out_dir / SP.step_amber_top}"       \\
                    -c "{PS.out_dir / SP.step_amber_coord}"     \\
                    -o "{PS.out_dir / SP.step_amber_min_out}"   \\
                    -x "{PS.out_dir / SP.step_amber_min_coord}" \\
                    -r "{PS.out_dir / SP.step_amber_min_rst}"   \\
                    </dev/null""")
    amber_logger.info("Running pmemd min")
    amber_logger.debug("Running {cmd}", cmd=AS.cmd_prefix + cmd_min)
    subprocess.run(
        AS.cmd_prefix + cmd_min,
        **app_settings.subprocess_settings.__dict__,
    )

    cmd_sim = dedent(f"""\
                $AMBERHOME/bin/pmemd.cuda -O                        \\
                    -i "{PS.out_dir / PS.amber_sim_in}"             \\
                    -p "{PS.out_dir / SP.step_amber_top}"           \\
                    -c "{PS.out_dir / SP.step_amber_min_rst}"       \\
                    -o "{PS.out_dir / SP.step_amber_sim_out}"       \\
                    -x "{PS.out_dir / SP.step_amber_sim_coord}"     \\
                    -e "{PS.out_dir / SP.step_amber_sim_ener}"      \\
                    -r "{PS.out_dir / SP.step_amber_sim_restart}"   \\
                    </dev/null""")
    amber_logger.info("Running pmemd sim")
    amber_logger.debug("Running {cmd}", cmd=AS.cmd_prefix + cmd_sim)
    subprocess.run(
        AS.cmd_prefix + cmd_sim,
        **app_settings.subprocess_settings.__dict__,
    )

    resnum: int = np.unique(aa_step.res_id).size
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
    amber_logger.trace(
        "Wrote file at {path}", path=PS.out_dir / SP.step_amber_ptraj_align_in
    )

    cmd_align = f"cpptraj \"{PS.out_dir / SP.step_amber_top}\" \"{PS.out_dir / SP.step_amber_ptraj_align_in}\""
    amber_logger.info("Running cpptraj")
    amber_logger.debug("Running {cmd}", cmd=AS.cmd_prefix + cmd_align)
    subprocess.run(
        AS.cmd_prefix + cmd_align,
        **app_settings.subprocess_settings.__dict__,
    )

    cmd_ambmask_AA = dedent(f"""\
            ambmask -p "{SP.step_amber_top}"                 \\
                    -c "{SP.step_amber_ptraj_algn_restart}"  \\
                    -prnlev 1 -out pdb""")

    amber_logger.info("Running ambmask AA")
    amber_logger.debug("Running {cmd}", cmd=AS.cmd_prefix + cmd_ambmask_AA)
    with open(
        PS.out_dir / SP.step_ambmask_AA_pdb, "w"
    ) as step_ambmask_AA_pdb_f:
        subprocess.run(
            AS.cmd_prefix + cmd_ambmask_AA,
            stdout=step_ambmask_AA_pdb_f,
            **app_settings.subprocess_settings.__dict__,
        )

    cmd_ambmask_CA = cmd_ambmask_AA + " -find @CA"
    amber_logger.info("Running ambmask CA")
    amber_logger.debug("Running {cmd}", cmd=AS.cmd_prefix + cmd_ambmask_CA)
    with open(
        PS.out_dir / SP.step_ambmask_CA_pdb, "w"
    ) as step_ambmask_CA_pdb_f:
        subprocess.run(
            AS.cmd_prefix + cmd_ambmask_CA,
            stdout=step_ambmask_CA_pdb_f,
            **app_settings.subprocess_settings.__dict__,
        )
