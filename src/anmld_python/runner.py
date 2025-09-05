from textwrap import dedent

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
    AS: AppSettings,
):
    ca_init = aa_init[(aa_init.atom_name == "CA") & (aa_init.element == "C")]
    hessian_init = ANM.build_hessian(
        coords=ca_init,
        cutoff=AppSettings.anmld_settings.rcut_ANM,
        gamma=AS.anmld_settings.gamma_ANM,
    )
    step_logger.debug("Calculated the Hessian matrix")
    W_init, V_init, Vx_init, Vy_init, Vz_init = ANM.calc_modes(
        hessian=hessian_init,
        mode_max=AS.anmld_settings.max_mode,
    )
    step_logger.debug("Calculated ANM modes")

    mode_selection = AS.mode_selection

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
                app_settings=AS,
            )

    pred_path = AS.out_dir / AS.path_settings.step_raw_pdb_format.format(
        step=step
    )

    pred_file = fastpdb.PDBFile()
    pred_file.set_structure(pred_aa)
    pred_file.write(pred_path)
    step_logger.info(f"Wrote raw step coordinates to {pred_path}")

    amber_tleap_step_in_path = (
        AS.out_dir
        / AS.path_settings.step_amber_tleap_anm_pdb_format.format(step=step)
    )

    with open(amber_tleap_step_in_path) as amber_tleap_step_in_f:
        amber_tleap_step_in_f.write(
            dedent(f"""\
                source {AS.amber_settings.forcefield}
                x=loadpdb {pred_path}
                saveamberparm x {AS.path_settings.step_amber_tleap_coord.format(step=step)}
                quit""")
        )
    amber_logger.trace("Wrote file at {path}", path=amber_tleap_step_in_path)
