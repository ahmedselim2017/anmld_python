import logging

from biotite.structure import AtomArray
import fastpdb

from anmld_python.settings import AppSettings
import anmld_python.anm as ANM


def run_step(
    step: int,
    app_settings: AppSettings,
    aa_init: AtomArray,
    aa_target: AtomArray,
):
    ca_init = aa_init[(aa_init.atom_name == "CA") & (aa_init.element == "C")]
    hessian_init = ANM.build_hessian(
        coords=ca_init,
        cutoff=AppSettings.anmld_settings.rcut_ANM,
        gamma=app_settings.anmld_settings.gamma_ANM,
    )
    W_init, V_init, Vx_init, Vy_init, Vz_init = ANM.calc_modes(
        hessian=hessian_init,
        mode_max=app_settings.anmld_settings.max_mode,
    )

    mode_selection = app_settings.mode_selection

    logging.info(f"{step=} | Mode selection method is: {mode_selection}")
    match mode_selection:
        case "MATLAB":
            import anmld_python.mode_selection.matlab_select as matlab_select

            pred_aa = matlab_select.generate_structures(
                step=step,
                app_settings=app_settings,
                aa_init=aa_init,
                aa_target=aa_target,
                Vx_init=Vx_init,
                Vy_init=Vy_init,
                Vz_init=Vz_init,
            )

    pred_path = app_settings.out_dir / f"RAW_step_{step}.pdb"

    pred_file = fastpdb.PDBFile()
    pred_file.set_structure(pred_aa)
    pred_file.write(pred_path)

    logging.info(f"{step=} | Wrote predicted file to {pred_path}")
