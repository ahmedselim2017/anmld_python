from __future__ import annotations
from pathlib import Path
import math

from openmm import app as mm_app
from openmm.app.topology import Topology
from openmm.app.simulation import Simulation

from biotite.structure import AtomArray
import biotite.interface.openmm as b_mm
import biotite.structure as b_structure

import loguru
import openmm as mm

from anmld_python.settings import AppSettings, StepPathSettings
from anmld_python.tools import (
    calc_aa_ca_rmsd,
    get_CAs,
    get_atomarray,
    safe_superimpose,
    write_atomarray,
)


def setup_sims(
    topology: Topology,
    ld_logger: loguru.Logger,
    app_settings: AppSettings,
) -> tuple[Simulation, Simulation]:
    # TODO: set seed if given

    # NOTE: AMBER uses solventDielectric=78.5 whereas OpenMM uses
    # solventDielectric=78.3
    # NOTE: ncyc=50 and drms = 0.01 is not used
    # NOTE: in LD simulation, ntf is not needed

    MS = app_settings.openmm_settings

    mm_forcefield = mm_app.ForceField(
        MS.forcefield,
        "implicit/hct.xml",  # AMBER igb=1
    )

    # κ = 367.434915 * sqrt(I/(ϵT))
    # I -> ionic strength (for 1:1, saltcon)
    # ϵ -> solvent solventDielectric NOTE: different than AMBER
    # T -> Temperature NOTE: AMBER defaults to 300
    min_kappa = 367.434915 * math.sqrt(0.1 / (300 * 78.3))
    ld_kappa = 367.434915 * math.sqrt(0.1 / (MS.ld_temp * 78.3))

    min_system = mm_forcefield.createSystem(
        topology=topology,
        nonbondedMethod=mm_app.NoCutoff,  # ntb=0, cut=1000.0
        implicitSolventKappa=min_kappa,  # saltcon=0.1
    )

    min_integrator = mm.LangevinIntegrator(
        MS.ld_temp,  # temp0
        5,  # gamma_ln
        0.002,  # dt
    )
    min_simulation = mm_app.Simulation(
        topology=topology,
        system=min_system,
        integrator=min_integrator,
        platform=MS.platform,
    )

    ld_system = mm_forcefield.createSystem(
        topology,
        nonbondedMethod=mm_app.NoCutoff,  # ntb=0, cut=1000.0
        implicitSolventKappa=ld_kappa,  # saltcon=0.1
        constraints=mm_app.HBonds,  # ntc=2
    )

    ld_integrator = mm.LangevinIntegrator(
        MS.ld_temp,  # temp0
        5,  # gamma_ln
        0.002,  # dt
    )

    ld_simulation = mm_app.Simulation(
        topology=topology,
        system=ld_system,
        integrator=ld_integrator,
        platform=MS.platform,
    )

    return min_simulation, ld_simulation


def run_setup(
    path_init: Path,
    path_target: Path,
    init_min_sim: Simulation,
    ld_logger: loguru.Logger,
    app_settings: AppSettings,
):
    PS = app_settings.path_settings
    MS = app_settings.openmm_settings

    pdb_init = mm_app.PDBFile(str(path_init))
    pdb_target = mm_app.PDBFile(str(path_target))

    aa_init = b_mm.from_topology(pdb_init.topology)
    aa_target = b_mm.from_topology(pdb_target.topology)

    # NOTE: min_sim is assumed to be not used
    ld_logger.debug("Running minimization for the initial structure.")
    init_min_sim.context.setPositions(pdb_init.positions)
    init_min_sim.minimizeEnergy(maxIterations=MS.min_step)

    min_init_aa = b_mm.from_context(aa_init, init_min_sim.context)
    if MS.save_ld or app_settings.logging_level == "DEBUG":
        write_atomarray(
            aa=min_init_aa,
            out_path=PS.out_dir / PS.openmm_min_init_pdb,
        )

    init_min_sim.context.reinitialize()

    ld_logger.debug("Running minimization for the target structure.")

    target_min_sim, _ = setup_sims(
        topology=pdb_target.topology,
        ld_logger=ld_logger,
        app_settings=app_settings,
    )
    target_min_sim.context.setPositions(pdb_target.positions)
    target_min_sim.minimizeEnergy(maxIterations=MS.min_step)

    min_target_aa = b_mm.from_context(aa_target, target_min_sim.context)

    write_atomarray(
        aa=min_target_aa,
        out_path=PS.out_dir / PS.openmm_min_target_pdb,
    )

    ld_logger.debug("Aligning the minimized initial to the minimized target")

    min_aligned_init_aa = safe_superimpose(
        aa_fixed=min_target_aa,
        aa_mobile=min_init_aa,
        app_settings=app_settings,
    )

    write_atomarray(
        aa=min_aligned_init_aa,
        out_path=PS.out_dir / PS.openmm_min_aligned_init_pdb,
    )


def run_ld_step(
    pred_abs_path: Path,
    aa_anm: AtomArray,
    aa_target: AtomArray,
    min_sim: Simulation,
    ld_sim: Simulation,
    ld_logger: loguru.Logger,
    app_settings: AppSettings,
    step_paths: StepPathSettings,
) -> dict:
    MS = app_settings.openmm_settings
    PS = app_settings.path_settings

    ld_logger.debug("Reinitializing simulations")
    min_sim.context.reinitialize()
    ld_sim.context.reinitialize()

    anm_pdb = mm_app.PDBFile(str(pred_abs_path))

    ld_logger.debug("Running minimization")
    min_sim.context.setPositions(anm_pdb.positions)
    min_sim.minimizeEnergy(maxIterations=MS.min_step)

    min_positions = min_sim.context.getState(positions=True).getPositions()

    if MS.save_min or app_settings.logging_level == "DEBUG":
        write_atomarray(
            aa=b_mm.from_context(aa_anm, min_sim.context),
            out_path=PS.out_dir / step_paths.step_openmm_min,
        )

    ld_logger.debug("Running Langevin dynamics simulation")

    ld_sim.context.setPositions(min_positions)
    ld_sim.context.setVelocitiesToTemperature(MS.ld_temp)  # tempi
    ld_sim.step(MS.ld_step)  # nstlim

    ld_logger.debug("Ran Langevin dynamics simulation")
    ld_aa = b_mm.from_context(aa_anm, ld_sim.context)

    if MS.save_ld or app_settings.logging_level == "DEBUG":
        write_atomarray(
            aa=ld_aa,
            out_path=PS.out_dir / step_paths.step_openmm_ld,
        )

    ld_logger.debug("Aligning the LD result to the target")

    ld_aligned_aa = safe_superimpose(
        aa_fixed=aa_target,
        aa_mobile=ld_aa,
        app_settings=app_settings,
    )

    ld_aa_rmsd_target = None
    if not app_settings.different_topologies:
        ld_aa_rmsd_target = b_structure.rmsd(aa_target, ld_aligned_aa)

    ld_ca = get_CAs(ld_aa)
    target_ca = get_CAs(aa_target)

    ld_aligned_ca, _ = b_structure.superimpose(fixed=target_ca, mobile=ld_ca)
    ld_ca_rmsd_target = b_structure.rmsd(target_ca, ld_aligned_ca)

    write_atomarray(
        aa=ld_aligned_aa,
        out_path=PS.out_dir / step_paths.step_anmld_pdb,
    )

    aa_init = get_atomarray(PS.out_dir / PS.openmm_min_aligned_init_pdb)
    step_info = {
        "aa_rmsd_target": ld_aa_rmsd_target,
        "ca_rmsd_target": ld_ca_rmsd_target,
    }
    step_info["aa_rmsd_init"], step_info["ca_rmsd_init"] = calc_aa_ca_rmsd(
        aa_fixed=aa_init,
        aa_mobile=ld_aa,
        app_settings=app_settings
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
