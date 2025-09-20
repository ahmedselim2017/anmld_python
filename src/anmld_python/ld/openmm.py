from __future__ import annotations
from pathlib import Path
import math

from openmm import app as mm_app
from openmm.app.topology import Topology
from openmm.app.simulation import Simulation

from biotite.structure import AtomArray
import biotite.interface.openmm as b_mm
import biotite.structure as b_structure
import fastpdb

import loguru
import openmm as mm

from anmld_python.settings import AppSettings, StepPathSettings


def setup_sims(
    topology: Topology, app_settings: AppSettings
) -> tuple[Simulation, Simulation]:
    # TODO: set platform
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
    )

    return min_simulation, ld_simulation


def run_ld_step(
    pred_abs_path: Path,
    aa_anm: AtomArray,
    aa_target: AtomArray,
    ld_logger: loguru.Logger,
    app_settings: AppSettings,
    step_paths: StepPathSettings,
):
    MS = app_settings.openmm_settings

    anm_pdb = mm_app.PDBFile(str(pred_abs_path))  # TODO: get it from biotite
    min_simulation, ld_simulation = setup_sims(
        topology=anm_pdb.topology,
        app_settings=app_settings,
    )

    ld_logger.debug("Running minimization")
    min_simulation.context.setPositions(anm_pdb.positions)
    min_simulation.minimizeEnergy(maxIterations=MS.min_step)

    min_positions = min_simulation.context.getState(
        positions=True
    ).getPositions()

    if MS.save_min or app_settings.logging_level == "DEBUG":
        min_aa = b_mm.from_context(aa_anm, min_simulation.context)
        min_out_file = fastpdb.PDBFile()
        min_out_file.set_structure(min_aa)
        min_out_file.write(
            app_settings.path_settings.out_dir / step_paths.step_openmm_min
        )

    ld_logger.debug("Running Langevin dynamics simulation")

    ld_simulation.context.setPositions(min_positions)
    ld_simulation.context.setVelocitiesToTemperature(MS.ld_temp)  # tempi
    ld_simulation.step(MS.ld_step)  # nstlim

    ld_logger.debug("Ran Langevin dynamics simulation")
    ld_aa = b_mm.from_context(aa_anm, ld_simulation.context)

    if MS.save_ld or app_settings.logging_level == "DEBUG":
        ld_out_file = fastpdb.PDBFile()
        ld_out_file.set_structure(ld_aa)
        ld_out_file.write(
            app_settings.path_settings.out_dir / step_paths.step_openmm_ld
        )

    ld_aligned_aa, _ = b_structure.superimpose(fixed=aa_target, mobile=ld_aa)
    ld_rmsd = b_structure.rmsd(aa_target, ld_aligned_aa)

    ld_out_file = fastpdb.PDBFile()
    ld_out_file.set_structure(ld_aligned_aa)
    ld_out_file.write(
        app_settings.path_settings.out_dir / step_paths.step_anmld_pdb
    )
    ld_logger.info(f"Finished LD step {ld_rmsd=}")
