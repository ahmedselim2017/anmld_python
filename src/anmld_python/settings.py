from pathlib import Path
from typing import Literal, Optional

from pydantic import (
    Field,
    PositiveInt,
    PositiveFloat,
    NewPath,
)
from pydantic_settings import BaseSettings

class StepPathSettings(BaseSettings):
    step_raw_pdb_format: str                = "STEP_{step}_raw.pdb"
    step_amber_tleap_anm_pdb_format: str    = "STEP_{step}_AMBER_tleap_anm_pdbs.in"

    step_amber_top: str     = "STEP_{step}_AMBER_raw.top"
    step_amber_coord: str   = "STEP_{step}_AMBER_raw.coord"

    step_amber_min_out: str     = "STEP_{step}_AMBER_min.out"
    step_amber_min_coord: str   = "STEP_{step}_AMBER_min.coord"
    step_amber_min_rst: str     = "STEP_{step}_AMBER_min.rst"

    step_amber_sim_out: str     = "STEP_{step}_AMBER_sim.out"
    step_amber_sim_coord: str   = "STEP_{step}_AMBER_sim.coord"
    step_amber_sim_ener: str    = "STEP_{step}_AMBER_sim.ener"
    step_amber_sim_restart: str = "STEP_{step}_AMBER_sim.restart"

    step_amber_ptraj_align_in: str      = "STEP_{step}_AMBER_sim_ptraj_align.in"
    step_amber_ptraj_rms_align_dat: str = "STEP_{step}_AMBER_rms.dat"
    step_amber_ptraj_algn_restart: str = "STEP_{step}_AMBER_algn.restart"

    # TODO: rename non-step ambnmask outs to be similar to step ones
    step_ambmask_AA_pdb: str = "STEP_{step}_ambmask_AA.pdb"
    step_ambmask_CA_pdb: str = "STEP_{step}_ambmask_CA.pdb"


    @classmethod
    def format_step(cls, step: int):
        # NOTE: A better way?
        f_dict = {}
        for key in cls.__dict__:
            if isinstance(f_dict[key], str):
                f_dict[key] = cls.__dict__[key].format(step)

        return StepPathSettings(**f_dict)


class PathSettings(BaseSettings):
    out_dir: NewPath = Field(Path("out"))

    sanitized_init_pdb_path: Path = Path("sanitized_init.pdb")
    sanitized_target_pdb_path: Path = Path("sanitized_target.pdb")

    amber_min_in: Path = Path("AMBER_min.in")
    amber_sim_in: Path = Path("AMBER_sim.in")

    amber_pdb_init_top: Path    = Path("AMBER_pdb_initial.top")
    amber_pdb_init_coord: Path  = Path("AMBER_pdb_initial.coord")

    amber_pdb_target_top: Path      = Path("AMBER_pdb_target.top")
    amber_pdb_target_coord: Path    = Path("AMBER_pdb_target.coord")

    amber_tleap_init_in: Path       = Path("AMBER_tleap_initial.in")

    amber_pdb_init_min_out: Path    = Path("AMBER_pdb_initial_min.out")
    amber_pdb_init_min_coord: Path  = Path("AMBER_pdb_initial_min.coord")
    amber_pdb_init_min_rst: Path    = Path("AMBER_pdb_initial_min.rst")

    amber_pdb_target_min_out: Path      = Path("AMBER_pdb_target_min.out")
    amber_pdb_target_min_coord: Path    = Path("AMBER_pdb_target_min.coord")
    amber_pdb_target_min_rst: Path      = Path("AMBER_pdb_target_min.rst")

    amber_ptraj_rewrite_init_in: Path       = Path("AMBER_ptraj_rewrite_initial.in")
    amber_pdb_rewrite_init_min_rst: Path    = Path("AMBER_initial_min.rst")

    amber_ptraj_align_target2initial_in: Path   = Path("AMBER_ptraj_align_target2initial.in")
    amber_rms_target_align_dat: Path            = Path("AMBER_rms_target_align.dat")
    amber_target_min_algn: Path                 = Path("AMBER_target_min_algn.rst")

    amber_pdb_initial_min_pdb: Path     = Path("AMBER_pdb_initial_min.pdb")
    amber_pdb_initial_min_c_pdb: Path   = Path("AMBER_pdb_initial_min_C.pdb")

    amber_pdb_target_min_pdb: Path     = Path("AMBER_pdb_target_min.pdb")
    amber_pdb_target_min_c_pdb: Path   = Path("AMBER_pdb_target_min_C.pdb")

    step_path_settings: StepPathSettings = Field(StepPathSettings() ,alias="STEP_PATHS")



class AmberSettings(BaseSettings):
    temp: PositiveFloat = Field(310)  # TODO: float or int?
    min_step: PositiveInt = Field(500)
    sim_step: PositiveInt = Field(100)
    forcefield: str = Field("leaprc.protein.ff14SB")
    cmd_prefix: str = "module load cuda/11.3 && module load amber/22_20240202 &&"


class ANMLDSettings(BaseSettings):
    n_steps: PositiveInt = Field(100)
    rcut_ANM: PositiveFloat = Field(8)
    gamma_ANM: float = Field(1.0)
    DF: PositiveFloat = Field(0.6)
    DF_SC_ratio: PositiveFloat = Field(1)
    max_mode: PositiveInt = Field(30)
    version: int = Field(0)


class AppSettings(BaseSettings):
    run_name: str = Field("anmld_run", alias="name")
    logging_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = (
        "INFO"
    )
    mode_selection: Literal["MATLAB"] = "MATLAB"

    anmld_settings: ANMLDSettings = Field(ANMLDSettings(), alias="ANMLD")
    amber_settings: AmberSettings = Field(AmberSettings(), alias="AMBER")
    path_settings: PathSettings = Field(PathSettings(), alias="PATHS")
