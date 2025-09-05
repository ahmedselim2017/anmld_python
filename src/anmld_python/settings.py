from pathlib import Path
from typing import Literal, Optional

from pydantic import (
    Field,
    PositiveInt,
    PositiveFloat,
    FilePath,
    NewPath,
)
from pydantic_settings import BaseSettings


class PathSettings(BaseSettings):
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


class AmberSettings(BaseSettings):
    temp: PositiveFloat = Field(310)  # TODO: float or int?
    min_step: PositiveInt = Field(500)
    sim_step: PositiveInt = Field(100)
    forcefield: str = Field("leaprc.protein.ff14SB")
    module_name: str


class ANMLDSettings(BaseSettings):
    n_cycle: PositiveInt = Field(100)
    rcut_ANM: PositiveFloat = Field(8)
    gamma_ANM: float = Field(1.0)
    DF: PositiveFloat = Field(0.6)
    DF_SC_ratio: PositiveFloat = Field(1)
    max_mode: PositiveInt = Field(30)
    version: int = Field(0)


class AppSettings(BaseSettings):
    run_name: str = Field("anmld_run", alias="name")
    structure_init: FilePath = Field(alias="initial")
    structure_target: FilePath = Field(alias="target")
    out_dir: NewPath = Field(Path("out"))
    logging_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = (
        "INFO"
    )
    mode_selection: Literal["MATLAB"] = "MATLAB"

    anmld_settings: ANMLDSettings = Field(alias="ANMLD")
    amber_settings: AmberSettings = Field(alias="AMBER")
    path_settings: PathSettings = Field(alias="PATHS")
