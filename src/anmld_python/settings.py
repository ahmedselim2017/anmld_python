from pathlib import Path
from typing import Literal, Optional

from pydantic import (
    Field,
    PositiveInt,
    PositiveFloat,
    NewPath,
    BaseModel
)
from pydantic_settings import BaseSettings

class SubprocessSettings(BaseModel):
    shell: bool = True
    cwd: Optional[Path] = None
    check: bool = True
    executable: Path = Path("/bin/bash")

class StepPathSettings(BaseSettings):
    step_anm_pdb: str                = "STEP_{step}_ANM.pdb"

    step_openmm_min: str    = "STEP_{step}_OpenMM_min.pdb"
    step_openmm_ld: str     = "STEP_{step}_OpenMM_ld.pdb"

    step_amber_tleap_anm_pdb: str    = "STEP_{step}_AMBER_tleap_anm_pdbs.in"
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

    step_anmld_pdb: str = "STEP_{step}_ANMLD.pdb"
    step_anmld_CA_pdb: str = "STEP_{step}_ANMLD_CA.pdb"


    def format_step(self, step: int):
        dump = self.model_dump()
        f_annot = {}
        for key in dump:
            f_annot[key] = getattr(self, key)
            if isinstance(getattr(self, key), str):
                f_annot[key] = getattr(self, key).format(step=step)

        return StepPathSettings(**f_annot)


class PathSettings(BaseSettings):
    out_dir: Path = Field(Path("out_openmm"))

    sanitized_init_structure: str = "sanitized_init.pdb"
    sanitized_target_structure: str = "sanitized_target.pdb"

    openmm_min_init_pdb: str            = "OpenMM_min_init.pdb"
    openmm_min_aligned_init_pdb: str    = "OpenMM_min_aligned_init.pdb"
    openmm_min_target_pdb: str          = "OpenMM_min_target.pdb"

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
    temp: PositiveFloat = Field(310)
    min_step: PositiveInt = Field(500)
    ld_step: PositiveInt = Field(100)
    forcefield: str = Field("leaprc.protein.ff14SB")
    cmd_prefix: str = "module load cuda/11.3 && module load amber/22_20231219 && "

class OpenMMSettings(BaseSettings):
    forcefield : str = Field("amber14/protein.ff14SB.xml")
    min_step: PositiveInt = Field(500)
    ld_step: PositiveInt = Field(100)
    ld_temp: PositiveFloat = Field(310)

    save_min: bool = False
    save_ld: bool = False

    platform_name: Literal["CPU", "OpenCL", "CUDA", "HIP"] = Field(
        "CUDA", alias="platform"
    )


class ANMLDSettings(BaseSettings):
    n_steps: PositiveInt = Field(100)
    rcut_ANM: PositiveFloat = Field(8)
    gamma_ANM: float = Field(1.0)
    DF: PositiveFloat = Field(0.6)
    DF_SC_ratio: PositiveFloat = Field(1)
    max_mode: PositiveInt = Field(30)
    early_stopping_rmsd: PositiveFloat = Field(2)


class AppSettings(BaseSettings):
    run_name: str = Field("anmld_run", alias="name")
    logging_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = (
        "DEBUG"
    )
    mode_selection: Literal["MATLAB"] = "MATLAB"
    LD_method: Literal["AMBER", "OpenMM"] = "OpenMM"

    anmld_settings: ANMLDSettings = Field(ANMLDSettings(), alias="ANMLD")   # type: ignore
    amber_settings: AmberSettings = Field(AmberSettings(), alias="AMBER")   # type: ignore
    openmm_settings: OpenMMSettings = Field(OpenMMSettings(), alias="OpenMM")   # type: ignore

    path_settings: PathSettings = Field(PathSettings(), alias="PATHS")      # type: ignore
    subprocess_settings: SubprocessSettings = Field(SubprocessSettings(), alias="SUBPROCESS") # type:ignore
