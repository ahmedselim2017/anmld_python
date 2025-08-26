from pathlib import Path
from typing import Literal

from pydantic import (
    Field,
    PositiveInt,
    PositiveFloat,
    FilePath,
    NewPath,
)
from pydantic_settings import BaseSettings


class AmberSettings(BaseSettings):
    temp: PositiveFloat = Field(310)  # TODO: float or int?
    min_step: PositiveInt = Field(500)
    sim_step: PositiveInt = Field(100)
    forcefield: str = Field("leaprc.protein.ff14SB")
    module_name: str



class ANMLDSettings(BaseSettings):
    n_cycle: PositiveInt = Field(100)
    rcut_GNM: PositiveFloat = Field(10)
    rcut_ANM: PositiveFloat = Field(8)
    DF: PositiveFloat = Field(0.6)
    max_mode: PositiveInt = Field(30)
    version: int = Field(0)


class AppSettings(BaseSettings):
    run_name: str = Field("anmld_run", alias="name")
    structure_init: FilePath = Field(alias="initial")
    structure_target: FilePath = Field(alias="target")
    out_dir: NewPath = Field(Path("out"))
    logging_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"

    anmld_settings: ANMLDSettings = Field(alias="ANMLD")
    amber_settings: AmberSettings = Field(alias="AMBER")
