from pydantic import BaseModel, Field, PositiveInt, PositiveFloat
from pydantic_settings import BaseSettings, SettingsConfigDict


class AmberSettings(BaseSettings):
    temp: PositiveFloat = Field(310)  # TODO: float or int?
    min_step: PositiveInt = Field(500)
    sim_step: PositiveInt = Field(100)
    forcefield: str = Field("leaprc.protein.ff14SB")


class AppSettings(BaseSettings):
    n_cycle: PositiveInt = Field(100)
    rcut_GNM: PositiveFloat = Field(10)
    rcut_ANM: PositiveFloat = Field(8)
    DF: PositiveFloat = Field(0.6)
    max_mode: PositiveInt = Field(30)
    version: int = Field(0)

    amber_settings: AmberSettings = Field(alias='AmberSettings')

