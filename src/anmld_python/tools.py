from pathlib import Path
from typing import Optional, cast

from biotite.structure.atoms import AtomArray
import fastpdb
import numpy as np
import biotite.structure.io.pdbx as b_pdbx
import biotite.structure as b_structure

from anmld_python.settings import AppSettings

class LDError(Exception):
    pass

def get_atomarray(
    structure_path: Path,
    structure_index: int = 0,
    extra_fields: Optional[list | str] = None,
    *args,
    **kwargs,
) -> AtomArray:
    if not extra_fields:
        extra_fields = []

    if structure_path.suffix == ".pdb":
        structure_file = fastpdb.PDBFile.read(structure_path)

        atomarray = structure_file.get_structure(
            extra_fields=extra_fields,
            model=structure_index + 1,
            *args,
            **kwargs,
        )
    elif structure_path.suffix == ".cif":
        structure_file = b_pdbx.CIFFile.read(structure_path)
        atomarray = b_pdbx.get_structure(
            structure_file,
            model=structure_index - 1,
            extra_fields=extra_fields,
            *args,
            **kwargs,
        )
    else:
        emsg = f"Given structure file {structure_path} is not supported."
        raise ValueError(emsg)

    return cast(AtomArray, atomarray)


def get_CAs(aa: AtomArray) -> AtomArray:
    if not (cas := aa[(aa.atom_name == "CA") & (aa.element == "C")]):
        cas = aa[(aa.atom_name == "CA")]
    return cas


def sanitize_pdb(
    in_path: Path,
    out_path: Path,
    app_settings: AppSettings,
    chain_id: Optional[str] = None,
    *args,
    **kwargs,
) -> AtomArray:
    aa = get_atomarray(in_path, *args, **kwargs)

    chains = np.unique(aa.chain_id)
    if chain_id:
        if chain_id not in chains:
            raise ValueError(
                f"The given {chain_id=} does not exists in the structure at {in_path.absolute()}."
            )
        aa = aa[aa.chain_id == chain_id]
    elif chains.size != 1:
        raise ValueError(
            f"The structure at {in_path} includes multiple chains {chains}. Please select a chain."
        )

    # NOTE: other filters?
    aa = aa[b_structure.filter_amino_acids(aa)]

    aa = aa[aa.element != "H"]
    out_file = fastpdb.PDBFile()
    out_file.set_structure(aa)
    out_file.write(out_path)

    if app_settings.LD_method == "OpenMM":
        import openmm.app as mm_app

        pdb = mm_app.PDBFile(str(out_path))
        modeller = mm_app.Modeller(pdb.topology, pdb.positions)

        mm_forcefield = mm_app.ForceField(
            app_settings.openmm_settings.forcefield,
            "implicit/hct.xml",  # AMBER igb=1
        )
        modeller.addHydrogens(mm_forcefield)

        with open(out_path, "w") as out_file:
            mm_app.PDBFile.writeFile(
                modeller.getTopology(),
                modeller.getPositions(),
                out_file,
            )

    return aa
