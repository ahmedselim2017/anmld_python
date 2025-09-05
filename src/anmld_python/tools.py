from pathlib import Path
from typing import Optional, cast

from biotite.structure.atoms import AtomArray
import fastpdb
import numpy as np
import biotite.structure.io.pdbx as b_pdbx
import biotite.structure as b_structure


def get_atomarray(
    structure_path: Path,
    structure_index: int = 0,
    extra_fields: Optional[list | str] = None,
) -> AtomArray:
    if not extra_fields:
        extra_fields = ["atom_id", "b_factor", "occupancy", "charge"]

    if structure_path.suffix == ".pdb":
        structure_file = fastpdb.PDBFile.read(structure_path)

        atomarray = structure_file.get_structure(
            extra_fields=extra_fields,
            model=structure_index + 1,
        )
    elif structure_path.suffix == ".cif":
        structure_file = b_pdbx.CIFFile.read(structure_path)
        atomarray = b_pdbx.get_structure(
            structure_file,
            model=structure_index - 1,
            extra_fields=extra_fields,
        )
    else:
        emsg = f"Given structure file {structure_path} is not supported."
        raise ValueError(emsg)

    return cast(AtomArray, atomarray)


def sanitize_pdb(in_path: Path, out_path: Path, *args, **kwargs) -> AtomArray:
    aa = get_atomarray(in_path, *args, **kwargs)

    if np.unique(aa.chain_id).size != 1:
        emsg = f"Given structure should include only 1 chain, not {np.unique(aa.chain_id).size}."
        raise ValueError(emsg)

    # NOTE: other filters?
    aa = aa[b_structure.filter_amino_acids(aa)]

    out_file = fastpdb.PDBFile()
    out_file.set_structure(aa)
    out_file.write(out_path)
    
    return aa
