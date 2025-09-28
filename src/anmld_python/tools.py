from pathlib import Path
from typing import Optional, cast

from biotite.structure.atoms import AtomArray
from loguru import logger
import biotite.structure as b_structure
import biotite.structure.io.pdb as b_pdb
import biotite.structure.io.pdbx as b_pdbx
import fastpdb
import numpy as np
import openmm as mm
import openmm.app as mm_app
import pdbfixer

from anmld_python.settings import AppSettings


class LDError(Exception):
    pass


class NonConnectedStructureError(Exception):
    pass


def write_atomarray(aa: AtomArray, out_path: Path):
    match out_path.suffix:
        case ".pdb":
            pdb_file = fastpdb.PDBFile()
            pdb_file.set_structure(aa)
            pdb_file.write(out_path)
        case ".cif":
            cif_file = b_pdbx.CIFFile()
            b_pdbx.set_structure(cif_file, aa)
            cif_file.write(out_path)


def get_atomarray(
    structure_path: Path,
    structure_index: int = 0,
    extra_fields: Optional[list | str] = None,
    *args,
    **kwargs,
) -> AtomArray:
    if not extra_fields:
        extra_fields = []

    match structure_path.suffix:
        case ".pdb":
            # fastpdb might panic while loading bonds
            # https://github.com/biotite-dev/fastpdb/pull/25
            try:
                structure_file = fastpdb.PDBFile.read(structure_path)
                atomarray = structure_file.get_structure(
                    extra_fields=extra_fields,
                    model=structure_index + 1,
                    *args,
                    **kwargs,
                )
            except BaseException:
                logger.warning(
                    "fastpdb panicked while loading the structure, using biotite to load the structure."
                )
                structure_file = b_pdb.PDBFile.read(structure_path)
                atomarray = structure_file.get_structure(
                    extra_fields=extra_fields,
                    model=structure_index + 1,
                    *args,
                    **kwargs,
                )
        case ".cif":
            structure_file = b_pdbx.CIFFile.read(structure_path)
            atomarray = b_pdbx.get_structure(
                structure_file,
                model=structure_index - 1,
                extra_fields=extra_fields,
                *args,
                **kwargs,
            )
        case _:
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
    sel_chains: Optional[list[str]] = None,
    *args,
    **kwargs,
) -> AtomArray:
    logger.debug("Loading atmarray")
    aa = get_atomarray(in_path, *args, **kwargs)
    logger.debug("Loaded atmarray")

    chains = np.unique(aa.chain_id)
    if sel_chains:
        if not np.all(np.isin(sel_chains, chains)):
            raise ValueError(
                f"The given {sel_chains=} does not exists in the structure at {in_path.absolute()}."
            )
        aa = aa[np.isin(aa.chain_id, sel_chains)]

    # NOTE: other filters?
    logger.debug("Filtering aminoacids")
    aa = aa[b_structure.filter_amino_acids(aa)]

    aa = aa[aa.element != "H"]
    out_file = fastpdb.PDBFile()
    out_file.set_structure(aa)
    out_file.write(out_path)

    err = None
    fixer = pdbfixer.PDBFixer(
        filename=str(out_path),
        platform=app_settings.openmm_settings.platform,
    )
    fixer.missingResidues = {}
    fixer.findMissingAtoms()
    for i in range(app_settings.sanitization_max_retry):
        logger.debug(f"Adding missing heavy atoms to the structure {in_path}", i=i)
        try:
            # Re-load structure s
            fixer.addMissingAtoms()

            topology = fixer.topology
            positions = fixer.positions
            break
        except Exception as err:
            logger.warning(
                f"PDBFixer did not run successfully. Retrying ({i + 1}/{app_settings.sanitization_max_retry})",
                err=err,
            )
    else:
        if err:
            logger.error(
                f"PDBFixer could not resolve clashes at the structure {in_path} while adding missing heavy atoms.",
                err=err,
            )
            raise err from None
    logger.info(f"Added missing heavy atoms to the structure {in_path}")

    if app_settings.LD_method == "OpenMM":
        modeller = mm_app.Modeller(topology, positions)

        mm_forcefield = mm_app.ForceField(
            app_settings.openmm_settings.forcefield,
            "implicit/hct.xml",  # AMBER igb=1
        )
        err = None
        for i in range(app_settings.sanitization_max_retry):
            try:
                logger.debug(f"Adding Hydrogens atoms to the structure {in_path}", i=i)
                modeller.addHydrogens(
                    mm_forcefield,
                    platform=app_settings.openmm_settings.platform,
                )

                topology = modeller.getTopology()
                positions = modeller.getPositions()
                break
            except mm.OpenMMException as err:
                logger.warning(
                    f"Could not add Hydrogens. Retrying ({i + 1}/{app_settings.sanitization_max_retry})",
                    err=err,
                )
        else:
            if err:
                logger.error(
                    f"Could not add Hydrogens to the structure {in_path}.",
                    err=err,
                )
                raise err from None
        logger.debug(f"Added Hydrogens atoms to the structure {in_path}")

    with open(out_path, "w") as out_file:
        mm_app.PDBFile.writeFile(topology, positions, out_file, keepIds=True)

    aa = get_atomarray(out_path, *args, **kwargs)

    return aa


def calc_aa_ca_rmsd(
    aa_fixed: AtomArray, aa_mobile: AtomArray, app_settings: AppSettings
) -> tuple[Optional[float], float]:
    aa_rmsd = None
    aa_aligned = None
    if not app_settings.different_topologies:
        aa_aligned, _ = b_structure.superimpose(fixed=aa_fixed, mobile=aa_mobile)
        aa_rmsd = float(b_structure.rmsd(aa_fixed, aa_aligned))

    ca_fixed = get_CAs(aa_fixed)
    ca_mobile = get_CAs(aa_mobile)

    ca_aligned, _ = b_structure.superimpose(fixed=ca_fixed, mobile=ca_mobile)
    ca_rmsd = float(b_structure.rmsd(ca_fixed, ca_aligned))

    return aa_rmsd, ca_rmsd


def safe_superimpose(
    aa_fixed: AtomArray, aa_mobile: AtomArray, app_settings: AppSettings
) -> AtomArray:
    if app_settings.different_topologies:
        aa_aligned, _, _, _ = b_structure.superimpose_homologs(
            fixed=aa_fixed,
            mobile=aa_mobile,
        )
    else:
        aa_aligned, _ = b_structure.superimpose(
            fixed=aa_fixed,
            mobile=aa_mobile,
        )
    return aa_aligned
