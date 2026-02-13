from __future__ import annotations
from pathlib import Path
from typing import Optional, cast

from biotite.structure.atoms import AtomArray, AtomArrayStack
from biotite.structure.io import save_structure
from loguru import logger
import biotite.structure as b_structure
import biotite.structure.io.pdbx as b_pdbx
import biotite.structure.io.pdb as b_pdb
import fastpdb
import loguru
import numpy as np
import openmm.app as mm_app
import pdbfixer

from anmld_python.settings import AppSettings


class LDError(Exception):
    pass


class NonConnectedStructureError(Exception):
    pass


def get_atomarray(
    file_path: Path,
    structure_index: int = 0,
    extra_fields: Optional[list | str] = None,
    *args,
    **kwargs,
) -> AtomArray:
    if not extra_fields:
        extra_fields = []

    match file_path.suffix:
        case ".pdb":
            # fastpdb might panic while loading bonds
            # https://github.com/biotite-dev/fastpdb/pull/25
            try:
                structure_file = fastpdb.PDBFile.read(file_path)
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
                structure_file = b_pdb.PDBFile.read(file_path)
                atomarray = structure_file.get_structure(
                    extra_fields=extra_fields,
                    model=structure_index + 1,
                    *args,
                    **kwargs,
                )
        case ".cif":
            structure_file = b_pdbx.CIFFile.read(file_path)
            atomarray = b_pdbx.get_structure(
                structure_file,
                model=structure_index + 1,
                extra_fields=extra_fields,
                *args,
                **kwargs,
            )
        case _:
            emsg = f"Given structure file {file_path} is not supported."
            raise ValueError(emsg)

    return cast(AtomArray, atomarray)


def get_CAs(aa: AtomArray) -> AtomArray | None:
    cas = None
    if not (cas := aa[(aa.atom_name == "CA") & (aa.element == "C")]):
        cas = aa[(aa.atom_name == "CA")]
    return cas


def _filter_atomarray(
    aa: AtomArray,
    sel_chains: Optional[list[str]],
    sanitization_logger: loguru.Logger,
) -> AtomArray:
    sanitization_logger.trace("Filtering atomarray")

    sanitization_logger.info("Filtering chains")
    chains = b_structure.get_chains(aa)
    if sel_chains:
        if not np.all(np.isin(sel_chains, chains)):
            emsg = f"The given {sel_chains=} does not exists"
            sanitization_logger.critical(emsg)
            raise ValueError(emsg) from None
        aa = aa[np.isin(aa.chain_id, sel_chains)]

    sanitization_logger.info("Filtering aminoacids")
    aa = aa[b_structure.filter_amino_acids(aa)]

    # TODO: make removing or using current H an option
    sanitization_logger.info("Removing hydrogens")
    aa = aa[aa.element != "H"]

    return aa


def _run_pdbfixer(
    structure_path: Path,
    sanitization_logger: loguru.Logger,
    app_settings: AppSettings,
) -> tuple[mm_app.Topology, list]:
    sanitization_logger.trace("Running PDBFixer")

    fixer = pdbfixer.PDBFixer(
        filename=str(structure_path.absolute()),
        platform=app_settings.openmm_settings.platform_obj,
    )

    sanitization_logger.info(f"Replacing nonstandard residues")
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()

    sanitization_logger.info("Adding missing heavy atoms")
    fixer.missingResidues = {}
    fixer.findMissingAtoms()
    passed = False
    for i in range(app_settings.sanitization_max_retry):
        try:
            fixer.addMissingAtoms()
            passed = True
            break
        except Exception as e:
            sanitization_logger.warning(
                "PDBFixer could not add missing heavy atoms"
                " due to clashes in the structure."
                f" Retrying ({i + 1}/{app_settings.sanitization_max_retry})",
                err=e,
            )
    if not passed:
        sanitization_logger.critical(
            "PDBFixer could not add missing heavy atoms",
            " due to clashes in the structure.",
        )
        raise Exception from None
    sanitization_logger.info(f"Added missing heavy atoms")

    return fixer.topology, fixer.positions


def sanitize_structure(
    in_path: Path,
    filtered_path: Path,
    sanitized_path: Path,
    app_settings: AppSettings,
    sel_chains: Optional[list[str]] = None,
    *args,
    **kwargs,
) -> AtomArray:
    sanitization_logger = logger.bind(in_path=in_path, out_path=sanitized_path)
    sanitization_logger.trace("Sanitizing structure")

    sanitization_logger.debug("Loading atomarray")
    aa = get_atomarray(file_path=in_path, *args, **kwargs)
    if isinstance(aa, AtomArrayStack):
        emsg = "The structure file include more than one models"
        sanitization_logger.critical(emsg)
        raise Exception(emsg)

    aa = _filter_atomarray(
        aa=aa,
        sel_chains=sel_chains,
        sanitization_logger=sanitization_logger,
    )

    save_structure(file_path=filtered_path, array=aa)

    if app_settings.LD_method == "OpenMM":
        topology, positions = _run_pdbfixer(
            structure_path=filtered_path,
            sanitization_logger=sanitization_logger,
            app_settings=app_settings,
        )

        import anmld_python.ld.openmm

        topology, positions = anmld_python.ld.openmm.add_H(
            topology=topology,
            positions=positions,
            sanitization_logger=sanitization_logger,
            app_settings=app_settings,
        )
        with open(sanitized_path, "w") as out_file:
            mm_app.PDBFile.writeFile(topology, positions, out_file, keepIds=True)

    elif app_settings.LD_method == "AMBER":
        import anmld_python.ld.amber

        anmld_python.ld.amber.run_pdb4amber(
            in_path=filtered_path,
            out_path=sanitized_path,
            stdout_stderr_redirection_path=sanitized_path.with_name(
                sanitized_path.stem + "_pdb4amber_stdout.txt"
            ),
            logger=sanitization_logger,
            app_settings=app_settings,
        )

    aa = get_atomarray(file_path=sanitized_path, *args, **kwargs)

    return aa


def calc_aa_ca_rmsd(
    aa_fixed: AtomArray, aa_mobile: AtomArray, app_settings: AppSettings
) -> tuple[Optional[float], float]:
    aa_rmsd = None
    aa_aligned = None
    if not app_settings._different_topologies:
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
    if app_settings._different_topologies:
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
