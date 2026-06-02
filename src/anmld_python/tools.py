from __future__ import annotations
from collections import Counter
from pathlib import Path
from typing import Optional, cast
import re

from biotite.structure.atoms import AtomArray, AtomArrayStack
from biotite.structure.io import save_structure
from loguru import logger
import biotite.structure as b_structure
import biotite.sequence as b_sequence
import biotite.structure.io.pdb as b_pdb
import biotite.structure.io.pdbx as b_pdbx
import fastpdb
import loguru
import networkx as nx
import numpy as np
import openmm.app as mm_app
import pdbfixer
import scipy.spatial as spatial

from anmld_python.settings import AppSettings, CropSettings


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
        platform=app_settings.openmm_settings._platform_obj,
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
    aa_fixed: AtomArray,
    aa_mobile: AtomArray,
    app_settings: AppSettings,
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


def check_backbone_continuity(aa: AtomArray, crop_settings: CropSettings):
    mask = np.zeros_like(aa, dtype=bool)

    is_continous = True
    for chain_id in b_structure.get_chains(aa):
        chain: b_structure.AtomArray = aa[aa.chain_id == chain_id]

        bb = chain[b_structure.filter_peptide_backbone(chain)]

        # needed to make sure tuples are not converted to a new dim
        bb_unique_res_ids = np.empty_like(bb, dtype="O")
        bb_unique_res_ids[:] = list(zip(bb.res_id, bb.ins_code))

        bb_bondlist = b_structure.connect_via_distances(bb)
        res_bondlist = bb_unique_res_ids[bb_bondlist.as_array()[:, :2]]

        G = nx.Graph()
        G.add_edges_from(res_bondlist)

        selected_res = set()
        for connected_comp in nx.connected_components(G):
            if len(connected_comp) < crop_settings.res_number_cutoff:
                is_continous = False
                continue
            selected_res.update(connected_comp)

        chain_unique_res_ids = list(zip(chain.res_id, chain.ins_code))
        mask[aa.chain_id == chain_id] = [
            x in selected_res for x in chain_unique_res_ids
        ]

    return is_continous, mask


def check_spatial_continuity(
    aa: AtomArray, crop_settings: CropSettings
) -> tuple[bool, AtomArray]:
    aa_atom_to_resid = b_structure.get_all_residue_positions(aa)
    ca_aa = aa[(aa.element == "C") & (aa.atom_name == "CA")]

    if b_structure.get_residue_count(aa) != b_structure.get_residue_count(ca_aa):
        raise ValueError("Structure has residues with missing carbon alpha atoms")

    ca_graph = nx.Graph(
        spatial.distance.squareform(
            spatial.distance.pdist(ca_aa.coord, metric="sqeuclidean")
        )  # type: ignore
        < crop_settings.connectedness_cutoff**2
    )
    con_comp = list(nx.connected_components(ca_graph))

    if len(con_comp) > 1:
        largest_con_comp = list(max(con_comp, key=len))
        largest_con_comp_atoms = np.isin(aa_atom_to_resid, largest_con_comp)
        return False, largest_con_comp_atoms
    return True, np.ones_like(aa, dtype=bool)


def crop_atomarrays(aa_A: AtomArray, aa_B: AtomArray, crop_settings: CropSettings):
    aa_A = aa_A[b_structure.filter_amino_acids(aa_A)]  # type:ignore
    aa_B = aa_B[b_structure.filter_amino_acids(aa_B)]  # type:ignore

    seqs_A = b_structure.to_sequence(aa_A)[0]
    seqs_B = b_structure.to_sequence(aa_B)[0]

    chains_A = b_structure.get_chains(aa_A)
    chains_B = b_structure.get_chains(aa_B)

    seq_sim_graph = nx.Graph()

    [seq_sim_graph.add_node(("A", cid, chains_A[cid])) for cid in range(len(seqs_A))]
    [seq_sim_graph.add_node(("B", cid, chains_B[cid])) for cid in range(len(seqs_B))]

    for cid_A, c_seq_A in enumerate(seqs_A):
        for cid_B, c_seq_B in enumerate(seqs_B):
            alignments = b_sequence.align.align_optimal(  # type: ignore
                b_sequence.ProteinSequence(c_seq_A),
                b_sequence.ProteinSequence(c_seq_B),
                b_sequence.align.SubstitutionMatrix.std_protein_matrix(),  # type: ignore
            )

            seq_id = None
            selected_al = None
            seq_ids = []
            for al in alignments:
                seq_id = b_sequence.align.get_pairwise_sequence_identity(al)[0, 1]  # type: ignore
                seq_ids.append(seq_id)
                if seq_id > crop_settings.seq_id_cutoff:
                    selected_al = al
                    break
            if selected_al is not None:
                assert seq_id is not None

                seq_sim_graph.add_edge(
                    ("A", cid_A, chains_A[cid_A]),
                    ("B", cid_B, chains_B[cid_B]),
                    alignment=selected_al,
                    seq_id=seq_id,
                )

    degrees = nx.degree(seq_sim_graph)

    if max(degrees, key=lambda x: x[1])[1] > 1:
        seq_matched_chains_A = set()
        seq_matched_chains_B = set()

        for edge in seq_sim_graph.edges():
            edge_cid_A = [v for v in edge if v[0] == "A"][0][1]
            edge_cid_B = [v for v in edge if v[0] == "B"][0][1]

            seq_matched_chains_A.add(chains_A[edge_cid_A])
            seq_matched_chains_B.add(chains_B[edge_cid_B])

        if len(seq_matched_chains_A) != len(seq_matched_chains_B):
            raise ValueError("Number of sequence matched chains must be equal")

        seq_matched_aa_A = aa_A[np.isin(aa_A.chain_id, list(seq_matched_chains_A))]
        seq_matched_aa_B = aa_B[np.isin(aa_B.chain_id, list(seq_matched_chains_B))]

        min_res_count = min(
            b_structure.get_residue_count(seq_matched_aa_A),
            b_structure.get_residue_count(seq_matched_aa_B),
        )

        seq_matched_aa_B, _, anchor_idx_A, anchor_idx_B = (
            b_structure.superimpose_homologs(
                fixed=seq_matched_aa_A,
                mobile=seq_matched_aa_B,
                min_anchors=min_res_count,
            )
        )

        anchors_chains_A = seq_matched_aa_A[anchor_idx_A].chain_id
        anchors_chains_B = seq_matched_aa_B[anchor_idx_B].chain_id

        anchor_counter = Counter(zip(anchors_chains_A, anchors_chains_B))

        edges = list(seq_sim_graph.edges())
        for edge in edges:
            edge_cid_A = [v for v in edge if v[0] == "A"][0][1]
            edge_cid_B = [v for v in edge if v[0] == "B"][0][1]

            edge_A = chains_A[edge_cid_A]
            edge_B = chains_B[edge_cid_B]

            matches_of_edge_A = {c_B: anchor_counter[(edge_A, c_B)] for c_B in chains_B}
            matches_of_edge_B = {c_A: anchor_counter[(c_A, edge_B)] for c_A in chains_A}

            best_match_of_edge_A = max(matches_of_edge_A, key=matches_of_edge_A.get)
            best_match_of_edge_B = max(matches_of_edge_B, key=matches_of_edge_B.get)

            if not (best_match_of_edge_A == edge_B and best_match_of_edge_B == edge_A):
                seq_sim_graph.remove_edge(*edge)

    selected_atoms_A = []
    selected_atoms_B = []
    for edge in seq_sim_graph.edges():
        edge_cid_A = [v for v in edge if v[0] == "A"][0][1]
        edge_cid_B = [v for v in edge if v[0] == "B"][0][1]

        edge_aa_A = aa_A[aa_A.chain_id == chains_A[edge_cid_A]]
        edge_aa_B = aa_B[aa_B.chain_id == chains_B[edge_cid_B]]

        alignment = nx.get_edge_attributes(seq_sim_graph, "alignment")[edge]
        non_gapped_mask = ~np.any(alignment.trace == -1, axis=1)

        aligned_res_idx_A = alignment.trace[non_gapped_mask, 0]
        aligned_res_idx_B = alignment.trace[non_gapped_mask, 1]

        edge_aa_atom_to_resid_A = b_structure.get_all_residue_positions(edge_aa_A)
        edge_aa_atom_to_resid_B = b_structure.get_all_residue_positions(edge_aa_B)

        edge_aligned_atoms_A = np.isin(edge_aa_atom_to_resid_A, aligned_res_idx_A)
        edge_aligned_atoms_B = np.isin(edge_aa_atom_to_resid_B, aligned_res_idx_B)

        selected_atoms_A.extend(edge_aa_A[edge_aligned_atoms_A])
        selected_atoms_B.extend(edge_aa_B[edge_aligned_atoms_B])

    sel_aa_A = b_structure.array(selected_atoms_A)
    sel_aa_B = b_structure.array(selected_atoms_B)

    is_bb_continous_A, bb_continuity_mask_A = check_backbone_continuity(
        aa=sel_aa_A, crop_settings=crop_settings
    )
    is_bb_continous_B, bb_continuity_mask_B = check_backbone_continuity(
        aa=sel_aa_B, crop_settings=crop_settings
    )

    if not is_bb_continous_A or not is_bb_continous_B:
        return crop_atomarrays(
            aa_A=sel_aa_A[bb_continuity_mask_A],
            aa_B=sel_aa_B[bb_continuity_mask_B],
            crop_settings=crop_settings,
        )

    # Check for spatially discontinued components
    is_spatially_connected_A, largest_cont_comp_mask_A = check_spatial_continuity(
        aa=sel_aa_A, crop_settings=crop_settings
    )
    is_spatially_connected_B, largest_cont_mask_B = check_spatial_continuity(
        aa=sel_aa_B, crop_settings=crop_settings
    )

    if not is_spatially_connected_A or not is_spatially_connected_B:
        return crop_atomarrays(
            aa_A=sel_aa_A[largest_cont_comp_mask_A],
            aa_B=sel_aa_B[largest_cont_mask_B],
            crop_settings=crop_settings,
        )

    return sel_aa_A, sel_aa_B


def crop_structures(
    init_path: Path,
    target_path: Path,
    cropped_init_path: Path,
    cropped_target_path: Path,
    crop_settings: CropSettings,
):
    init_aa = get_atomarray(init_path)
    target_aa = get_atomarray(target_path)

    cropped_init_aa, cropped_target_aa = crop_atomarrays(
        aa_A=init_aa, aa_B=target_aa, crop_settings=crop_settings
    )

    save_structure(file_path=cropped_init_path, array=cropped_init_aa)
    save_structure(file_path=cropped_target_path, array=cropped_target_aa)


def get_valid_filename(name):
    """
    Return the given string converted to a string that can be used for a clean
    filename. Remove leading and trailing spaces; convert other spaces to
    underscores; and remove anything that is not an alphanumeric, dash,
    underscore, or dot.
    >>> get_valid_filename("john's portrait in 2004.jpg")
    'johns_portrait_in_2004.jpg'

    https://github.com/django/django/blob/main/django/utils/text.py
    """
    s = str(name).strip().replace(" ", "_").replace(".", "_")
    s = re.sub(r"(?u)[^-\w.]", "", s)
    if s in {"", ".."}:
        raise ValueError("Could not derive file name from '%s'" % name)
    return s
