//! Coordinate transformation utilities.
//!
//! Provides functions for extracting, filtering, and aligning coordinates:
//! - Backbone/CA extraction
//! - Atom filtering
//! - Kabsch alignment

mod alignment;
mod extract;
mod interpolate;

pub use alignment::{
    align_coords_bytes, align_to_reference, kabsch_alignment,
    kabsch_alignment_with_scale, transform_coords, transform_coords_with_scale,
};
pub use extract::{
    build_ca_position_map, centroid, extract_backbone_chains,
    extract_ca_from_chains, extract_ca_positions, get_atom_by_name,
    get_atom_position, get_backbone_atoms_from_chains, get_ca_for_residue,
    get_ca_position_from_chains, get_closest_atom_for_residue,
    get_closest_atom_with_name, get_closest_backbone_atom, set_atom_position,
    NamedResidueAtomSearch, ResidueAtomSearch,
};
pub use interpolate::{interpolate_coords, interpolate_coords_collapse};

use crate::types::coords::{Coords, Element};

/// Standard amino acid residue names
pub const PROTEIN_RESIDUES: &[&str] = &[
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR",
    "VAL", // Non-standard but protein-like
    "MSE", "SEC", "PYL",
];

/// Filter COORDS to only protein residues (remove water, ligands, etc.).
#[must_use]
pub fn protein_only(coords: &Coords) -> Coords {
    filter_residues(coords, |res_name| {
        let name_str = std::str::from_utf8(res_name).unwrap_or("").trim();
        PROTEIN_RESIDUES.contains(&name_str)
    })
}

/// Filter COORDS to only heavy atoms (exclude hydrogens).
#[must_use]
pub fn heavy_atoms_only(coords: &Coords) -> Coords {
    filter_atoms(coords, |name| {
        let name_str = std::str::from_utf8(name).unwrap_or("").trim();
        !name_str.starts_with('H')
            && !name_str.starts_with("1H")
            && !name_str.starts_with("2H")
            && !name_str.starts_with("3H")
    })
}

/// Filter COORDS to only backbone atoms (N, CA, C, O).
#[must_use]
pub fn backbone_only(coords: &Coords) -> Coords {
    filter_atoms(coords, |name| {
        let name_str = std::str::from_utf8(name).unwrap_or("").trim();
        matches!(name_str, "N" | "CA" | "C" | "O")
    })
}

/// Filter atoms by predicate on atom name.
#[must_use]
pub fn filter_atoms(
    coords: &Coords,
    predicate: impl Fn(&[u8; 4]) -> bool,
) -> Coords {
    let mut atoms = Vec::new();
    let mut chain_ids = Vec::new();
    let mut res_names = Vec::new();
    let mut res_nums = Vec::new();
    let mut atom_names = Vec::new();
    let mut elements = Vec::new();

    for i in 0..coords.num_atoms {
        if predicate(&coords.atom_names[i]) {
            atoms.push(coords.atoms[i].clone());
            chain_ids.push(coords.chain_ids[i]);
            res_names.push(coords.res_names[i]);
            res_nums.push(coords.res_nums[i]);
            atom_names.push(coords.atom_names[i]);
            elements.push(
                coords.elements.get(i).copied().unwrap_or(Element::Unknown),
            );
        }
    }

    Coords {
        num_atoms: atoms.len(),
        atoms,
        chain_ids,
        res_names,
        res_nums,
        atom_names,
        elements,
    }
}

/// Filter atoms by predicate on residue name.
#[must_use]
pub fn filter_residues(
    coords: &Coords,
    predicate: impl Fn(&[u8; 3]) -> bool,
) -> Coords {
    let mut atoms = Vec::new();
    let mut chain_ids = Vec::new();
    let mut res_names = Vec::new();
    let mut res_nums = Vec::new();
    let mut atom_names = Vec::new();
    let mut elements = Vec::new();

    for i in 0..coords.num_atoms {
        if predicate(&coords.res_names[i]) {
            atoms.push(coords.atoms[i].clone());
            chain_ids.push(coords.chain_ids[i]);
            res_names.push(coords.res_names[i]);
            res_nums.push(coords.res_nums[i]);
            atom_names.push(coords.atom_names[i]);
            elements.push(
                coords.elements.get(i).copied().unwrap_or(Element::Unknown),
            );
        }
    }

    Coords {
        num_atoms: atoms.len(),
        atoms,
        chain_ids,
        res_names,
        res_nums,
        atom_names,
        elements,
    }
}
