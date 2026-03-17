//! Residue classification, entity splitting, merging, and extraction by type.

use std::collections::{BTreeMap, HashSet};

use super::{
    AtomSet, EntityKind, MoleculeEntity, MoleculeType, PolymerChain,
    PolymerData, Residue,
};
use crate::ops::transform::PROTEIN_RESIDUES;
use crate::types::coords::{Coords, Element};

/// Standard DNA residue names (mmCIF convention + common alias THY).
const DNA_RESIDUES: &[&str] = &["DA", "DC", "DG", "DT", "DU", "DI", "THY"];

/// Standard RNA residue names.
const RNA_RESIDUES: &[&str] = &[
    "A", "C", "G", "U", "ADE", "CYT", "GUA", "URA", "I", "RAD", "RCY", "RGU",
];

/// Water residue names.
const WATER_RESIDUES: &[&str] = &[
    "HOH", "WAT", "H2O", "DOD",
    // MD simulation water models (GROMACS, AMBER, CHARMM, etc.)
    "SOL", "TIP", "TP3", "TIP3", "T3P", "SPC", "TP4", "TIP4", "T4P", "TP5",
    "TIP5",
];

/// Known ion residue names. These are single-atom residues with well-known
/// names.
const ION_RESIDUES: &[&str] = &[
    "ZN", "MG", "NA", "CL", "FE", "MN", "CO", "NI", "CU", "K", "CA", "BR", "I",
    "F", "LI", "CD", "SR", "BA", "CS", "RB", "PB", "HG", "PT", "AU", "AG",
];

/// Known lipid residue 3-char truncated names.
const LIPID_RESIDUES: &[&str] = &[
    // Phosphatidylcholines (DPPC, POPC, DOPC, DMPC, DSPC, DLPC)
    "DPP", "POP", "DOP", "DMP", "DSP", "DLP",
    // Phosphatidylethanolamines (DPPE, POPE, DOPE)
    "PPE", "DPE", // Phosphatidylglycerols (DPPG, POPG, DOPG)
    "PPG", "DPG", // Phosphatidylserines (DPPS, POPS, DOPS)
    "PPS", "DPS", // Cholesterol variants
    "CHO", "CHL", // Sphingomyelin, ceramide
    "SPH", "CER", // CHARMM-GUI lipid residue names (full 3-letter)
    "PAL", "OLE", "STE", "MYR", "LAU",
    // PDB crystallographic lipid codes (thylakoid/membrane lipids)
    "LHG", // dipalmitoyl phosphatidylglycerol
    "LMG", // monogalactosyl diglyceride (MGDG)
    "DGD", // digalactosyl diacyl glycerol (DGDG)
    "SQD", // sulfoquinovosyl diacylglycerol (SQDG)
    // PDB detergent codes (amphipathic, treated as lipid-like)
    "LMT", // dodecyl-beta-D-maltoside
    "HTG", // heptyl 1-thiohexopyranoside
];

/// Known cofactor residue names (exact match, checked before lipid truncation).
const COFACTOR_RESIDUES: &[&str] = &[
    // Porphyrins / chlorins
    "HEM", "HEC", "HEA", "HEB", "CLA", "CHL", "PHO", "BCR",
    "BCB", // Quinones
    "PL9", "PLQ", "UQ1", "UQ2", "MQ7", // Nucleotide cofactors
    "NAD", "NAP", "NAI", "NDP", "FAD", "FMN", "ATP", "ADP", "AMP", "ANP",
    "GTP", "GDP", "GMP", "GNP", // Other
    "SAM", "SAH", "COA", "ACO", "PLP", "PMP", "TPP", "TDP", "BTN", "BIO",
    "H4B", "BH4", // Fe-S clusters
    "SF4", "FES", "F3S",
];

/// Known solvent / crystallization artifact residue names (exact match).
const SOLVENT_RESIDUES: &[&str] = &[
    // Polyols / PEGs
    "GOL", "EDO", "PEG", "1PE", "P6G", "PG4", "PGE", // Salts / buffers
    "SO4", "SUL", "PO4", "ACT", "ACE", "CIT", "FMT", // Buffers
    "TRS", "MES", "EPE", "IMD", // Cryoprotectants
    "MPD", "DMS", "BME", "IPA", "EOH",
];

/// Human-readable display name for a cofactor residue code.
fn cofactor_display_name(res_name: &str) -> &str {
    match res_name {
        "CLA" => "Chlorophyll A",
        "CHL" => "Chlorophyll B",
        "BCR" => "Beta-Carotene",
        "BCB" => "Beta-Carotene B",
        "HEM" => "Heme",
        "HEC" => "Heme C",
        "HEA" => "Heme A",
        "HEB" => "Heme B",
        "PHO" => "Pheophytin",
        "PL9" | "PLQ" => "Plastoquinone",
        "UQ1" | "UQ2" => "Ubiquinone",
        "MQ7" => "Menaquinone",
        "NAD" | "NAP" | "NAI" | "NDP" => "NAD",
        "SAM" | "SAH" => "SAM/SAH",
        "COA" | "ACO" => "Coenzyme A",
        "PLP" | "PMP" => "PLP",
        "TPP" | "TDP" => "Thiamine PP",
        "BTN" | "BIO" => "Biotin",
        "H4B" | "BH4" => "Tetrahydrobiopterin",
        "SF4" => "[4Fe-4S] Cluster",
        "FES" => "[2Fe-2S] Cluster",
        "F3S" => "[3Fe-4S] Cluster",
        _ => res_name,
    }
}

/// Display name for a small molecule, dispatching by molecule type.
pub(super) fn small_molecule_display_name(
    mol_type: MoleculeType,
    res_name: &str,
) -> String {
    match mol_type {
        MoleculeType::Cofactor => cofactor_display_name(res_name).to_owned(),
        _ => res_name.to_owned(),
    }
}

/// Classify a residue name into a `MoleculeType`.
///
/// The name should be trimmed of whitespace before calling.
#[must_use]
pub fn classify_residue(name: &str) -> MoleculeType {
    if PROTEIN_RESIDUES.contains(&name) {
        return MoleculeType::Protein;
    }
    if WATER_RESIDUES.contains(&name) {
        return MoleculeType::Water;
    }
    if DNA_RESIDUES.contains(&name) {
        return MoleculeType::DNA;
    }
    // RNA single-letter names overlap with element symbols, but in the context
    // of residue names (label_comp_id), single letters are nucleotides.
    if RNA_RESIDUES.contains(&name) {
        return MoleculeType::RNA;
    }
    if ION_RESIDUES.contains(&name) {
        return MoleculeType::Ion;
    }
    // Cofactor: exact match, checked before lipid truncation
    if COFACTOR_RESIDUES.contains(&name) {
        return MoleculeType::Cofactor;
    }
    // Solvent / crystallization artifacts: exact match
    if SOLVENT_RESIDUES.contains(&name) {
        return MoleculeType::Solvent;
    }
    // Check lipid residues: exact match on truncated 3-char names
    let truncated = if name.len() > 3 { &name[..3] } else { name };
    if LIPID_RESIDUES.contains(&truncated) {
        return MoleculeType::Lipid;
    }
    MoleculeType::Ligand
}

/// Check whether a set of atom indices (belonging to one residue) contains
/// protein backbone atoms N, CA, and C — the hallmark of an amino acid.
fn residue_has_backbone(indices: &[usize], coords: &Coords) -> bool {
    let mut has_n = false;
    let mut has_ca = false;
    let mut has_c = false;
    for &idx in indices {
        let name = &coords.atom_names[idx];
        match name {
            [b' ', b'N', b' ', b' '] | [b'N', b' ', b' ', b' '] => has_n = true,
            [b' ', b'C', b'A', b' '] | [b'C', b'A', b' ', b' '] => {
                has_ca = true;
            }
            [b' ', b'C', b' ', b' '] | [b'C', b' ', b' ', b' '] => has_c = true,
            _ => {}
        }
    }
    has_n && has_ca && has_c
}

// ---------------------------------------------------------------------------
// Entity splitting / merging
// ---------------------------------------------------------------------------

/// Key for grouping atoms into entities.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum EntityKey {
    /// A polymeric chain (protein, DNA, RNA) on a specific chain.
    Chain(u8, MoleculeTypeOrd),
    /// All water molecules consolidated into one entity.
    Water,
    /// All solvent molecules consolidated into one entity.
    Solvent,
    /// A single non-polymer molecule, keyed by (chain_id, res_num, type).
    SmallMolecule(u8, i32, MoleculeTypeOrd),
}

/// Wrapper for MoleculeType that implements Ord (for BTreeMap keys).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct MoleculeTypeOrd(MoleculeType);

impl PartialOrd for MoleculeTypeOrd {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for MoleculeTypeOrd {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        (self.0 as u8).cmp(&(other.0 as u8))
    }
}

/// Split a flat `Coords` into per-entity `MoleculeEntity` groups.
///
/// Grouping rules:
/// - Protein/DNA/RNA: grouped by (chain_id, molecule_type) — each polymer chain
///   is one entity
/// - Water: all water residues are consolidated into a single entity
/// - Solvent: all solvent residues are consolidated into a single entity
/// - Ligand/Ion/Cofactor/Lipid: each unique (chain_id, res_num) is its own
///   entity
///
/// Entity IDs are assigned sequentially starting from 0.
#[must_use]
#[allow(
    clippy::excessive_nesting,
    reason = "grouping logic with nested match arms is natural"
)]
#[allow(
    clippy::too_many_lines,
    reason = "entity splitting with modified-residue merging is a single \
              logical operation"
)]
pub fn split_into_entities(coords: &Coords) -> Vec<MoleculeEntity> {
    // Group atom indices by entity key (BTreeMap for deterministic ordering)
    let mut groups: BTreeMap<EntityKey, Vec<usize>> = BTreeMap::new();

    for i in 0..coords.num_atoms {
        let res_name = std::str::from_utf8(&coords.res_names[i])
            .unwrap_or("")
            .trim();
        let mol_type = classify_residue(res_name);
        let chain_id = coords.chain_ids[i];

        let key = match mol_type {
            MoleculeType::Water => EntityKey::Water,
            MoleculeType::Solvent => EntityKey::Solvent,
            MoleculeType::Protein | MoleculeType::DNA | MoleculeType::RNA => {
                EntityKey::Chain(chain_id, MoleculeTypeOrd(mol_type))
            }
            MoleculeType::Ligand
            | MoleculeType::Ion
            | MoleculeType::Cofactor
            | MoleculeType::Lipid => {
                let res_num = coords.res_nums[i];
                EntityKey::SmallMolecule(
                    chain_id,
                    res_num,
                    MoleculeTypeOrd(mol_type),
                )
            }
        };

        groups.entry(key).or_default().push(i);
    }

    // Merge modified amino acids back into their protein chain.
    // A SmallMolecule group that has backbone atoms (N, CA, C) and whose chain
    // already has a protein entity is a modified residue, not a ligand.
    let protein_chains: Vec<u8> = groups
        .keys()
        .filter_map(|k| match k {
            EntityKey::Chain(cid, mt) if mt.0 == MoleculeType::Protein => {
                Some(*cid)
            }
            _ => None,
        })
        .collect();

    let merge_keys: Vec<EntityKey> = groups
        .iter()
        .filter_map(|(key, indices)| {
            if let EntityKey::SmallMolecule(chain_id, _, _) = key {
                if protein_chains.contains(chain_id)
                    && residue_has_backbone(indices, coords)
                {
                    return Some(key.clone());
                }
            }
            None
        })
        .collect();

    for key in merge_keys {
        if let EntityKey::SmallMolecule(chain_id, _, _) = &key {
            let chain_key = EntityKey::Chain(
                *chain_id,
                MoleculeTypeOrd(MoleculeType::Protein),
            );
            if let Some(indices) = groups.remove(&key) {
                groups.entry(chain_key).or_default().extend(indices);
            }
        }
    }

    // Convert groups to entities
    groups
        .into_iter()
        .enumerate()
        .map(|(entity_id, (key, indices))| {
            let mol_type = match &key {
                EntityKey::Chain(_, mt)
                | EntityKey::SmallMolecule(_, _, mt) => mt.0,
                EntityKey::Water => MoleculeType::Water,
                EntityKey::Solvent => MoleculeType::Solvent,
            };

            let kind = match mol_type {
                MoleculeType::Protein
                | MoleculeType::DNA
                | MoleculeType::RNA => build_polymer_kind(&indices, coords),
                MoleculeType::Water | MoleculeType::Solvent => {
                    build_bulk_kind(&indices, coords)
                }
                _ => build_small_molecule_kind(mol_type, &indices, coords),
            };

            #[allow(
                clippy::cast_possible_truncation,
                reason = "entity count fits in u32"
            )]
            MoleculeEntity {
                entity_id: entity_id as u32,
                molecule_type: mol_type,
                kind,
            }
        })
        .collect()
}

/// Build `EntityKind::Polymer` from a set of atom indices belonging to one
/// polymer chain.
fn build_polymer_kind(indices: &[usize], coords: &Coords) -> EntityKind {
    // Group atoms by (chain_id, res_num) preserving insertion order within each
    let mut chain_residue_map: BTreeMap<u8, BTreeMap<i32, Vec<usize>>> =
        BTreeMap::new();
    for &idx in indices {
        chain_residue_map
            .entry(coords.chain_ids[idx])
            .or_default()
            .entry(coords.res_nums[idx])
            .or_default()
            .push(idx);
    }

    let mut atom_set_atoms = Vec::with_capacity(indices.len());
    let mut atom_set_names = Vec::with_capacity(indices.len());
    let mut atom_set_elements = Vec::with_capacity(indices.len());
    let mut chains = Vec::new();

    for (&chain_id, residues) in &chain_residue_map {
        let mut chain_residues = Vec::new();
        for (&res_num, atom_indices) in residues {
            let start = atom_set_atoms.len();
            let res_name = coords.res_names[atom_indices[0]];
            for &idx in atom_indices {
                atom_set_atoms.push(coords.atoms[idx].clone());
                atom_set_names.push(coords.atom_names[idx]);
                atom_set_elements.push(
                    coords
                        .elements
                        .get(idx)
                        .copied()
                        .unwrap_or(Element::Unknown),
                );
            }
            let end = atom_set_atoms.len();
            chain_residues.push(Residue {
                name: res_name,
                number: res_num,
                atom_range: start..end,
            });
        }
        chains.push(PolymerChain {
            chain_id,
            residues: chain_residues,
        });
    }

    EntityKind::Polymer(PolymerData {
        atoms: AtomSet {
            atoms: atom_set_atoms,
            atom_names: atom_set_names,
            elements: atom_set_elements,
        },
        chains,
    })
}

/// Build `EntityKind::SmallMolecule` from a set of atom indices.
fn build_small_molecule_kind(
    mol_type: MoleculeType,
    indices: &[usize],
    coords: &Coords,
) -> EntityKind {
    let mut atoms = Vec::with_capacity(indices.len());
    let mut atom_names = Vec::with_capacity(indices.len());
    let mut elements = Vec::with_capacity(indices.len());

    for &idx in indices {
        atoms.push(coords.atoms[idx].clone());
        atom_names.push(coords.atom_names[idx]);
        elements.push(
            coords
                .elements
                .get(idx)
                .copied()
                .unwrap_or(Element::Unknown),
        );
    }

    let residue_name = coords.res_names[indices[0]];
    let rn_str = std::str::from_utf8(&residue_name).unwrap_or("???").trim();
    let display_name = small_molecule_display_name(mol_type, rn_str);

    EntityKind::SmallMolecule {
        atoms: AtomSet {
            atoms,
            atom_names,
            elements,
        },
        residue_name,
        display_name,
    }
}

/// Build `EntityKind::Bulk` from a set of atom indices (water/solvent).
fn build_bulk_kind(indices: &[usize], coords: &Coords) -> EntityKind {
    let mut atoms = Vec::with_capacity(indices.len());
    let mut atom_names = Vec::with_capacity(indices.len());
    let mut elements = Vec::with_capacity(indices.len());

    // Count unique (chain_id, res_num) pairs for molecule_count
    let mut seen = HashSet::new();
    for &idx in indices {
        atoms.push(coords.atoms[idx].clone());
        atom_names.push(coords.atom_names[idx]);
        elements.push(
            coords
                .elements
                .get(idx)
                .copied()
                .unwrap_or(Element::Unknown),
        );
        let _ = seen.insert((coords.chain_ids[idx], coords.res_nums[idx]));
    }

    let residue_name = coords.res_names[indices[0]];

    EntityKind::Bulk {
        atoms: AtomSet {
            atoms,
            atom_names,
            elements,
        },
        residue_name,
        molecule_count: seen.len(),
    }
}

/// Convert flat `Coords` + molecule type into an `EntityKind`.
///
/// Use this when you have raw Coords from deserialization and need to
/// construct the appropriate EntityKind based on molecule type.
#[must_use]
pub fn coords_to_entity_kind(
    mol_type: MoleculeType,
    coords: &Coords,
) -> EntityKind {
    match mol_type {
        MoleculeType::Protein | MoleculeType::DNA | MoleculeType::RNA => {
            let indices: Vec<usize> = (0..coords.num_atoms).collect();
            build_polymer_kind(&indices, coords)
        }
        MoleculeType::Water | MoleculeType::Solvent => {
            let indices: Vec<usize> = (0..coords.num_atoms).collect();
            build_bulk_kind(&indices, coords)
        }
        _ => {
            let indices: Vec<usize> = (0..coords.num_atoms).collect();
            build_small_molecule_kind(mol_type, &indices, coords)
        }
    }
}

/// Merge multiple entities back into a single flat `Coords`.
///
/// Entities are concatenated in order. Useful for recombining before
/// sending to backends that expect a single coordinate set.
#[must_use]
pub fn merge_entities(entities: &[MoleculeEntity]) -> Coords {
    let total_atoms: usize =
        entities.iter().map(MoleculeEntity::atom_count).sum();

    let mut atoms = Vec::with_capacity(total_atoms);
    let mut chain_ids = Vec::with_capacity(total_atoms);
    let mut res_names = Vec::with_capacity(total_atoms);
    let mut res_nums = Vec::with_capacity(total_atoms);
    let mut atom_names = Vec::with_capacity(total_atoms);
    let mut elements = Vec::with_capacity(total_atoms);

    for entity in entities {
        let c = entity.to_coords();
        atoms.extend(c.atoms);
        chain_ids.extend(c.chain_ids);
        res_names.extend(c.res_names);
        res_nums.extend(c.res_nums);
        atom_names.extend(c.atom_names);
        elements.extend(c.elements);
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

/// Extract a merged `Coords` containing only entities of the given molecule
/// type.
#[must_use]
pub fn extract_by_type(
    entities: &[MoleculeEntity],
    mol_type: MoleculeType,
) -> Option<Coords> {
    let matching: Vec<&MoleculeEntity> = entities
        .iter()
        .filter(|e| e.molecule_type == mol_type)
        .collect();

    if matching.is_empty() {
        return None;
    }

    let total_atoms: usize = matching.iter().map(|e| e.atom_count()).sum();
    let mut atoms = Vec::with_capacity(total_atoms);
    let mut chain_ids = Vec::with_capacity(total_atoms);
    let mut res_names = Vec::with_capacity(total_atoms);
    let mut res_nums = Vec::with_capacity(total_atoms);
    let mut atom_names = Vec::with_capacity(total_atoms);
    let mut elements = Vec::with_capacity(total_atoms);

    for entity in matching {
        let c = entity.to_coords();
        atoms.extend(c.atoms);
        chain_ids.extend(c.chain_ids);
        res_names.extend(c.res_names);
        res_nums.extend(c.res_nums);
        atom_names.extend(c.atom_names);
        elements.extend(c.elements);
    }

    Some(Coords {
        num_atoms: atoms.len(),
        atoms,
        chain_ids,
        res_names,
        res_nums,
        atom_names,
        elements,
    })
}

#[cfg(test)]
#[path = "classify_tests.rs"]
mod tests;
