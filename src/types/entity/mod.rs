//! Molecule-type classification and per-entity coordinate splitting.
//!
//! Provides:
//! - `MoleculeType` — classification of residues into protein, DNA, RNA,
//!   ligand, ion, water
//! - `AtomSet` — core atom data (positions + chemistry, no PDB artifacts)
//! - `PolymerData` / `PolymerChain` / `Residue` — structured polymer hierarchy
//! - `EntityKind` — discriminated union of polymer, small molecule, and bulk
//!   entity data
//! - `MoleculeEntity` — a single entity with its own `EntityKind`
//! - `classify_residue()` — classify a residue name into a `MoleculeType`
//! - `split_into_entities()` — split flat `Coords` into per-entity groups
//! - `merge_entities()` — recombine entities back into flat `Coords`

mod classify;
mod extract;
mod polymer;

pub use classify::{
    classify_residue, coords_to_entity_kind, extract_by_type, merge_entities,
    split_into_entities,
};
pub use extract::NucleotideRing;
use glam::Vec3;
pub use polymer::{PolymerChain, PolymerData, Residue};

use super::coords::{Coords, CoordsAtom, Element};

/// Classification of molecule types found in structural biology files.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MoleculeType {
    /// Amino acid polymer.
    Protein,
    /// Deoxyribonucleic acid polymer.
    DNA,
    /// Ribonucleic acid polymer.
    RNA,
    /// Non-polymer small molecule (drug, substrate, etc.).
    Ligand,
    /// Single-atom metal or halide ion.
    Ion,
    /// Water molecule.
    Water,
    /// Lipid or detergent molecule.
    Lipid,
    /// Enzyme cofactor (heme, NAD, FAD, Fe-S cluster, etc.).
    Cofactor,
    /// Crystallization solvent or buffer artifact.
    Solvent,
}

impl MoleculeType {
    /// Convert to a wire byte for ASSEM01 binary format.
    #[must_use]
    pub fn to_wire_byte(self) -> u8 {
        match self {
            MoleculeType::Protein => 0,
            MoleculeType::DNA => 1,
            MoleculeType::RNA => 2,
            MoleculeType::Ligand => 3,
            MoleculeType::Ion => 4,
            MoleculeType::Water => 5,
            MoleculeType::Lipid => 6,
            MoleculeType::Cofactor => 7,
            MoleculeType::Solvent => 8,
        }
    }

    /// Parse from a wire byte in ASSEM01 binary format.
    #[must_use]
    pub fn from_wire_byte(b: u8) -> Option<Self> {
        match b {
            0 => Some(MoleculeType::Protein),
            1 => Some(MoleculeType::DNA),
            2 => Some(MoleculeType::RNA),
            3 => Some(MoleculeType::Ligand),
            4 => Some(MoleculeType::Ion),
            5 => Some(MoleculeType::Water),
            6 => Some(MoleculeType::Lipid),
            7 => Some(MoleculeType::Cofactor),
            8 => Some(MoleculeType::Solvent),
            _ => None,
        }
    }
}

/// Axis-aligned bounding box (AABB).
#[derive(Debug, Clone, Copy)]
pub struct Aabb {
    /// Minimum corner of the bounding box.
    pub min: Vec3,
    /// Maximum corner of the bounding box.
    pub max: Vec3,
}

impl Aabb {
    /// Geometric center of the box.
    #[must_use]
    pub fn center(&self) -> Vec3 {
        (self.min + self.max) * 0.5
    }

    /// Size along each axis (max - min).
    #[must_use]
    pub fn extents(&self) -> Vec3 {
        self.max - self.min
    }

    /// Half-diagonal length (bounding sphere radius from center).
    #[must_use]
    pub fn radius(&self) -> f32 {
        self.extents().length() * 0.5
    }

    /// Merge two AABBs into one that contains both.
    #[must_use]
    pub fn union(&self, other: &Aabb) -> Aabb {
        Aabb {
            min: self.min.min(other.min),
            max: self.max.max(other.max),
        }
    }

    /// Build AABB from positions. Returns `None` if the slice is empty.
    #[must_use]
    pub fn from_positions(positions: &[Vec3]) -> Option<Aabb> {
        let first = *positions.first()?;
        let mut min = first;
        let mut max = first;
        for &p in &positions[1..] {
            min = min.min(p);
            max = max.max(p);
        }
        Some(Aabb { min, max })
    }

    /// Build unified AABB from multiple AABBs.
    #[must_use]
    pub fn from_aabbs(aabbs: &[Aabb]) -> Option<Aabb> {
        aabbs.iter().copied().reduce(|a, b| a.union(&b))
    }
}

// ---------------------------------------------------------------------------
// Core atom data
// ---------------------------------------------------------------------------

/// Core atom data — chemistry only, no format artifacts.
#[derive(Debug, Clone)]
pub struct AtomSet {
    /// Per-atom coordinate and occupancy data.
    pub atoms: Vec<CoordsAtom>,
    /// PDB-style 4-character atom names.
    pub atom_names: Vec<[u8; 4]>,
    /// Chemical element for each atom.
    pub elements: Vec<Element>,
}

impl AtomSet {
    /// Number of atoms in this set.
    #[must_use]
    pub fn len(&self) -> usize {
        self.atoms.len()
    }

    /// Returns `true` if the set contains no atoms.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.atoms.is_empty()
    }

    /// Collect all atom positions as `Vec3`.
    #[must_use]
    pub fn positions(&self) -> Vec<Vec3> {
        self.atoms
            .iter()
            .map(|a| Vec3::new(a.x, a.y, a.z))
            .collect()
    }

    /// Convert to a minimal `Coords` (for bond inference compatibility).
    /// Chain IDs and residue metadata are set to dummy values.
    #[must_use]
    pub fn to_coords_minimal(&self) -> Coords {
        let n = self.atoms.len();
        Coords {
            num_atoms: n,
            atoms: self.atoms.clone(),
            chain_ids: vec![b' '; n],
            res_names: vec![[b' '; 3]; n],
            res_nums: vec![0; n],
            atom_names: self.atom_names.clone(),
            elements: self.elements.clone(),
        }
    }
}

// ---------------------------------------------------------------------------
// EntityKind
// ---------------------------------------------------------------------------

/// Discriminated union of entity data variants.
#[derive(Debug, Clone)]
pub enum EntityKind {
    /// Polymer chain (protein/DNA/RNA) — structured chain/residue hierarchy.
    Polymer(PolymerData),
    /// Single non-polymer molecule (ligand, cofactor, lipid, ion).
    SmallMolecule {
        /// Atom data for this molecule.
        atoms: AtomSet,
        /// 3-character residue code (e.g. b"ATP").
        residue_name: [u8; 3],
        /// Human-readable name for display (e.g. "Chlorophyll A").
        display_name: String,
    },
    /// Bulk group (water, solvent) — many identical small molecules.
    Bulk {
        /// Atom data for all molecules in this group.
        atoms: AtomSet,
        /// 3-character residue code (e.g. b"HOH").
        residue_name: [u8; 3],
        /// Number of individual molecules in this group.
        molecule_count: usize,
    },
}

// ---------------------------------------------------------------------------
// MoleculeEntity
// ---------------------------------------------------------------------------

/// A single entity: one logical molecule (a protein chain, a ligand, waters,
/// etc.)
#[derive(Debug, Clone)]
pub struct MoleculeEntity {
    /// Unique sequential identifier assigned during entity splitting.
    pub entity_id: u32,
    /// Classification of this entity's molecule type.
    pub molecule_type: MoleculeType,
    /// Structured data payload (polymer, small molecule, or bulk).
    pub kind: EntityKind,
}

impl MoleculeEntity {
    // -- Convenience accessors --

    /// Reference to the underlying `AtomSet`.
    #[must_use]
    pub fn atom_set(&self) -> &AtomSet {
        match &self.kind {
            EntityKind::Polymer(data) => &data.atoms,
            EntityKind::SmallMolecule { atoms, .. }
            | EntityKind::Bulk { atoms, .. } => atoms,
        }
    }

    /// All atom positions as Vec3.
    #[must_use]
    pub fn positions(&self) -> Vec<Vec3> {
        self.atom_set().positions()
    }

    /// Number of atoms in this entity.
    #[must_use]
    pub fn atom_count(&self) -> usize {
        self.atom_set().len()
    }

    /// Slice of all atoms.
    #[must_use]
    pub fn atoms(&self) -> &[CoordsAtom] {
        &self.atom_set().atoms
    }

    /// Slice of all elements.
    #[must_use]
    pub fn elements(&self) -> &[Element] {
        &self.atom_set().elements
    }

    /// Slice of all atom names.
    #[must_use]
    pub fn atom_names(&self) -> &[[u8; 4]] {
        &self.atom_set().atom_names
    }

    /// If this entity is a polymer, return the structured data.
    #[must_use]
    pub fn as_polymer(&self) -> Option<&PolymerData> {
        match &self.kind {
            EntityKind::Polymer(data) => Some(data),
            _ => None,
        }
    }

    /// Convert to a flat `Coords` for serialization or interop.
    #[must_use]
    #[allow(
        clippy::too_many_lines,
        reason = "match arms for each EntityKind variant are inherently \
                  verbose"
    )]
    pub fn to_coords(&self) -> Coords {
        match &self.kind {
            EntityKind::Polymer(data) => {
                let n = data.atoms.len();
                let mut chain_ids = Vec::with_capacity(n);
                let mut res_names = Vec::with_capacity(n);
                let mut res_nums = Vec::with_capacity(n);
                #[allow(
                    clippy::excessive_nesting,
                    reason = "triple-nested loop over chains/residues/atoms \
                              is natural for polymer data"
                )]
                for chain in &data.chains {
                    for residue in &chain.residues {
                        for _ in residue.atom_range.clone() {
                            chain_ids.push(chain.chain_id);
                            res_names.push(residue.name);
                            res_nums.push(residue.number);
                        }
                    }
                }
                Coords {
                    num_atoms: n,
                    atoms: data.atoms.atoms.clone(),
                    chain_ids,
                    res_names,
                    res_nums,
                    atom_names: data.atoms.atom_names.clone(),
                    elements: data.atoms.elements.clone(),
                }
            }
            EntityKind::SmallMolecule {
                atoms,
                residue_name,
                ..
            } => {
                let n = atoms.len();
                Coords {
                    num_atoms: n,
                    atoms: atoms.atoms.clone(),
                    chain_ids: vec![b' '; n],
                    res_names: vec![*residue_name; n],
                    res_nums: vec![1; n],
                    atom_names: atoms.atom_names.clone(),
                    elements: atoms.elements.clone(),
                }
            }
            EntityKind::Bulk {
                atoms,
                residue_name,
                ..
            } => {
                let n = atoms.len();
                #[allow(
                    clippy::cast_possible_truncation,
                    clippy::cast_possible_wrap,
                    reason = "atom count fits in i32 for valid structures"
                )]
                let res_nums_vec = (1..=n as i32).collect();
                Coords {
                    num_atoms: n,
                    atoms: atoms.atoms.clone(),
                    chain_ids: vec![b' '; n],
                    res_names: vec![*residue_name; n],
                    res_nums: res_nums_vec,
                    atom_names: atoms.atom_names.clone(),
                    elements: atoms.elements.clone(),
                }
            }
        }
    }

    /// Compute the axis-aligned bounding box for this entity's atoms.
    #[must_use]
    pub fn aabb(&self) -> Option<Aabb> {
        Aabb::from_positions(&self.positions())
    }

    /// Human-readable label (e.g. "Protein Chain A", "Ligand (ATP)", "Zn²⁺
    /// Ion").
    #[must_use]
    #[allow(
        clippy::too_many_lines,
        reason = "match arms per molecule type are straightforward"
    )]
    pub fn label(&self) -> String {
        match self.molecule_type {
            MoleculeType::Protein => {
                if let EntityKind::Polymer(data) = &self.kind {
                    let chains: Vec<u8> =
                        data.chains.iter().map(|c| c.chain_id).collect();
                    if chains.len() == 1 {
                        format!("Protein Chain {}", chains[0] as char)
                    } else {
                        let chain_str: String =
                            chains.iter().map(|&c| c as char).collect();
                        format!("Protein Chains {chain_str}")
                    }
                } else {
                    "Protein".to_owned()
                }
            }
            MoleculeType::DNA => {
                if let EntityKind::Polymer(data) = &self.kind {
                    let chains: Vec<u8> =
                        data.chains.iter().map(|c| c.chain_id).collect();
                    if chains.len() == 1 {
                        format!("DNA Chain {}", chains[0] as char)
                    } else {
                        "DNA".to_owned()
                    }
                } else {
                    "DNA".to_owned()
                }
            }
            MoleculeType::RNA => {
                if let EntityKind::Polymer(data) = &self.kind {
                    let chains: Vec<u8> =
                        data.chains.iter().map(|c| c.chain_id).collect();
                    if chains.len() == 1 {
                        format!("RNA Chain {}", chains[0] as char)
                    } else {
                        "RNA".to_owned()
                    }
                } else {
                    "RNA".to_owned()
                }
            }
            MoleculeType::Ligand => {
                if let EntityKind::SmallMolecule { display_name, .. } =
                    &self.kind
                {
                    format!("Ligand ({display_name})")
                } else {
                    "Ligand".to_owned()
                }
            }
            MoleculeType::Ion => {
                if let EntityKind::SmallMolecule { display_name, .. } =
                    &self.kind
                {
                    format!("{display_name} Ion")
                } else {
                    "Ion".to_owned()
                }
            }
            MoleculeType::Water => {
                if let EntityKind::Bulk { molecule_count, .. } = &self.kind {
                    format!("Water ({molecule_count} molecules)")
                } else {
                    "Water".to_owned()
                }
            }
            MoleculeType::Lipid => {
                if let EntityKind::SmallMolecule { display_name, .. } =
                    &self.kind
                {
                    format!("Lipid ({display_name})")
                } else {
                    format!("Lipid ({} molecules)", self.residue_count())
                }
            }
            MoleculeType::Cofactor => {
                if let EntityKind::SmallMolecule { display_name, .. } =
                    &self.kind
                {
                    display_name.clone()
                } else {
                    "Cofactor".to_owned()
                }
            }
            MoleculeType::Solvent => {
                if let EntityKind::Bulk { molecule_count, .. } = &self.kind {
                    format!("Solvent ({molecule_count} molecules)")
                } else {
                    "Solvent".to_owned()
                }
            }
        }
    }

    /// Whether this entity type participates in tab-cycling focus.
    /// Protein: no (focused at group level). Water, Ion: no (ambient).
    /// Ligand, DNA, RNA: yes.
    #[must_use]
    pub fn is_focusable(&self) -> bool {
        !matches!(
            self.molecule_type,
            MoleculeType::Water | MoleculeType::Ion | MoleculeType::Solvent
        )
    }

    /// Number of residues (for polymer/nucleic) or molecules (for small
    /// mol/ion/water).
    #[must_use]
    pub fn residue_count(&self) -> usize {
        match &self.kind {
            EntityKind::Polymer(data) => {
                data.chains.iter().map(|c| c.residues.len()).sum()
            }
            EntityKind::SmallMolecule { .. } => 1,
            EntityKind::Bulk { molecule_count, .. } => *molecule_count,
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn test_aabb_from_positions() {
        let positions = vec![
            Vec3::new(1.0, 2.0, 3.0),
            Vec3::new(-1.0, 5.0, 0.0),
            Vec3::new(3.0, -2.0, 7.0),
        ];
        let aabb = Aabb::from_positions(&positions).unwrap();
        assert_eq!(aabb.min, Vec3::new(-1.0, -2.0, 0.0));
        assert_eq!(aabb.max, Vec3::new(3.0, 5.0, 7.0));
    }

    #[test]
    fn test_aabb_empty() {
        assert!(Aabb::from_positions(&[]).is_none());
    }

    #[test]
    fn test_aabb_union() {
        let a = Aabb {
            min: Vec3::new(0.0, 0.0, 0.0),
            max: Vec3::new(1.0, 1.0, 1.0),
        };
        let b = Aabb {
            min: Vec3::new(-1.0, 2.0, -3.0),
            max: Vec3::new(0.5, 4.0, 0.5),
        };
        let merged = a.union(&b);
        assert_eq!(merged.min, Vec3::new(-1.0, 0.0, -3.0));
        assert_eq!(merged.max, Vec3::new(1.0, 4.0, 1.0));
    }

    #[test]
    fn test_aabb_from_aabbs() {
        let aabbs = vec![
            Aabb {
                min: Vec3::ZERO,
                max: Vec3::ONE,
            },
            Aabb {
                min: Vec3::splat(2.0),
                max: Vec3::splat(3.0),
            },
        ];
        let merged = Aabb::from_aabbs(&aabbs).unwrap();
        assert_eq!(merged.min, Vec3::ZERO);
        assert_eq!(merged.max, Vec3::splat(3.0));
        assert!(Aabb::from_aabbs(&[]).is_none());
    }

    #[test]
    fn test_aabb_center_extents_radius() {
        let aabb = Aabb {
            min: Vec3::ZERO,
            max: Vec3::new(4.0, 6.0, 8.0),
        };
        assert_eq!(aabb.center(), Vec3::new(2.0, 3.0, 4.0));
        assert_eq!(aabb.extents(), Vec3::new(4.0, 6.0, 8.0));
        let expected_radius = Vec3::new(4.0, 6.0, 8.0).length() * 0.5;
        assert!((aabb.radius() - expected_radius).abs() < 1e-6);
    }
}
