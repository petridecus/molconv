//! Assembly helpers: free functions for working with `Vec<MoleculeEntity>`.
//!
//! Previously this module housed the `Assembly` struct which wrapped
//! `Vec<MoleculeEntity>` + cached `Coords`.  The struct was a "smart cache"
//! that added indirection without real benefit — `merge_entities` already
//! derives `Coords` from entities.  Now every operation is a free function
//! that takes `&[MoleculeEntity]` or `&mut Vec<MoleculeEntity>`.

use glam::Vec3;

use super::coords::{serialize_assembly, CoordsError};
use super::entity::{
    merge_entities, split_into_entities, MoleculeEntity, MoleculeType,
};
use crate::ops::transform::{extract_ca_positions, protein_only};
use crate::types::coords::Coords;

// ============================================================================
// Derivation helpers (replace Assembly accessor methods)
// ============================================================================

/// Protein-only Coords derived from entities.
#[must_use]
pub fn protein_coords(entities: &[MoleculeEntity]) -> Coords {
    let merged = merge_entities(entities);
    protein_only(&merged)
}

/// All entities serialized to ASSEM01 bytes (includes molecule type metadata).
///
/// # Errors
///
/// Returns `CoordsError` if serialization fails.
pub fn assembly_bytes(
    entities: &[MoleculeEntity],
) -> Result<Vec<u8>, CoordsError> {
    serialize_assembly(entities)
}

/// CA positions from the protein portion.
#[must_use]
pub fn ca_positions(entities: &[MoleculeEntity]) -> Vec<Vec3> {
    extract_ca_positions(&protein_coords(entities))
}

/// Number of protein residues (CA count).
#[must_use]
pub fn residue_count(entities: &[MoleculeEntity]) -> usize {
    ca_positions(entities).len()
}

// ============================================================================
// Mutation helpers (replace Assembly mutation methods)
// ============================================================================

/// Replace protein entity coords (keeps non-protein entities).
/// Splits the incoming combined protein coords by chain ID so each
/// entity only receives its own chain's atoms, avoiding duplication.
pub fn update_protein_entities(
    entities: &mut Vec<MoleculeEntity>,
    protein: &Coords,
) {
    // Re-split the incoming protein coords into per-chain entities
    let new_protein = split_into_entities(protein);

    // Remove old protein entities, keep non-protein
    entities.retain(|e| e.molecule_type != MoleculeType::Protein);

    // Prepend new protein entities (proteins first, then non-proteins)
    let mut updated: Vec<MoleculeEntity> = new_protein
        .into_iter()
        .filter(|e| e.molecule_type == MoleculeType::Protein)
        .collect();
    updated.append(entities);

    // Re-assign sequential entity IDs
    for (i, entity) in updated.iter_mut().enumerate() {
        #[allow(clippy::cast_possible_truncation)] // entity count fits in u32
        {
            entity.entity_id = i as u32;
        }
    }

    *entities = updated;
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::cast_precision_loss,
    clippy::too_many_lines,
    clippy::float_cmp
)]
mod tests {
    use super::*;
    use crate::types::coords::{CoordsAtom, Element};
    use crate::types::entity::MoleculeType;

    fn make_atom(x: f32) -> CoordsAtom {
        CoordsAtom {
            x,
            y: 0.0,
            z: 0.0,
            occupancy: 1.0,
            b_factor: 0.0,
        }
    }

    fn res_name(s: &str) -> [u8; 3] {
        let mut name = [b' '; 3];
        for (i, b) in s.bytes().take(3).enumerate() {
            name[i] = b;
        }
        name
    }

    fn atom_name(s: &str) -> [u8; 4] {
        let mut name = [b' '; 4];
        for (i, b) in s.bytes().take(4).enumerate() {
            name[i] = b;
        }
        name
    }

    /// Regression: `update_protein_entities` on a multi-chain set must NOT
    /// duplicate atoms.
    #[test]
    fn test_update_protein_entities_no_duplication() {
        let coords = Coords {
            num_atoms: 7,
            atoms: (0..7).map(|i| make_atom(i as f32)).collect(),
            chain_ids: vec![b'A', b'A', b'A', b'B', b'B', b'B', b'C'],
            res_names: vec![
                res_name("ALA"),
                res_name("ALA"),
                res_name("ALA"),
                res_name("GLY"),
                res_name("GLY"),
                res_name("GLY"),
                res_name("HOH"),
            ],
            res_nums: vec![1, 1, 1, 1, 1, 1, 100],
            atom_names: vec![
                atom_name("N"),
                atom_name("CA"),
                atom_name("C"),
                atom_name("N"),
                atom_name("CA"),
                atom_name("C"),
                atom_name("O"),
            ],
            elements: vec![Element::Unknown; 7],
        };

        let mut entities = split_into_entities(&coords);
        assert_eq!(entities.len(), 3); // chain A, chain B, water
        let total: usize =
            entities.iter().map(MoleculeEntity::atom_count).sum();
        assert_eq!(total, 7);

        // Simulate an update: combined protein coords (both chains, 6 atoms)
        let updated_protein = Coords {
            num_atoms: 6,
            atoms: (10..16).map(|i| make_atom(i as f32)).collect(),
            chain_ids: vec![b'A', b'A', b'A', b'B', b'B', b'B'],
            res_names: vec![
                res_name("ALA"),
                res_name("ALA"),
                res_name("ALA"),
                res_name("GLY"),
                res_name("GLY"),
                res_name("GLY"),
            ],
            res_nums: vec![1, 1, 1, 1, 1, 1],
            atom_names: vec![
                atom_name("N"),
                atom_name("CA"),
                atom_name("C"),
                atom_name("N"),
                atom_name("CA"),
                atom_name("C"),
            ],
            elements: vec![Element::Unknown; 6],
        };

        update_protein_entities(&mut entities, &updated_protein);

        // Must still have 7 total atoms (6 protein + 1 water), NOT 13
        let total: usize =
            entities.iter().map(MoleculeEntity::atom_count).sum();
        assert_eq!(total, 7);
        assert_eq!(entities.len(), 3); // chain A, chain B, water

        // Protein entities should have 3 atoms each, not 6
        let protein_entities: Vec<_> = entities
            .iter()
            .filter(|e| e.molecule_type == MoleculeType::Protein)
            .collect();
        assert_eq!(protein_entities.len(), 2);
        assert_eq!(protein_entities[0].atom_count(), 3);
        assert_eq!(protein_entities[1].atom_count(), 3);

        // Verify protein coords were actually updated (x values should be 10+)
        let prot = protein_coords(&entities);
        assert_eq!(prot.num_atoms, 6);
        assert!(prot.atoms[0].x >= 10.0);

        // Second update should not grow either
        let updated_protein2 = Coords {
            num_atoms: 6,
            atoms: (20..26).map(|i| make_atom(i as f32)).collect(),
            chain_ids: vec![b'A', b'A', b'A', b'B', b'B', b'B'],
            res_names: vec![
                res_name("ALA"),
                res_name("ALA"),
                res_name("ALA"),
                res_name("GLY"),
                res_name("GLY"),
                res_name("GLY"),
            ],
            res_nums: vec![1, 1, 1, 1, 1, 1],
            atom_names: vec![
                atom_name("N"),
                atom_name("CA"),
                atom_name("C"),
                atom_name("N"),
                atom_name("CA"),
                atom_name("C"),
            ],
            elements: vec![Element::Unknown; 6],
        };

        update_protein_entities(&mut entities, &updated_protein2);
        let total: usize =
            entities.iter().map(MoleculeEntity::atom_count).sum();
        assert_eq!(total, 7); // Still 7, not growing
    }

    #[test]
    fn test_assembly_bytes_roundtrip_mixed() {
        use crate::types::coords::deserialize_assembly;

        let coords = Coords {
            num_atoms: 5,
            atoms: vec![
                make_atom(1.0),
                make_atom(2.0),
                make_atom(3.0),
                make_atom(10.0),
                make_atom(20.0),
            ],
            chain_ids: vec![b'A', b'A', b'A', b'B', b'C'],
            res_names: vec![
                res_name("ALA"),
                res_name("ALA"),
                res_name("ALA"),
                res_name("ATP"),
                res_name("ZN"),
            ],
            res_nums: vec![1, 1, 1, 1, 1],
            atom_names: vec![
                atom_name("N"),
                atom_name("CA"),
                atom_name("C"),
                atom_name("C1"),
                atom_name("ZN"),
            ],
            elements: vec![
                Element::N,
                Element::C,
                Element::C,
                Element::C,
                Element::Zn,
            ],
        };

        let entities = split_into_entities(&coords);
        assert!(entities.len() >= 3);

        let bytes = assembly_bytes(&entities).unwrap();
        assert_eq!(&bytes[0..8], b"ASSEM01\0");

        let roundtripped = deserialize_assembly(&bytes).unwrap();
        assert_eq!(roundtripped.len(), entities.len());

        for (orig, rt) in entities.iter().zip(roundtripped.iter()) {
            assert_eq!(orig.molecule_type, rt.molecule_type);
            assert_eq!(orig.atom_count(), rt.atom_count());
        }

        let orig_protein: Vec<_> = entities
            .iter()
            .filter(|e| e.molecule_type == MoleculeType::Protein)
            .collect();
        let rt_protein: Vec<_> = roundtripped
            .iter()
            .filter(|e| e.molecule_type == MoleculeType::Protein)
            .collect();
        assert_eq!(orig_protein.len(), rt_protein.len());
        for (o, r) in orig_protein.iter().zip(rt_protein.iter()) {
            let oc = o.to_coords();
            let rc = r.to_coords();
            for i in 0..o.atom_count() {
                assert!((oc.atoms[i].x - rc.atoms[i].x).abs() < 1e-6);
                assert_eq!(oc.chain_ids[i], rc.chain_ids[i]);
                assert_eq!(oc.res_names[i], rc.res_names[i]);
                assert_eq!(oc.res_nums[i], rc.res_nums[i]);
                assert_eq!(oc.atom_names[i], rc.atom_names[i]);
            }
        }
    }

    #[test]
    fn test_assembly_bytes_protein_only() {
        use crate::types::coords::deserialize_assembly;

        let coords = Coords {
            num_atoms: 3,
            atoms: vec![make_atom(1.0), make_atom(2.0), make_atom(3.0)],
            chain_ids: vec![b'A'; 3],
            res_names: vec![res_name("ALA"); 3],
            res_nums: vec![1; 3],
            atom_names: vec![atom_name("N"), atom_name("CA"), atom_name("C")],
            elements: vec![Element::N, Element::C, Element::C],
        };

        let entities = split_into_entities(&coords);
        let bytes = assembly_bytes(&entities).unwrap();
        let roundtripped = deserialize_assembly(&bytes).unwrap();
        assert_eq!(roundtripped.len(), 1);
        assert_eq!(roundtripped[0].molecule_type, MoleculeType::Protein);
        assert_eq!(roundtripped[0].atom_count(), 3);
    }

    #[test]
    fn test_assembly_bytes_single_atom_ion() {
        use crate::types::coords::deserialize_assembly;

        let coords = Coords {
            num_atoms: 1,
            atoms: vec![make_atom(5.5)],
            chain_ids: vec![b'X'],
            res_names: vec![res_name("ZN")],
            res_nums: vec![99],
            atom_names: vec![atom_name("ZN")],
            elements: vec![Element::Zn],
        };

        let entities = split_into_entities(&coords);
        let bytes = assembly_bytes(&entities).unwrap();
        let roundtripped = deserialize_assembly(&bytes).unwrap();
        assert_eq!(roundtripped.len(), 1);
        assert_eq!(roundtripped[0].molecule_type, MoleculeType::Ion);
        assert_eq!(roundtripped[0].atom_count(), 1);
        assert!((roundtripped[0].atoms()[0].x - 5.5).abs() < 1e-6);
    }

    #[test]
    fn test_assembly_bytes_empty_entities() {
        use crate::types::coords::{deserialize_assembly, serialize_assembly};

        let entities: Vec<MoleculeEntity> = Vec::new();
        let bytes = serialize_assembly(&entities).unwrap();
        let roundtripped = deserialize_assembly(&bytes).unwrap();
        assert!(roundtripped.is_empty());
    }

    #[test]
    fn test_assembly_byte_layout() {
        use crate::types::coords::serialize_assembly;

        let coords = Coords {
            num_atoms: 1,
            atoms: vec![CoordsAtom {
                x: 1.0,
                y: 2.0,
                z: 3.0,
                occupancy: 1.0,
                b_factor: 0.0,
            }],
            chain_ids: vec![b'A'],
            res_names: vec![res_name("ALA")],
            res_nums: vec![1],
            atom_names: vec![atom_name("CA")],
            elements: vec![Element::C],
        };
        let kind = crate::types::entity::coords_to_entity_kind(
            MoleculeType::Protein,
            &coords,
        );
        let entities = vec![MoleculeEntity {
            entity_id: 0,
            molecule_type: MoleculeType::Protein,
            kind,
        }];

        let bytes = serialize_assembly(&entities).unwrap();

        assert_eq!(&bytes[0..8], b"ASSEM01\0");
        assert_eq!(u32::from_be_bytes(bytes[8..12].try_into().unwrap()), 1);
        assert_eq!(bytes[12], 0); // Protein
        assert_eq!(u32::from_be_bytes(bytes[13..17].try_into().unwrap()), 1);
        assert_eq!(f32::from_be_bytes(bytes[17..21].try_into().unwrap()), 1.0);
        assert_eq!(f32::from_be_bytes(bytes[21..25].try_into().unwrap()), 2.0);
        assert_eq!(f32::from_be_bytes(bytes[25..29].try_into().unwrap()), 3.0);
        assert_eq!(bytes[29], b'A');
        assert_eq!(bytes.len(), 43);
    }
}
