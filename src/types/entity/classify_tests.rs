#![allow(clippy::unwrap_used, clippy::cast_precision_loss, clippy::panic)]

use super::*;
use crate::types::coords::CoordsAtom;

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

#[test]
fn test_classify_protein() {
    assert_eq!(classify_residue("ALA"), MoleculeType::Protein);
    assert_eq!(classify_residue("GLY"), MoleculeType::Protein);
    assert_eq!(classify_residue("MSE"), MoleculeType::Protein);
}

#[test]
fn test_classify_nucleic() {
    assert_eq!(classify_residue("DA"), MoleculeType::DNA);
    assert_eq!(classify_residue("DT"), MoleculeType::DNA);
    assert_eq!(classify_residue("A"), MoleculeType::RNA);
    assert_eq!(classify_residue("U"), MoleculeType::RNA);
}

#[test]
fn test_classify_water_ion_ligand() {
    assert_eq!(classify_residue("HOH"), MoleculeType::Water);
    assert_eq!(classify_residue("WAT"), MoleculeType::Water);
    assert_eq!(classify_residue("ZN"), MoleculeType::Ion);
    assert_eq!(classify_residue("MG"), MoleculeType::Ion);
    // ATP and HEM are now cofactors, not ligands
    assert_eq!(classify_residue("ATP"), MoleculeType::Cofactor);
    assert_eq!(classify_residue("HEM"), MoleculeType::Cofactor);
    // Unknown small molecules remain ligands
    assert_eq!(classify_residue("UNL"), MoleculeType::Ligand);
}

#[test]
fn test_split_protein_only() {
    let coords = Coords {
        num_atoms: 6,
        atoms: (0..6).map(|i| make_atom(i as f32)).collect(),
        chain_ids: vec![b'A'; 6],
        res_names: vec![
            res_name("ALA"),
            res_name("ALA"),
            res_name("ALA"),
            res_name("GLY"),
            res_name("GLY"),
            res_name("GLY"),
        ],
        res_nums: vec![1, 1, 1, 2, 2, 2],
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

    let entities = split_into_entities(&coords);
    assert_eq!(entities.len(), 1);
    assert_eq!(entities[0].molecule_type, MoleculeType::Protein);
    assert_eq!(entities[0].atom_count(), 6);
    assert!(entities[0].as_polymer().is_some());
    let data = entities[0].as_polymer().unwrap();
    assert_eq!(data.chains.len(), 1);
    assert_eq!(data.chains[0].residues.len(), 2);
}

#[test]
fn test_split_mixed() {
    let coords = Coords {
        num_atoms: 5,
        atoms: (0..5).map(|i| make_atom(i as f32)).collect(),
        chain_ids: vec![b'A', b'A', b'A', b'B', b'C'],
        res_names: vec![
            res_name("ALA"),
            res_name("ALA"),
            res_name("HOH"),
            res_name("ATP"),
            res_name("HOH"),
        ],
        res_nums: vec![1, 1, 100, 1, 200],
        atom_names: vec![
            atom_name("N"),
            atom_name("CA"),
            atom_name("O"),
            atom_name("C1"),
            atom_name("O"),
        ],
        elements: vec![Element::Unknown; 5],
    };

    let entities = split_into_entities(&coords);
    // Protein(A), Water, Cofactor(ATP) = 3 entities
    assert_eq!(entities.len(), 3);

    let protein = entities
        .iter()
        .find(|e| e.molecule_type == MoleculeType::Protein)
        .unwrap();
    assert_eq!(protein.atom_count(), 2);

    let water = entities
        .iter()
        .find(|e| e.molecule_type == MoleculeType::Water)
        .unwrap();
    assert_eq!(water.atom_count(), 2);

    // ATP is now classified as Cofactor -> SmallMolecule
    let cofactor = entities
        .iter()
        .find(|e| e.molecule_type == MoleculeType::Cofactor)
        .unwrap();
    assert_eq!(cofactor.atom_count(), 1);
    assert!(matches!(cofactor.kind, EntityKind::SmallMolecule { .. }));
}

#[test]
fn test_merge_roundtrip() {
    let coords = Coords {
        num_atoms: 4,
        atoms: (0..4).map(|i| make_atom(i as f32)).collect(),
        chain_ids: vec![b'A', b'A', b'B', b'B'],
        res_names: vec![
            res_name("ALA"),
            res_name("ALA"),
            res_name("HOH"),
            res_name("HOH"),
        ],
        res_nums: vec![1, 1, 100, 101],
        atom_names: vec![
            atom_name("N"),
            atom_name("CA"),
            atom_name("O"),
            atom_name("O"),
        ],
        elements: vec![Element::Unknown; 4],
    };

    let entities = split_into_entities(&coords);
    let merged = merge_entities(&entities);
    assert_eq!(merged.num_atoms, coords.num_atoms);
}

#[test]
fn test_extract_by_type() {
    let coords = Coords {
        num_atoms: 3,
        atoms: (0..3).map(|i| make_atom(i as f32)).collect(),
        chain_ids: vec![b'A', b'A', b'A'],
        res_names: vec![res_name("ALA"), res_name("HOH"), res_name("ZN")],
        res_nums: vec![1, 100, 200],
        atom_names: vec![atom_name("CA"), atom_name("O"), atom_name("ZN")],
        elements: vec![Element::Unknown; 3],
    };

    let entities = split_into_entities(&coords);

    let protein = extract_by_type(&entities, MoleculeType::Protein);
    assert!(protein.is_some());
    assert_eq!(protein.unwrap().num_atoms, 1);

    let dna = extract_by_type(&entities, MoleculeType::DNA);
    assert!(dna.is_none());
}

#[test]
fn test_classify_cofactor() {
    assert_eq!(classify_residue("CLA"), MoleculeType::Cofactor);
    assert_eq!(classify_residue("HEM"), MoleculeType::Cofactor);
    assert_eq!(classify_residue("FAD"), MoleculeType::Cofactor);
    assert_eq!(classify_residue("NAD"), MoleculeType::Cofactor);
    assert_eq!(classify_residue("SF4"), MoleculeType::Cofactor);
    assert_eq!(classify_residue("BCR"), MoleculeType::Cofactor);
    assert_eq!(classify_residue("PL9"), MoleculeType::Cofactor);
}

#[test]
fn test_classify_solvent() {
    assert_eq!(classify_residue("GOL"), MoleculeType::Solvent);
    assert_eq!(classify_residue("EDO"), MoleculeType::Solvent);
    assert_eq!(classify_residue("SO4"), MoleculeType::Solvent);
    assert_eq!(classify_residue("PEG"), MoleculeType::Solvent);
    assert_eq!(classify_residue("MPD"), MoleculeType::Solvent);
    assert_eq!(classify_residue("DMS"), MoleculeType::Solvent);
}

#[test]
fn test_split_cofactor_grouping() {
    // Two CLA residues on different chains should become 2 separate
    // SmallMolecule entities
    let coords = Coords {
        num_atoms: 4,
        atoms: (0..4).map(|i| make_atom(i as f32)).collect(),
        chain_ids: vec![b'A', b'A', b'D', b'D'],
        res_names: vec![
            res_name("CLA"),
            res_name("CLA"),
            res_name("CLA"),
            res_name("CLA"),
        ],
        res_nums: vec![1, 1, 2, 2],
        atom_names: vec![
            atom_name("MG"),
            atom_name("NA"),
            atom_name("MG"),
            atom_name("NA"),
        ],
        elements: vec![Element::Unknown; 4],
    };

    let entities = split_into_entities(&coords);
    // Each (chain_id, res_num) pair produces its own SmallMolecule entity
    assert_eq!(entities.len(), 2);
    assert_eq!(entities[0].molecule_type, MoleculeType::Cofactor);
    assert_eq!(entities[1].molecule_type, MoleculeType::Cofactor);
    assert_eq!(entities[0].atom_count(), 2);
    assert_eq!(entities[1].atom_count(), 2);
    assert!(matches!(entities[0].kind, EntityKind::SmallMolecule { .. }));
}

#[test]
fn test_split_solvent_consolidated() {
    // GOL and SO4 are both Solvent, should be consolidated into one entity
    let coords = Coords {
        num_atoms: 3,
        atoms: (0..3).map(|i| make_atom(i as f32)).collect(),
        chain_ids: vec![b'A', b'B', b'C'],
        res_names: vec![res_name("GOL"), res_name("SO4"), res_name("GOL")],
        res_nums: vec![1, 2, 3],
        atom_names: vec![atom_name("O1"), atom_name("S"), atom_name("O1")],
        elements: vec![Element::Unknown; 3],
    };

    let entities = split_into_entities(&coords);
    assert_eq!(entities.len(), 1);
    assert_eq!(entities[0].molecule_type, MoleculeType::Solvent);
    assert_eq!(entities[0].atom_count(), 3);
    assert!(matches!(entities[0].kind, EntityKind::Bulk { .. }));
}

#[test]
fn test_polymer_structure() {
    let coords = Coords {
        num_atoms: 6,
        atoms: (0..6).map(|i| make_atom(i as f32)).collect(),
        chain_ids: vec![b'A', b'A', b'A', b'A', b'A', b'A'],
        res_names: vec![
            res_name("ALA"),
            res_name("ALA"),
            res_name("ALA"),
            res_name("GLY"),
            res_name("GLY"),
            res_name("GLY"),
        ],
        res_nums: vec![1, 1, 1, 2, 2, 2],
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

    let entities = split_into_entities(&coords);
    assert_eq!(entities.len(), 1);
    let data = entities[0].as_polymer().unwrap();
    assert_eq!(data.chains.len(), 1);
    assert_eq!(data.chains[0].chain_id, b'A');
    assert_eq!(data.chains[0].residues.len(), 2);
    assert_eq!(data.chains[0].residues[0].name, res_name("ALA"));
    assert_eq!(data.chains[0].residues[0].number, 1);
    assert_eq!(data.chains[0].residues[0].atom_range, 0..3);
    assert_eq!(data.chains[0].residues[1].name, res_name("GLY"));
    assert_eq!(data.chains[0].residues[1].number, 2);
    assert_eq!(data.chains[0].residues[1].atom_range, 3..6);
}

#[test]
fn test_small_molecule_no_chain_residue() {
    let coords = Coords {
        num_atoms: 1,
        atoms: vec![make_atom(1.0)],
        chain_ids: vec![b'A'],
        res_names: vec![res_name("ZN")],
        res_nums: vec![300],
        atom_names: vec![atom_name("ZN")],
        elements: vec![Element::Zn],
    };

    let entities = split_into_entities(&coords);
    assert_eq!(entities.len(), 1);
    assert_eq!(entities[0].molecule_type, MoleculeType::Ion);
    match &entities[0].kind {
        EntityKind::SmallMolecule {
            atoms,
            residue_name,
            ..
        } => {
            assert_eq!(atoms.len(), 1);
            assert_eq!(*residue_name, res_name("ZN"));
        }
        _ => panic!("expected SmallMolecule"),
    }
}

#[test]
fn test_to_coords_roundtrip() {
    let coords = Coords {
        num_atoms: 6,
        atoms: (0..6).map(|i| make_atom(i as f32)).collect(),
        chain_ids: vec![b'A', b'A', b'A', b'A', b'A', b'A'],
        res_names: vec![
            res_name("ALA"),
            res_name("ALA"),
            res_name("ALA"),
            res_name("GLY"),
            res_name("GLY"),
            res_name("GLY"),
        ],
        res_nums: vec![1, 1, 1, 2, 2, 2],
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

    let entities = split_into_entities(&coords);
    let recovered = entities[0].to_coords();
    assert_eq!(recovered.num_atoms, 6);
    assert_eq!(recovered.chain_ids, vec![b'A'; 6]);
    assert_eq!(recovered.res_nums, vec![1, 1, 1, 2, 2, 2]);
}

#[test]
fn test_split_modified_amino_acid_merges_into_protein() {
    // SEP (phosphoserine) on a protein chain should merge into the protein
    // entity, not create a separate SmallMolecule.
    let coords = Coords {
        num_atoms: 9,
        atoms: (0..9).map(|i| make_atom(i as f32)).collect(),
        chain_ids: vec![b'A'; 9],
        res_names: vec![
            res_name("ALA"),
            res_name("ALA"),
            res_name("ALA"),
            res_name("SEP"),
            res_name("SEP"),
            res_name("SEP"),
            res_name("GLY"),
            res_name("GLY"),
            res_name("GLY"),
        ],
        res_nums: vec![1, 1, 1, 2, 2, 2, 3, 3, 3],
        atom_names: vec![
            atom_name("N"),
            atom_name("CA"),
            atom_name("C"),
            atom_name("N"),
            atom_name("CA"),
            atom_name("C"),
            atom_name("N"),
            atom_name("CA"),
            atom_name("C"),
        ],
        elements: vec![Element::Unknown; 9],
    };

    let entities = split_into_entities(&coords);
    // Should be 1 protein entity with all 9 atoms, not 1 protein + 1 ligand
    assert_eq!(entities.len(), 1);
    assert_eq!(entities[0].molecule_type, MoleculeType::Protein);
    assert_eq!(entities[0].atom_count(), 9);
    let data = entities[0].as_polymer().unwrap();
    assert_eq!(data.chains[0].residues.len(), 3);
}
