//! Sidechain atom domain types.
//!
//! [`SidechainAtoms`] owns extracted sidechain data from a protein entity,
//! including positions, topology, and per-atom metadata.

use glam::Vec3;

/// Data for a single sidechain atom.
#[derive(Debug, Clone)]
pub struct SidechainAtomData {
    /// 3D position of this atom.
    pub position: Vec3,
    /// Index of the parent residue.
    pub residue_idx: u32,
    /// PDB atom name (e.g. "CB", "CG").
    pub atom_name: String,
    /// Whether the parent residue is hydrophobic.
    pub is_hydrophobic: bool,
}

/// Owned sidechain atom data extracted from a protein entity.
///
/// This is the canonical "source of truth" for sidechain geometry.
/// GPU renderers borrow from this via a sidechain view.
#[derive(Debug, Clone)]
pub struct SidechainAtoms {
    /// Per-atom data.
    pub atoms: Vec<SidechainAtomData>,
    /// Intra-sidechain bonds as `(atom_idx_a, atom_idx_b)` pairs.
    pub bonds: Vec<(u32, u32)>,
    /// Backbone-to-sidechain bonds as `(CA_position, sidechain_atom_idx)`.
    pub backbone_bonds: Vec<(Vec3, u32)>,
}

impl SidechainAtoms {
    /// All sidechain atom positions.
    #[must_use]
    pub fn positions(&self) -> Vec<Vec3> {
        self.atoms.iter().map(|a| a.position).collect()
    }

    /// Per-atom hydrophobicity flags.
    #[must_use]
    pub fn hydrophobicity(&self) -> Vec<bool> {
        self.atoms.iter().map(|a| a.is_hydrophobic).collect()
    }

    /// Per-atom residue indices.
    #[must_use]
    pub fn residue_indices(&self) -> Vec<u32> {
        self.atoms.iter().map(|a| a.residue_idx).collect()
    }

    /// Per-atom PDB names.
    #[must_use]
    pub fn atom_names(&self) -> Vec<String> {
        self.atoms.iter().map(|a| a.atom_name.clone()).collect()
    }

    /// Number of sidechain atoms.
    #[must_use]
    pub fn len(&self) -> usize {
        self.atoms.len()
    }

    /// Whether there are no sidechain atoms.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.atoms.is_empty()
    }

    /// Create an empty `SidechainAtoms`.
    #[must_use]
    pub fn empty() -> Self {
        Self {
            atoms: Vec::new(),
            bonds: Vec::new(),
            backbone_bonds: Vec::new(),
        }
    }
}

impl Default for SidechainAtoms {
    fn default() -> Self {
        Self::empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sidechain_atoms_accessors() {
        let sc = SidechainAtoms {
            atoms: vec![
                SidechainAtomData {
                    position: Vec3::new(1.0, 2.0, 3.0),
                    residue_idx: 0,
                    atom_name: "CB".to_owned(),
                    is_hydrophobic: true,
                },
                SidechainAtomData {
                    position: Vec3::new(4.0, 5.0, 6.0),
                    residue_idx: 0,
                    atom_name: "CG".to_owned(),
                    is_hydrophobic: false,
                },
            ],
            bonds: vec![(0, 1)],
            backbone_bonds: vec![(Vec3::new(0.5, 1.0, 1.5), 0)],
        };

        assert_eq!(sc.len(), 2);
        assert_eq!(sc.positions().len(), 2);
        assert_eq!(sc.hydrophobicity(), vec![true, false]);
        assert_eq!(sc.residue_indices(), vec![0, 0]);
        assert_eq!(sc.atom_names(), vec!["CB".to_owned(), "CG".to_owned()]);
    }

    #[test]
    fn test_empty_sidechain_atoms() {
        let sc = SidechainAtoms::empty();
        assert!(sc.is_empty());
        assert_eq!(sc.len(), 0);
    }
}
