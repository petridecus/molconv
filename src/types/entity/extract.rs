//! Backbone, sidechain, and nucleotide ring extraction from entities.

use std::collections::{BTreeMap, HashMap};

use glam::Vec3;

use super::{EntityKind, MoleculeEntity, MoleculeType, PolymerData};
use crate::render::backbone::{BackboneChain, ProteinBackbone};
use crate::render::sidechain::{SidechainAtomData, SidechainAtoms};

/// Pre-extracted ring geometry for a single nucleotide base.
#[derive(Debug, Clone)]
pub struct NucleotideRing {
    /// Hexagonal ring atom positions in order: N1, C2, N3, C4, C5, C6
    pub hex_ring: Vec<Vec3>,
    /// Pentagonal ring for purines: C4, C5, N7, C8, N9 (empty for pyrimidines)
    pub pent_ring: Vec<Vec3>,
    /// NDB color for this base
    pub color: [f32; 3],
    /// C1' sugar carbon position (for anchoring stem to backbone spline).
    pub c1_prime: Option<Vec3>,
}

const HEX_RING_ATOMS: &[&str] = &["N1", "C2", "N3", "C4", "C5", "C6"];
const PENT_RING_ATOMS: &[&str] = &["C4", "C5", "N7", "C8", "N9"];

fn ndb_base_color(res_name: &str) -> Option<[f32; 3]> {
    match res_name {
        "DA" | "A" | "ADE" | "RAD" => Some([0.85, 0.20, 0.20]),
        "DG" | "G" | "GUA" | "RGU" => Some([0.20, 0.80, 0.20]),
        "DC" | "C" | "CYT" | "RCY" => Some([0.90, 0.90, 0.20]),
        "DT" | "THY" => Some([0.20, 0.20, 0.85]),
        "DU" | "U" | "URA" => Some([0.20, 0.85, 0.85]),
        _ => None,
    }
}

fn is_purine(res_name: &str) -> bool {
    matches!(
        res_name,
        "DA" | "DG" | "DI" | "A" | "G" | "ADE" | "GUA" | "I" | "RAD" | "RGU"
    )
}

impl MoleculeEntity {
    /// Extract phosphorus (P) atom positions grouped by chain ID.
    /// Chains are split at gaps where consecutive P-P distance exceeds ~8 Å.
    /// Only meaningful for DNA/RNA entities; returns empty for other molecule
    /// types.
    #[must_use]
    #[allow(
        clippy::too_many_lines,
        reason = "chain extraction with gap-splitting logic is inherently \
                  verbose"
    )]
    pub fn extract_p_atom_chains(&self) -> Vec<Vec<Vec3>> {
        const MAX_PP_DIST_SQ: f32 = 8.0 * 8.0;

        if !matches!(self.molecule_type, MoleculeType::DNA | MoleculeType::RNA)
        {
            return Vec::new();
        }

        let EntityKind::Polymer(data) = &self.kind else {
            return Vec::new();
        };

        let mut raw_chains: BTreeMap<u8, Vec<Vec3>> = BTreeMap::new();

        for chain in &data.chains {
            for residue in &chain.residues {
                #[allow(
                    clippy::excessive_nesting,
                    reason = "iterating atoms within residues within chains \
                              is natural"
                )]
                for idx in residue.atom_range.clone() {
                    let name = std::str::from_utf8(&data.atoms.atom_names[idx])
                        .unwrap_or("")
                        .trim();
                    if name == "P" {
                        let a = &data.atoms.atoms[idx];
                        raw_chains
                            .entry(chain.chain_id)
                            .or_default()
                            .push(Vec3::new(a.x, a.y, a.z));
                    }
                }
            }
        }

        // Split chains at large gaps (missing residues / chain breaks)
        let mut result = Vec::new();
        for chain in raw_chains.into_values() {
            let mut segment = Vec::new();
            for pos in chain {
                if let Some(&prev) = segment.last() {
                    #[allow(
                        clippy::excessive_nesting,
                        reason = "gap-splitting logic within chain iteration"
                    )]
                    if pos.distance_squared(prev) > MAX_PP_DIST_SQ {
                        if segment.len() >= 2 {
                            result.push(std::mem::take(&mut segment));
                        } else {
                            segment.clear();
                        }
                    }
                }
                segment.push(pos);
            }
            if segment.len() >= 2 {
                result.push(segment);
            }
        }

        result
    }

    /// Extract base ring geometry for each nucleotide residue.
    /// Only meaningful for DNA/RNA entities; returns empty for other molecule
    /// types.
    #[must_use]
    #[allow(
        clippy::too_many_lines,
        reason = "ring extraction with atom mapping is inherently verbose"
    )]
    pub fn extract_base_rings(&self) -> Vec<NucleotideRing> {
        if !matches!(self.molecule_type, MoleculeType::DNA | MoleculeType::RNA)
        {
            return Vec::new();
        }

        let EntityKind::Polymer(data) = &self.kind else {
            return Vec::new();
        };

        let mut rings = Vec::new();
        let mut skipped_partial = 0u32;

        for chain in &data.chains {
            #[allow(
                clippy::excessive_nesting,
                reason = "iterating residues within chains with atom lookup \
                          is natural"
            )]
            for residue in &chain.residues {
                let res_name =
                    std::str::from_utf8(&residue.name).unwrap_or("").trim();

                let Some(color) = ndb_base_color(res_name) else {
                    continue;
                };

                // Build atom_name -> position map for this residue
                let mut atom_map: HashMap<String, Vec3> = HashMap::new();
                for idx in residue.atom_range.clone() {
                    let name = std::str::from_utf8(&data.atoms.atom_names[idx])
                        .unwrap_or("")
                        .trim()
                        .trim_matches('\0')
                        .to_owned();
                    let a = &data.atoms.atoms[idx];
                    let _ = atom_map.insert(name, Vec3::new(a.x, a.y, a.z));
                }

                // Collect hex ring positions
                let hex_ring: Vec<Vec3> = HEX_RING_ATOMS
                    .iter()
                    .filter_map(|name| atom_map.get(*name).copied())
                    .collect();
                if hex_ring.len() != 6 {
                    skipped_partial += 1;
                    continue;
                }

                // Collect pent ring for purines
                let pent_ring = if is_purine(res_name) {
                    let pent: Vec<Vec3> = PENT_RING_ATOMS
                        .iter()
                        .filter_map(|name| atom_map.get(*name).copied())
                        .collect();
                    if pent.len() == 5 {
                        pent
                    } else {
                        Vec::new()
                    }
                } else {
                    Vec::new()
                };

                let c1_prime = atom_map
                    .get("C1'")
                    .or_else(|| atom_map.get("C1*"))
                    .copied();

                rings.push(NucleotideRing {
                    hex_ring,
                    pent_ring,
                    color,
                    c1_prime,
                });
            }
        }

        #[cfg(debug_assertions)]
        if skipped_partial > 0 {
            log::debug!(
                "[base_rings] {} rings extracted, {} residues skipped \
                 (missing ring atoms)",
                rings.len(),
                skipped_partial
            );
        }

        rings
    }

    /// Extract protein backbone chains (N-CA-C interleaved, split at chain
    /// breaks).
    ///
    /// Returns a [`ProteinBackbone`] containing one [`BackboneChain`] per
    /// contiguous polymer segment.
    #[must_use]
    pub fn extract_backbone(&self) -> ProteinBackbone {
        if self.molecule_type != MoleculeType::Protein {
            return ProteinBackbone {
                chains: Vec::new(),
                chain_ids: Vec::new(),
            };
        }

        let EntityKind::Polymer(data) = &self.kind else {
            return ProteinBackbone {
                chains: Vec::new(),
                chain_ids: Vec::new(),
            };
        };

        extract_backbone_from_polymer(data)
    }

    /// Extract sidechain atom data with topology.
    pub fn extract_sidechains<F, G>(
        &self,
        is_hydrophobic: F,
        get_bonds: G,
    ) -> SidechainAtoms
    where
        F: Fn(&str) -> bool,
        G: Fn(&str) -> Option<Vec<(&'static str, &'static str)>>,
    {
        if self.molecule_type != MoleculeType::Protein {
            return SidechainAtoms::default();
        }

        let EntityKind::Polymer(data) = &self.kind else {
            return SidechainAtoms::default();
        };

        extract_sidechains_from_polymer(data, is_hydrophobic, get_bonds)
    }
}

/// Inner implementation for backbone extraction from polymer data.
fn extract_backbone_from_polymer(data: &PolymerData) -> ProteinBackbone {
    let mut chains: Vec<Vec<Vec3>> = Vec::new();
    let mut chain_ids: Vec<u8> = Vec::new();

    for polymer_chain in &data.chains {
        let mut current_chain: Vec<Vec3> = Vec::new();
        let mut last_res_num: Option<i32> = None;

        for residue in &polymer_chain.residues {
            // Check for sequence gap -> chain break
            let is_sequence_gap =
                last_res_num.is_some_and(|r| (residue.number - r).abs() > 1);

            if is_sequence_gap && !current_chain.is_empty() {
                chains.push(std::mem::take(&mut current_chain));
                chain_ids.push(polymer_chain.chain_id);
            }

            // Collect backbone atoms (N, CA, C) for this residue
            #[allow(
                clippy::excessive_nesting,
                reason = "iterating atoms within residues within chains is \
                          natural"
            )]
            for idx in residue.atom_range.clone() {
                let atom_name =
                    std::str::from_utf8(&data.atoms.atom_names[idx])
                        .unwrap_or("")
                        .trim();
                if atom_name == "N" || atom_name == "CA" || atom_name == "C" {
                    let a = &data.atoms.atoms[idx];
                    current_chain.push(Vec3::new(a.x, a.y, a.z));
                }
            }

            last_res_num = Some(residue.number);
        }

        if !current_chain.is_empty() {
            chains.push(current_chain);
            chain_ids.push(polymer_chain.chain_id);
        }
    }

    ProteinBackbone {
        chains: chains.into_iter().map(BackboneChain::new).collect(),
        chain_ids,
    }
}

/// Returns true if `atom_name` looks like a hydrogen atom.
fn is_hydrogen_atom(atom_name: &str) -> bool {
    atom_name.starts_with('H')
        || atom_name.starts_with("1H")
        || atom_name.starts_with("2H")
        || atom_name.starts_with("3H")
        || (atom_name.len() >= 2
            && atom_name.as_bytes().first().is_some_and(u8::is_ascii_digit)
            && atom_name.as_bytes().get(1) == Some(&b'H'))
}

/// Per-residue context passed during sidechain atom collection.
struct ResidueContext<'a> {
    data: &'a PolymerData,
    chain_id: u8,
    residue: &'a super::Residue,
    res_name: &'a str,
    res_key: (u8, i32),
}

/// Intermediate state for sidechain extraction (first pass).
struct SidechainCollector {
    atoms: Vec<SidechainAtomData>,
    atom_index_map: HashMap<(u8, i32, String), u32>,
    residue_idx_map: HashMap<(u8, i32), u32>,
    next_residue_idx: u32,
}

impl SidechainCollector {
    fn new() -> Self {
        Self {
            atoms: Vec::new(),
            atom_index_map: HashMap::new(),
            residue_idx_map: HashMap::new(),
            next_residue_idx: 0,
        }
    }

    /// First pass: collect sidechain atoms and assign residue indices.
    fn collect_atoms<F>(&mut self, data: &PolymerData, is_hydrophobic: &F)
    where
        F: Fn(&str) -> bool,
    {
        for chain in &data.chains {
            for residue in &chain.residues {
                let res_name =
                    std::str::from_utf8(&residue.name).unwrap_or("UNK").trim();
                let ctx = ResidueContext {
                    data,
                    chain_id: chain.chain_id,
                    residue,
                    res_name,
                    res_key: (chain.chain_id, residue.number),
                };
                for idx in residue.atom_range.clone() {
                    self.process_atom(&ctx, is_hydrophobic, idx);
                }
            }
        }
    }

    /// Process a single atom within a residue.
    fn process_atom<F>(
        &mut self,
        ctx: &ResidueContext<'_>,
        is_hydrophobic: &F,
        idx: usize,
    ) where
        F: Fn(&str) -> bool,
    {
        let atom_name = std::str::from_utf8(&ctx.data.atoms.atom_names[idx])
            .unwrap_or("")
            .trim()
            .to_owned();

        if atom_name == "CA" {
            self.assign_residue_idx(ctx.res_key);
        } else if !matches!(atom_name.as_str(), "N" | "C" | "O")
            && !is_hydrogen_atom(&atom_name)
        {
            self.push_sidechain_atom(ctx, &atom_name, idx, is_hydrophobic);
        }
    }

    /// Assign a residue index if this residue hasn't been seen yet.
    fn assign_residue_idx(&mut self, res_key: (u8, i32)) {
        if let std::collections::hash_map::Entry::Vacant(e) =
            self.residue_idx_map.entry(res_key)
        {
            let _ = e.insert(self.next_residue_idx);
            self.next_residue_idx += 1;
        }
    }

    /// Add a non-backbone, non-hydrogen atom to the sidechain collection.
    #[allow(
        clippy::cast_possible_truncation,
        reason = "sidechain atom count fits in u32"
    )]
    fn push_sidechain_atom<F>(
        &mut self,
        ctx: &ResidueContext<'_>,
        atom_name: &str,
        idx: usize,
        is_hydrophobic: &F,
    ) where
        F: Fn(&str) -> bool,
    {
        let a = &ctx.data.atoms.atoms[idx];
        let pos = Vec3::new(a.x, a.y, a.z);
        let sidechain_idx = self.atoms.len() as u32;
        let _ = self.atom_index_map.insert(
            (ctx.chain_id, ctx.residue.number, atom_name.to_owned()),
            sidechain_idx,
        );
        let residue_idx =
            self.residue_idx_map.get(&ctx.res_key).copied().unwrap_or(0);
        self.atoms.push(SidechainAtomData {
            position: pos,
            residue_idx,
            atom_name: atom_name.to_owned(),
            is_hydrophobic: is_hydrophobic(ctx.res_name),
        });
    }
}

/// Second pass: generate CA->CB backbone-sidechain bonds.
#[allow(
    clippy::excessive_nesting,
    reason = "triple-nested loop over chains/residues/atoms is natural"
)]
fn collect_backbone_bonds(
    data: &PolymerData,
    atom_index_map: &HashMap<(u8, i32, String), u32>,
) -> Vec<(Vec3, u32)> {
    let mut backbone_bonds = Vec::new();
    for chain in &data.chains {
        for residue in &chain.residues {
            for idx in residue.atom_range.clone() {
                let name = std::str::from_utf8(&data.atoms.atom_names[idx])
                    .unwrap_or("")
                    .trim();
                if name == "CA" {
                    let a = &data.atoms.atoms[idx];
                    let ca_pos = Vec3::new(a.x, a.y, a.z);
                    let cb_key =
                        (chain.chain_id, residue.number, "CB".to_owned());
                    if let Some(&cb_idx) = atom_index_map.get(&cb_key) {
                        backbone_bonds.push((ca_pos, cb_idx));
                    }
                }
            }
        }
    }
    backbone_bonds
}

/// Third pass: generate intra-residue sidechain bonds from topology.
fn collect_topology_bonds<G>(
    data: &PolymerData,
    atom_index_map: &HashMap<(u8, i32, String), u32>,
    get_bonds: &G,
) -> Vec<(u32, u32)>
where
    G: Fn(&str) -> Option<Vec<(&'static str, &'static str)>>,
{
    let mut bonds = Vec::new();
    for chain in &data.chains {
        for residue in &chain.residues {
            let res_name =
                std::str::from_utf8(&residue.name).unwrap_or("UNK").trim();
            let Some(residue_bonds) = get_bonds(res_name) else {
                continue;
            };
            collect_residue_bonds(
                &mut bonds,
                atom_index_map,
                chain.chain_id,
                residue.number,
                &residue_bonds,
            );
        }
    }
    bonds
}

/// Match bond pairs from topology against the atom index map.
fn collect_residue_bonds(
    bonds: &mut Vec<(u32, u32)>,
    atom_index_map: &HashMap<(u8, i32, String), u32>,
    chain_id: u8,
    res_num: i32,
    residue_bonds: &[(&str, &str)],
) {
    for (a1, a2) in residue_bonds {
        let key1 = (chain_id, res_num, (*a1).to_owned());
        let key2 = (chain_id, res_num, (*a2).to_owned());
        if let (Some(&idx1), Some(&idx2)) =
            (atom_index_map.get(&key1), atom_index_map.get(&key2))
        {
            bonds.push((idx1, idx2));
        }
    }
}

/// Inner implementation for sidechain extraction from polymer data.
fn extract_sidechains_from_polymer<F, G>(
    data: &PolymerData,
    is_hydrophobic: F,
    get_bonds: G,
) -> SidechainAtoms
where
    F: Fn(&str) -> bool,
    G: Fn(&str) -> Option<Vec<(&'static str, &'static str)>>,
{
    let mut collector = SidechainCollector::new();
    collector.collect_atoms(data, &is_hydrophobic);

    let backbone_bonds =
        collect_backbone_bonds(data, &collector.atom_index_map);
    let bonds =
        collect_topology_bonds(data, &collector.atom_index_map, &get_bonds);

    SidechainAtoms {
        atoms: collector.atoms,
        bonds,
        backbone_bonds,
    }
}
