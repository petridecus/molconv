//! Coordinate extraction and lookup utilities.

use glam::Vec3;

use crate::types::coords::Coords;

/// Extract backbone chains from COORDS data.
///
/// Returns a vector of chains, where each chain is a sequence of N-CA-C
/// positions. Chain breaks are detected by chain ID change or residue number
/// gap.
#[must_use]
pub fn extract_backbone_chains(coords: &Coords) -> Vec<Vec<Vec3>> {
    let mut chains: Vec<Vec<Vec3>> = Vec::new();
    let mut current_chain: Vec<Vec3> = Vec::new();
    let mut last_chain_id: Option<u8> = None;
    let mut last_res_num: Option<i32> = None;

    for i in 0..coords.num_atoms {
        let atom_name = std::str::from_utf8(&coords.atom_names[i])
            .unwrap_or("")
            .trim();

        // Only include N, CA, C for backbone spline (skip O and sidechains)
        if atom_name != "N" && atom_name != "CA" && atom_name != "C" {
            continue;
        }

        let chain_id = coords.chain_ids[i];
        let res_num = coords.res_nums[i];
        let pos =
            Vec3::new(coords.atoms[i].x, coords.atoms[i].y, coords.atoms[i].z);

        let is_chain_break = last_chain_id.is_some_and(|c| c != chain_id);
        let is_sequence_gap =
            last_res_num.is_some_and(|r| (res_num - r).abs() > 1);

        if (is_chain_break || is_sequence_gap) && !current_chain.is_empty() {
            chains.push(std::mem::take(&mut current_chain));
        }

        current_chain.push(pos);
        last_chain_id = Some(chain_id);

        if atom_name == "CA" {
            last_res_num = Some(res_num);
        }
    }

    if !current_chain.is_empty() {
        chains.push(current_chain);
    }

    chains
}

/// Extract CA positions from COORDS data.
#[must_use]
pub fn extract_ca_positions(coords: &Coords) -> Vec<Vec3> {
    let mut ca_positions = Vec::new();
    for i in 0..coords.num_atoms {
        let atom_name = std::str::from_utf8(&coords.atom_names[i])
            .unwrap_or("")
            .trim();
        if atom_name == "CA" {
            ca_positions.push(Vec3::new(
                coords.atoms[i].x,
                coords.atoms[i].y,
                coords.atoms[i].z,
            ));
        }
    }
    ca_positions
}

/// Extract CA positions from backbone chains (every 2nd element in N-CA-C
/// pattern).
#[must_use]
pub fn extract_ca_from_chains(chains: &[Vec<Vec3>]) -> Vec<Vec3> {
    let mut ca_positions = Vec::new();
    for chain in chains {
        // Backbone chains are N-CA-C pattern, so CA is every 3rd atom starting
        // at index 1
        for (i, pos) in chain.iter().enumerate() {
            if i % 3 == 1 {
                ca_positions.push(*pos);
            }
        }
    }
    ca_positions
}

/// Get a single CA position by residue index from backbone chains.
/// Returns None if residue_idx is out of bounds.
#[must_use]
pub fn get_ca_position_from_chains(
    chains: &[Vec<Vec3>],
    residue_idx: usize,
) -> Option<Vec3> {
    let mut current_idx = 0;
    for chain in chains {
        let residues_in_chain = chain.len() / 3;
        if residue_idx < current_idx + residues_in_chain {
            let local_idx = residue_idx - current_idx;
            let ca_idx = local_idx * 3 + 1; // CA is at index 1 in (N, CA, C)
            return chain.get(ca_idx).copied();
        }
        current_idx += residues_in_chain;
    }
    None
}

/// Get all backbone atom positions (N, CA, C) for a residue by index.
/// Returns None if residue_idx is out of bounds.
#[must_use]
pub fn get_backbone_atoms_from_chains(
    chains: &[Vec<Vec3>],
    residue_idx: usize,
) -> Option<(Vec3, Vec3, Vec3)> {
    let mut current_idx = 0;
    for chain in chains {
        let residues_in_chain = chain.len() / 3;
        if residue_idx < current_idx + residues_in_chain {
            let local_idx = residue_idx - current_idx;
            let base_idx = local_idx * 3;
            let n = chain.get(base_idx).copied()?;
            let ca = chain.get(base_idx + 1).copied()?;
            let c = chain.get(base_idx + 2).copied()?;
            return Some((n, ca, c));
        }
        current_idx += residues_in_chain;
    }
    None
}

/// Get the closest backbone atom position to a reference point for a residue.
///
/// Returns the position of N, CA, or C - whichever is closest to
/// `reference_point`. Returns None if residue_idx is out of bounds.
#[must_use]
pub fn get_closest_backbone_atom(
    chains: &[Vec<Vec3>],
    residue_idx: usize,
    reference_point: Vec3,
) -> Option<Vec3> {
    let (n, ca, c) = get_backbone_atoms_from_chains(chains, residue_idx)?;

    let dist_n = n.distance_squared(reference_point);
    let dist_ca = ca.distance_squared(reference_point);
    let dist_c = c.distance_squared(reference_point);

    if dist_n <= dist_ca && dist_n <= dist_c {
        Some(n)
    } else if dist_ca <= dist_c {
        Some(ca)
    } else {
        Some(c)
    }
}

/// Spatial data needed for per-residue atom proximity lookups.
pub struct ResidueAtomSearch<'a> {
    /// Backbone chains (N-CA-C interleaved).
    pub chains: &'a [Vec<Vec3>],
    /// Sidechain atom positions.
    pub sidechain_positions: &'a [Vec3],
    /// Per-sidechain-atom residue index.
    pub sidechain_residue_indices: &'a [u32],
}

/// Get the closest atom (backbone or sidechain) to a reference point for a
/// residue.
#[must_use]
pub fn get_closest_atom_for_residue(
    search: &ResidueAtomSearch<'_>,
    residue_idx: usize,
    reference_point: Vec3,
) -> Option<Vec3> {
    let ResidueAtomSearch {
        chains,
        sidechain_positions,
        sidechain_residue_indices,
    } = search;
    let mut closest: Option<(Vec3, f32)> = None;

    // Check backbone atoms (N, CA, C)
    if let Some((n, ca, c)) =
        get_backbone_atoms_from_chains(chains, residue_idx)
    {
        #[allow(clippy::tuple_array_conversions)]
        for pos in [n, ca, c] {
            let dist = pos.distance_squared(reference_point);
            if closest.as_ref().is_none_or(|(_, d)| dist < *d) {
                closest = Some((pos, dist));
            }
        }
    }

    // Check sidechain atoms for this residue
    #[allow(
        clippy::cast_precision_loss,
        reason = "residue index fits in f32 for comparison"
    )]
    for (i, &res_idx) in sidechain_residue_indices.iter().enumerate() {
        if res_idx as usize == residue_idx {
            if let Some(&pos) = sidechain_positions.get(i) {
                let dist = pos.distance_squared(reference_point);
                if closest.as_ref().is_none_or(|(_, d)| dist < *d) {
                    closest = Some((pos, dist));
                }
            }
        }
    }

    closest.map(|(pos, _)| pos)
}

/// Spatial data for per-residue atom proximity lookups that also carry atom
/// names.
pub struct NamedResidueAtomSearch<'a> {
    /// Backbone chains (N-CA-C interleaved).
    pub chains: &'a [Vec<Vec3>],
    /// Sidechain atom positions.
    pub sidechain_positions: &'a [Vec3],
    /// Per-sidechain-atom residue index.
    pub sidechain_residue_indices: &'a [u32],
    /// Per-sidechain-atom name (parallel to `sidechain_positions`).
    pub sidechain_atom_names: &'a [String],
}

/// Find the closest atom to a reference point within a residue, returning both
/// position and atom name.
#[must_use]
pub fn get_closest_atom_with_name(
    search: &NamedResidueAtomSearch<'_>,
    residue_idx: usize,
    reference_point: Vec3,
) -> Option<(Vec3, String)> {
    let NamedResidueAtomSearch {
        chains,
        sidechain_positions,
        sidechain_residue_indices,
        sidechain_atom_names,
    } = search;
    let mut closest: Option<(Vec3, String, f32)> = None;

    // Check backbone atoms (N, CA, C)
    if let Some((n, ca, c)) =
        get_backbone_atoms_from_chains(chains, residue_idx)
    {
        for (pos, name) in [(n, "N"), (ca, "CA"), (c, "C")] {
            let dist = pos.distance_squared(reference_point);
            if closest.as_ref().is_none_or(|(_, _, d)| dist < *d) {
                closest = Some((pos, name.to_owned(), dist));
            }
        }
    }

    // Check sidechain atoms for this residue
    #[allow(
        clippy::cast_precision_loss,
        reason = "residue index fits in f32 for comparison"
    )]
    for (i, &res_idx) in sidechain_residue_indices.iter().enumerate() {
        if res_idx as usize == residue_idx {
            if let (Some(&pos), Some(name)) =
                (sidechain_positions.get(i), sidechain_atom_names.get(i))
            {
                let dist = pos.distance_squared(reference_point);
                if closest.as_ref().is_none_or(|(_, _, d)| dist < *d) {
                    closest = Some((pos, name.clone(), dist));
                }
            }
        }
    }

    closest.map(|(pos, name, _)| (pos, name))
}

/// Get atom position by index.
#[must_use]
pub fn get_atom_position(coords: &Coords, index: usize) -> Option<Vec3> {
    coords.atoms.get(index).map(|a| Vec3::new(a.x, a.y, a.z))
}

/// Set atom position by index.
pub fn set_atom_position(coords: &mut Coords, index: usize, pos: Vec3) {
    if let Some(atom) = coords.atoms.get_mut(index) {
        atom.x = pos.x;
        atom.y = pos.y;
        atom.z = pos.z;
    }
}

/// Get position of a specific atom by residue number, chain ID, and atom name.
#[must_use]
pub fn get_atom_by_name(
    coords: &Coords,
    res_num: i32,
    chain_id: u8,
    atom_name: &str,
) -> Option<Vec3> {
    for i in 0..coords.num_atoms {
        if coords.res_nums[i] == res_num && coords.chain_ids[i] == chain_id {
            let name = std::str::from_utf8(&coords.atom_names[i])
                .unwrap_or("")
                .trim();
            if name == atom_name {
                return Some(Vec3::new(
                    coords.atoms[i].x,
                    coords.atoms[i].y,
                    coords.atoms[i].z,
                ));
            }
        }
    }
    None
}

/// Get CA position for a specific residue by residue number and chain ID.
#[must_use]
pub fn get_ca_for_residue(
    coords: &Coords,
    res_num: i32,
    chain_id: u8,
) -> Option<Vec3> {
    get_atom_by_name(coords, res_num, chain_id, "CA")
}

/// Build a map of (chain_id, res_num) -> CA position for efficient lookup.
#[must_use]
pub fn build_ca_position_map(
    coords: &Coords,
) -> std::collections::HashMap<(u8, i32), Vec3> {
    let mut map = std::collections::HashMap::new();
    for i in 0..coords.num_atoms {
        let name = std::str::from_utf8(&coords.atom_names[i])
            .unwrap_or("")
            .trim();
        if name == "CA" {
            let key = (coords.chain_ids[i], coords.res_nums[i]);
            let _ = map.insert(
                key,
                Vec3::new(
                    coords.atoms[i].x,
                    coords.atoms[i].y,
                    coords.atoms[i].z,
                ),
            );
        }
    }
    map
}

/// Compute centroid of a point set.
#[must_use]
#[allow(clippy::cast_precision_loss, reason = "point count fits in f32")]
pub fn centroid(points: &[Vec3]) -> Vec3 {
    if points.is_empty() {
        return Vec3::ZERO;
    }
    let sum: Vec3 = points.iter().copied().sum();
    sum / points.len() as f32
}
