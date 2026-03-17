//! Helpers for extracting sidechain bond topology from `Coords`.

use std::collections::HashMap;

use crate::types::coords::Coords;

/// Generate sidechain bonds from residue bond topology callbacks.
pub(super) fn generate_sidechain_bonds<G>(
    coords: &Coords,
    atom_index_map: &HashMap<(u8, i32, String), u32>,
    get_bonds_fn: &G,
) -> Vec<(u32, u32)>
where
    G: Fn(&str) -> Option<Vec<(&'static str, &'static str)>>,
{
    let mut bonds: Vec<(u32, u32)> = Vec::new();
    let mut seen_residues: std::collections::HashSet<(u8, i32)> =
        std::collections::HashSet::new();

    for i in 0..coords.num_atoms {
        let atom_name = std::str::from_utf8(&coords.atom_names[i])
            .unwrap_or("")
            .trim();
        let chain_id = coords.chain_ids[i];
        let res_num = coords.res_nums[i];
        let res_name = std::str::from_utf8(&coords.res_names[i])
            .unwrap_or("UNK")
            .trim();

        if atom_name != "CA" || !seen_residues.insert((chain_id, res_num)) {
            continue;
        }

        let Some(residue_bonds) = get_bonds_fn(res_name) else {
            continue;
        };
        for (a1, a2) in residue_bonds {
            let key1 = (chain_id, res_num, a1.to_owned());
            let key2 = (chain_id, res_num, a2.to_owned());

            if let (Some(&idx1), Some(&idx2)) =
                (atom_index_map.get(&key1), atom_index_map.get(&key2))
            {
                bonds.push((idx1, idx2));
            }
        }
    }

    bonds
}
