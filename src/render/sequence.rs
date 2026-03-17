//! Amino acid sequence extraction from `Coords`.

use crate::types::coords::Coords;

/// Extract amino acid sequences from `Coords`.
#[must_use]
pub fn extract_sequences(coords: &Coords) -> (String, Vec<(u8, String)>) {
    let mut full_sequence = String::new();
    let mut chain_sequences: Vec<(u8, String)> = Vec::new();
    let mut current_chain_id: Option<u8> = None;
    let mut current_chain_seq = String::new();
    let mut last_res_key: Option<(u8, i32)> = None;

    for i in 0..coords.num_atoms {
        let atom_name = std::str::from_utf8(&coords.atom_names[i])
            .unwrap_or("")
            .trim();
        if atom_name != "CA" {
            continue;
        }

        let chain_id = coords.chain_ids[i];
        let res_num = coords.res_nums[i];
        let res_name = std::str::from_utf8(&coords.res_names[i])
            .unwrap_or("UNK")
            .trim();

        let res_key = (chain_id, res_num);
        if last_res_key == Some(res_key) {
            continue;
        }

        if let Some(cid) = current_chain_id {
            if cid != chain_id && !current_chain_seq.is_empty() {
                chain_sequences
                    .push((cid, std::mem::take(&mut current_chain_seq)));
            }
        }

        let aa = three_to_one(res_name);
        full_sequence.push(aa);
        current_chain_seq.push(aa);
        current_chain_id = Some(chain_id);
        last_res_key = Some(res_key);
    }

    if let Some(cid) = current_chain_id {
        if !current_chain_seq.is_empty() {
            chain_sequences.push((cid, current_chain_seq));
        }
    }

    (full_sequence, chain_sequences)
}

fn three_to_one(three: &str) -> char {
    match three {
        "ALA" => 'A',
        "CYS" | "CYX" => 'C',
        "ASP" => 'D',
        "GLU" => 'E',
        "PHE" => 'F',
        "GLY" => 'G',
        "HIS" | "HSD" | "HSE" | "HSP" => 'H',
        "ILE" => 'I',
        "LYS" => 'K',
        "LEU" => 'L',
        "MET" | "MSE" => 'M',
        "ASN" => 'N',
        "PRO" => 'P',
        "GLN" => 'Q',
        "ARG" => 'R',
        "SER" => 'S',
        "THR" => 'T',
        "VAL" => 'V',
        "TRP" => 'W',
        "TYR" => 'Y',
        _ => 'X',
    }
}
