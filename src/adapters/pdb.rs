//! PDB and mmCIF format parsing and writing.

use std::fmt::Write as _;
use std::io::BufReader;

use pdbtbx::{
    ContainsAtomConformer, ContainsAtomConformerResidue,
    ContainsAtomConformerResidueChain, Format, ReadOptions, StrictnessLevel,
};

use crate::types::coords::{
    deserialize, serialize, ChainIdMapper, Coords, CoordsAtom, CoordsError,
    Element,
};
use crate::types::entity::{split_into_entities, MoleculeEntity};

/// Parse PDB format string to COORDS binary format.
///
/// # Errors
///
/// Returns [`CoordsError`] if parsing or serialization fails.
pub fn pdb_to_coords(pdb_str: &str) -> Result<Vec<u8>, CoordsError> {
    let coords = parse_structure_to_coords(pdb_str, Format::Pdb)?;
    serialize(&coords)
}

/// Parse mmCIF format string to COORDS binary format.
///
/// # Errors
///
/// Returns [`CoordsError`] if parsing or serialization fails.
pub fn mmcif_to_coords(cif_str: &str) -> Result<Vec<u8>, CoordsError> {
    let coords = parse_structure_to_coords(cif_str, Format::Mmcif)?;
    serialize(&coords)
}

/// Parse PDB format string directly to Coords struct.
///
/// # Errors
///
/// Returns [`CoordsError`] if parsing fails.
pub fn pdb_str_to_coords(pdb_str: &str) -> Result<Coords, CoordsError> {
    parse_structure_to_coords(pdb_str, Format::Pdb)
}

/// Parse mmCIF format string directly to Coords struct.
///
/// # Errors
///
/// Returns [`CoordsError`] if parsing fails.
pub fn mmcif_str_to_coords(cif_str: &str) -> Result<Coords, CoordsError> {
    parse_structure_to_coords(cif_str, Format::Mmcif)
}

/// Load mmCIF file directly to Coords struct.
///
/// # Errors
///
/// Returns [`CoordsError`] if the file cannot be read or parsing fails.
pub fn mmcif_file_to_coords(
    path: &std::path::Path,
) -> Result<Coords, CoordsError> {
    let content = std::fs::read_to_string(path).map_err(|e| {
        CoordsError::PdbParseError(format!("Failed to read file: {e}"))
    })?;
    parse_structure_to_coords(&content, Format::Mmcif)
}

/// Load PDB file directly to Coords struct.
///
/// Sanitizes non-standard lines (e.g. GROMACS/MemProtMD output) before parsing.
///
/// # Errors
///
/// Returns [`CoordsError`] if the file cannot be read or parsing fails.
pub fn pdb_file_to_coords(
    path: &std::path::Path,
) -> Result<Coords, CoordsError> {
    let content = std::fs::read_to_string(path).map_err(|e| {
        CoordsError::PdbParseError(format!("Failed to read file: {e}"))
    })?;
    let sanitized = sanitize_pdb(&content);
    parse_structure_to_coords(&sanitized, Format::Pdb)
}

/// Strip or fix PDB lines that cause pdbtbx to error even in Loose mode.
/// GROMACS/MemProtMD PDBs have bare `REMARK    text` without the required
/// remark number, which pdbtbx treats as InvalidatingError.
fn sanitize_pdb(content: &str) -> String {
    content
        .lines()
        .map(|line| {
            line.strip_prefix("REMARK").map_or_else(
                || line.to_owned(),
                |after| {
                    // PDB spec: columns 8-10 should be a remark number.
                    // If there's no number, prefix one (0) so pdbtbx accepts
                    // it.
                    let trimmed = after.trim_start();
                    if trimmed.is_empty()
                        || !trimmed.as_bytes()[0].is_ascii_digit()
                    {
                        format!("REMARK   0 {trimmed}")
                    } else {
                        line.to_owned()
                    }
                },
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Load a structure file (PDB or mmCIF) by detecting format from extension.
/// Falls back to mmCIF if extension is unrecognized.
///
/// # Errors
///
/// Returns [`CoordsError`] if the file cannot be read or parsing fails.
pub fn structure_file_to_coords(
    path: &std::path::Path,
) -> Result<Coords, CoordsError> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();
    match ext.as_str() {
        "pdb" | "ent" => pdb_file_to_coords(path),
        _ => mmcif_file_to_coords(path),
    }
}

/// Convert a residue/conformer name to a 3-byte array, space-padded.
fn name_to_res_bytes(name: &str) -> [u8; 3] {
    let mut buf = [b' '; 3];
    for (i, b) in name.bytes().take(3).enumerate() {
        buf[i] = b;
    }
    buf
}

/// Convert an atom name string to a 4-byte array, space-padded.
fn name_to_atom_bytes(name: &str) -> [u8; 4] {
    let mut buf = [b' '; 4];
    for (i, b) in name.bytes().take(4).enumerate() {
        buf[i] = b;
    }
    buf
}

/// Extract per-atom data from a pdbtbx hierarchy entry.
fn extract_atom_data(
    hier: &pdbtbx::AtomConformerResidueChainModel<'_>,
    chain_mapper: &mut ChainIdMapper,
) -> (CoordsAtom, u8, [u8; 3], i32, [u8; 4], Element) {
    let atom = hier.atom();
    let chain = hier.chain();
    let residue = hier.residue();
    let conformer = hier.conformer();

    #[allow(clippy::cast_possible_truncation)]
    let coords_atom = CoordsAtom {
        x: atom.x() as f32,
        y: atom.y() as f32,
        z: atom.z() as f32,
        occupancy: atom.occupancy() as f32,
        b_factor: atom.b_factor() as f32,
    };

    let chain_id = chain_mapper.get_or_assign(chain.id());
    let res_name = name_to_res_bytes(conformer.name());
    #[allow(clippy::cast_possible_truncation)]
    let res_num = residue.serial_number() as i32;
    let aname = atom.name();
    let atom_name = name_to_atom_bytes(aname);
    let elem = atom.element().map_or_else(
        || Element::from_atom_name(aname),
        |e| Element::from_symbol(e.symbol()),
    );

    (coords_atom, chain_id, res_name, res_num, atom_name, elem)
}

/// Internal function to parse either PDB or mmCIF using pdbtbx.
fn parse_structure_to_coords(
    input: &str,
    format: Format,
) -> Result<Coords, CoordsError> {
    let reader = BufReader::new(input.as_bytes());

    let (pdb, _errors) = ReadOptions::new()
        .set_format(format)
        .set_level(StrictnessLevel::Loose)
        .read_raw(reader)
        .map_err(|errs| {
            CoordsError::PdbParseError(
                errs.iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
                    .join("; "),
            )
        })?;

    let mut atoms = Vec::new();
    let mut chain_ids = Vec::new();
    let mut res_names = Vec::new();
    let mut res_nums = Vec::new();
    let mut atom_names = Vec::new();
    let mut elements = Vec::new();
    let mut chain_mapper = ChainIdMapper::new();

    for hier in pdb.atoms_with_hierarchy() {
        let (atom, cid, rn, rnum, an, elem) =
            extract_atom_data(&hier, &mut chain_mapper);
        atoms.push(atom);
        chain_ids.push(cid);
        res_names.push(rn);
        res_nums.push(rnum);
        atom_names.push(an);
        elements.push(elem);
    }

    if atoms.is_empty() {
        return Err(CoordsError::PdbParseError(
            "No atoms found in structure".to_owned(),
        ));
    }

    Ok(Coords {
        num_atoms: atoms.len(),
        atoms,
        chain_ids,
        res_names,
        res_nums,
        atom_names,
        elements,
    })
}

/// Convert COORDS binary to PDB format string.
///
/// # Errors
///
/// Returns [`CoordsError`] if deserialization fails.
pub fn coords_to_pdb(coords_bytes: &[u8]) -> Result<String, CoordsError> {
    let coords = deserialize(coords_bytes)?;

    let mut pdb_string = String::new();

    for i in 0..coords.num_atoms {
        let atom = &coords.atoms[i];
        let chain_id = coords.chain_ids[i] as char;
        let res_num = coords.res_nums[i];

        let atom_name =
            std::str::from_utf8(&coords.atom_names[i]).unwrap_or("X   ");
        let res_name =
            std::str::from_utf8(&coords.res_names[i]).unwrap_or("UNK");

        let _ = writeln!(
            pdb_string,
            "ATOM  {:>5} {:<4} {:>3} {}{:>4}    \
             {:>8.3}{:>8.3}{:>8.3}{:>6.2}{:>6.2}",
            i + 1,
            atom_name,
            res_name,
            chain_id,
            res_num,
            atom.x,
            atom.y,
            atom.z,
            atom.occupancy,
            atom.b_factor
        );
    }

    pdb_string.push_str("END\n");

    Ok(pdb_string)
}

// ---------------------------------------------------------------------------
// Entity-returning variants
// ---------------------------------------------------------------------------

/// Parse PDB format string to entity list.
///
/// # Errors
///
/// Returns [`CoordsError`] if parsing fails.
pub fn pdb_str_to_entities(
    pdb_str: &str,
) -> Result<Vec<MoleculeEntity>, CoordsError> {
    let coords = parse_structure_to_coords(pdb_str, Format::Pdb)?;
    Ok(split_into_entities(&coords))
}

/// Parse mmCIF format string to entity list.
///
/// # Errors
///
/// Returns [`CoordsError`] if parsing fails.
pub fn mmcif_str_to_entities(
    cif_str: &str,
) -> Result<Vec<MoleculeEntity>, CoordsError> {
    let coords = parse_structure_to_coords(cif_str, Format::Mmcif)?;
    Ok(split_into_entities(&coords))
}

/// Load PDB file to entity list.
///
/// # Errors
///
/// Returns [`CoordsError`] if the file cannot be read or parsing fails.
pub fn pdb_file_to_entities(
    path: &std::path::Path,
) -> Result<Vec<MoleculeEntity>, CoordsError> {
    let coords = pdb_file_to_coords(path)?;
    Ok(split_into_entities(&coords))
}

/// Load mmCIF file to entity list.
///
/// # Errors
///
/// Returns [`CoordsError`] if the file cannot be read or parsing fails.
pub fn mmcif_file_to_entities(
    path: &std::path::Path,
) -> Result<Vec<MoleculeEntity>, CoordsError> {
    let coords = mmcif_file_to_coords(path)?;
    Ok(split_into_entities(&coords))
}

/// Load structure file (PDB or mmCIF, detected by extension) to entity list.
///
/// # Errors
///
/// Returns [`CoordsError`] if the file cannot be read or parsing fails.
pub fn structure_file_to_entities(
    path: &std::path::Path,
) -> Result<Vec<MoleculeEntity>, CoordsError> {
    let coords = structure_file_to_coords(path)?;
    Ok(split_into_entities(&coords))
}
