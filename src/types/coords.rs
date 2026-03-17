//! Core data structures and binary serialization for COORDS format.
//!
//! Binary format (COORDS01, backward compatible with COORDS00):
//! - 8-byte magic header: "COORDS01" (or "COORDS00" for legacy)
//! - 4-byte big-endian u32: number of atoms
//! - Per-atom payload:
//!   - 12 bytes: x, y, z (f32 each, big-endian)
//!   - 1 byte: `chain_id` (ASCII byte)
//!   - 3 bytes: residue name (3-character code)
//!   - 4 bytes: residue number (`i32`, big-endian)
//!   - 4 bytes: atom name (4-character code)
//!   - [COORDS01 only] 2 bytes: element symbol (padded with 0)

use thiserror::Error;

pub use super::element::Element;

/// Errors that can occur during COORDS operations.
#[derive(Error, Debug)]
pub enum CoordsError {
    /// The binary data does not conform to the expected COORDS/ASSEM format.
    #[error("Invalid COORDS format: {0}")]
    InvalidFormat(String),
    /// A PDB file could not be parsed.
    #[error("Failed to parse PDB: {0}")]
    PdbParseError(String),
    /// An error occurred during binary serialization or deserialization.
    #[error("Serialization error: {0}")]
    SerializationError(String),
}

/// Maps multi-character chain ID strings to unique `u8` values.
///
/// Structures with >26 chains (ribosomes, virus capsids) use multi-character
/// chain IDs in mmCIF format (e.g., "AA", "AB"). Since `Coords.chain_ids`
/// stores a single `u8` per atom, this mapper assigns a unique byte to each
/// distinct chain string, preventing collisions that cause cross-chain
/// rendering artifacts.
///
/// Assigns printable ASCII characters (A-Z, a-z, 0-9, then other printable
/// chars) so that PDB export produces valid chain ID columns.
pub struct ChainIdMapper {
    map: std::collections::HashMap<String, u8>,
    next_idx: usize,
}

/// Printable chain ID characters in conventional order: A-Z, a-z, 0-9, then
/// remaining printable ASCII. Covers up to 94 unique chains.
const CHAIN_CHARS: &[u8] =
    b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!#$%&()*+,-./:;<=>?@[]^_`{|}~";

impl Default for ChainIdMapper {
    fn default() -> Self {
        Self::new()
    }
}

impl ChainIdMapper {
    /// Create a new empty mapper with no chain assignments.
    #[must_use]
    pub fn new() -> Self {
        Self {
            map: std::collections::HashMap::new(),
            next_idx: 0,
        }
    }

    /// Get or assign a unique `u8` for the given chain ID string.
    pub fn get_or_assign(&mut self, chain_id: &str) -> u8 {
        if let Some(&id) = self.map.get(chain_id) {
            return id;
        }
        let byte = if self.next_idx < CHAIN_CHARS.len() {
            CHAIN_CHARS[self.next_idx]
        } else {
            // Fallback for >94 chains: use raw sequential bytes past printable
            // range. Extremely rare — only theoretical virus
            // capsids approach this.
            #[allow(clippy::cast_possible_truncation)]
            // intentional wrapping for rare >94 chain case
            {
                (self.next_idx - CHAIN_CHARS.len()) as u8
            }
        };
        self.next_idx += 1;
        let _ = self.map.insert(chain_id.to_owned(), byte);
        byte
    }
}

/// Single atom with coordinates and crystallographic factors.
#[derive(Debug, Clone)]
pub struct CoordsAtom {
    /// X coordinate in angstroms.
    pub x: f32,
    /// Y coordinate in angstroms.
    pub y: f32,
    /// Z coordinate in angstroms.
    pub z: f32,
    /// Crystallographic occupancy (0.0 to 1.0).
    pub occupancy: f32,
    /// Temperature factor (B-factor) in square angstroms.
    pub b_factor: f32,
}

/// Complete coordinate structure with atom metadata.
#[derive(Debug, Clone)]
pub struct Coords {
    /// Total number of atoms in this structure.
    pub num_atoms: usize,
    /// Per-atom position and crystallographic data.
    pub atoms: Vec<CoordsAtom>,
    /// Per-atom chain identifier byte (e.g., b'A').
    pub chain_ids: Vec<u8>,
    /// Per-atom 3-character residue name (e.g., b"ALA").
    pub res_names: Vec<[u8; 3]>,
    /// Per-atom residue sequence number.
    pub res_nums: Vec<i32>,
    /// Per-atom 4-character PDB atom name (e.g., b" CA ").
    pub atom_names: Vec<[u8; 4]>,
    /// Chemical element per atom (for ball-and-stick rendering, bond
    /// inference).
    pub elements: Vec<Element>,
}

/// Metadata about atoms for GPU uniform buffers (coloring, selection, etc.)
#[derive(Debug, Clone)]
pub struct AtomMetadata {
    /// Chain IDs as bytes (e.g., b'A', b'B')
    pub chain_ids: Vec<u8>,
    /// Residue indices
    pub residue_indices: Vec<i32>,
    /// Atom type indices (derived from atom name for coloring)
    pub atom_type_indices: Vec<u8>,
    /// B-factors (can drive sphere size/opacity)
    pub b_factors: Vec<f32>,
}

/// Atoms present in a single residue (for validation).
#[derive(Debug, Clone)]
pub struct ResidueAtoms {
    /// Chain identifier byte for this residue.
    pub chain_id: u8,
    /// Residue sequence number.
    pub res_num: i32,
    /// 3-character residue name.
    pub res_name: [u8; 3],
    /// Atom names present in this residue
    pub atoms: Vec<[u8; 4]>,
}

/// Result of validating COORDS completeness.
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Whether all expected atoms are present
    pub is_complete: bool,
    /// Missing atoms: (res_num, res_name, list of missing atom names)
    pub missing_atoms: Vec<(i32, String, Vec<String>)>,
    /// Unexpected atoms: (res_num, res_name, list of extra atom names)
    pub extra_atoms: Vec<(i32, String, Vec<String>)>,
    /// Total residues checked
    pub total_residues: usize,
    /// Residues with missing atoms
    pub incomplete_residues: usize,
}

// ============================================================================
// Binary serialization (COORDS01 format)
// ============================================================================

/// Magic header bytes identifying the COORDS01 binary format.
pub(crate) const COORDS_MAGIC: &[u8; 8] = b"COORDS01";
const COORDS_MAGIC_V0: &[u8; 8] = b"COORDS00";
/// Magic header bytes identifying the ASSEM01 assembly binary format.
pub const ASSEMBLY_MAGIC: &[u8; 8] = b"ASSEM01\0";

/// One atom's worth of parsed COORDS binary data.
struct ParsedAtom<'a> {
    atom: CoordsAtom,
    chain_id: u8,
    res_name: [u8; 3],
    res_num: i32,
    atom_name: [u8; 4],
    element: Element,
    rest: &'a [u8],
}

/// Read one atom's worth of COORDS binary data from a cursor slice.
/// `has_elements` controls whether a 2-byte element symbol follows
/// (COORDS01) or elements are inferred from atom name (COORDS00).
fn read_atom_from_cursor(
    cursor: &[u8],
    has_elements: bool,
) -> Result<ParsedAtom<'_>, CoordsError> {
    let x = f32::from_be_bytes(cursor[0..4].try_into().map_err(|_| {
        CoordsError::SerializationError("Invalid x coordinate".to_owned())
    })?);
    let y = f32::from_be_bytes(cursor[4..8].try_into().map_err(|_| {
        CoordsError::SerializationError("Invalid y coordinate".to_owned())
    })?);
    let z = f32::from_be_bytes(cursor[8..12].try_into().map_err(|_| {
        CoordsError::SerializationError("Invalid z coordinate".to_owned())
    })?);
    let atom = CoordsAtom {
        x,
        y,
        z,
        occupancy: 1.0,
        b_factor: 0.0,
    };
    let rest = &cursor[12..];

    let chain_id = rest[0];
    let rest = &rest[1..];

    let mut res_name = [0u8; 3];
    res_name.copy_from_slice(&rest[0..3]);
    let rest = &rest[3..];

    let res_num = i32::from_be_bytes(rest[0..4].try_into().map_err(|_| {
        CoordsError::SerializationError("Invalid residue number".to_owned())
    })?);
    let rest = &rest[4..];

    let mut atom_name = [0u8; 4];
    atom_name.copy_from_slice(&rest[0..4]);
    let rest = &rest[4..];

    let (elem, rest) = if has_elements {
        let sym_str = std::str::from_utf8(&rest[0..2])
            .unwrap_or("")
            .trim_matches('\0')
            .trim();
        (Element::from_symbol(sym_str), &rest[2..])
    } else {
        let aname = std::str::from_utf8(&atom_name).unwrap_or("");
        (Element::from_atom_name(aname), rest)
    };

    Ok(ParsedAtom {
        atom,
        chain_id,
        res_name,
        res_num,
        atom_name,
        element: elem,
        rest,
    })
}

/// Read `num_atoms` atoms from a cursor, returning a `Coords` struct and the
/// remaining bytes.
fn read_atoms_to_coords(
    mut cursor: &[u8],
    num_atoms: usize,
    has_elements: bool,
) -> Result<(Coords, &[u8]), CoordsError> {
    let mut atoms = Vec::with_capacity(num_atoms);
    let mut chain_ids = Vec::with_capacity(num_atoms);
    let mut res_names = Vec::with_capacity(num_atoms);
    let mut res_nums = Vec::with_capacity(num_atoms);
    let mut atom_names = Vec::with_capacity(num_atoms);
    let mut elements = Vec::with_capacity(num_atoms);

    for _ in 0..num_atoms {
        let parsed = read_atom_from_cursor(cursor, has_elements)?;
        atoms.push(parsed.atom);
        chain_ids.push(parsed.chain_id);
        res_names.push(parsed.res_name);
        res_nums.push(parsed.res_num);
        atom_names.push(parsed.atom_name);
        elements.push(parsed.element);
        cursor = parsed.rest;
    }

    let coords = Coords {
        num_atoms,
        atoms,
        chain_ids,
        res_names,
        res_nums,
        atom_names,
        elements,
    };
    Ok((coords, cursor))
}

/// Deserialize COORDS binary format to `Coords` struct.
/// Supports both COORDS00 (no element data) and COORDS01 (with element data).
///
/// # Errors
///
/// Returns `CoordsError::InvalidFormat` if the magic header is wrong or data is
/// truncated. Returns `CoordsError::SerializationError` if individual atom
/// fields cannot be parsed.
pub fn deserialize(coords_bytes: &[u8]) -> Result<Coords, CoordsError> {
    if coords_bytes.len() < 8 {
        return Err(CoordsError::InvalidFormat(
            "Data too short to be valid COORDS".to_owned(),
        ));
    }

    let magic = &coords_bytes[0..8];
    let has_elements = magic == COORDS_MAGIC;
    if magic != COORDS_MAGIC && magic != COORDS_MAGIC_V0 {
        return Err(CoordsError::InvalidFormat(
            "Invalid magic number in COORDS header".to_owned(),
        ));
    }

    let cursor = &coords_bytes[8..];
    let num_atoms = u32::from_be_bytes(
        cursor
            .get(0..4)
            .ok_or_else(|| {
                CoordsError::InvalidFormat("Missing num_atoms field".to_owned())
            })?
            .try_into()
            .map_err(|_| {
                CoordsError::SerializationError(
                    "Invalid num_atoms size".to_owned(),
                )
            })?,
    ) as usize;

    let per_atom = if has_elements { 26 } else { 24 };
    if cursor.len() - 4 < num_atoms * per_atom {
        return Err(CoordsError::InvalidFormat(
            "Data too short for declared number of atoms".to_owned(),
        ));
    }

    let (coords, _) =
        read_atoms_to_coords(&cursor[4..], num_atoms, has_elements)?;
    Ok(coords)
}

/// Serialize `Coords` struct to COORDS binary format (COORDS01).
///
/// # Errors
///
/// Currently infallible but returns `Result` for API consistency.
pub fn serialize(coords: &Coords) -> Result<Vec<u8>, CoordsError> {
    let mut buffer =
        Vec::with_capacity(8 + 4 + coords.num_atoms * (12 + 1 + 3 + 4 + 4 + 2));

    buffer.extend_from_slice(COORDS_MAGIC);

    #[allow(clippy::cast_possible_truncation)] // atom counts fit in u32
    let num_atoms_u32 = coords.num_atoms as u32;
    buffer.extend_from_slice(&num_atoms_u32.to_be_bytes());

    for i in 0..coords.num_atoms {
        let atom = &coords.atoms[i];
        buffer.extend_from_slice(&atom.x.to_be_bytes());
        buffer.extend_from_slice(&atom.y.to_be_bytes());
        buffer.extend_from_slice(&atom.z.to_be_bytes());

        buffer.push(coords.chain_ids[i]);

        buffer.extend_from_slice(&coords.res_names[i]);

        buffer.extend_from_slice(&coords.res_nums[i].to_be_bytes());

        buffer.extend_from_slice(&coords.atom_names[i]);

        // Element symbol: 2 bytes, null-padded
        let sym = coords.elements.get(i).map_or("X", |e| e.symbol());
        let sym_bytes = sym.as_bytes();
        buffer.push(sym_bytes.first().copied().unwrap_or(b'X'));
        buffer.push(sym_bytes.get(1).copied().unwrap_or(0));
    }

    Ok(buffer)
}

/// Get the number of atoms in a COORDS binary without full deserialization.
/// Useful for pre-allocating GPU buffers.
///
/// # Errors
///
/// Returns `CoordsError::InvalidFormat` if the data is too short or has an
/// invalid magic header.
pub fn atom_count(coords_bytes: &[u8]) -> Result<usize, CoordsError> {
    if coords_bytes.len() < 12 {
        return Err(CoordsError::InvalidFormat(
            "Data too short to read atom count".to_owned(),
        ));
    }

    let magic = &coords_bytes[0..8];
    if magic != COORDS_MAGIC && magic != COORDS_MAGIC_V0 {
        return Err(CoordsError::InvalidFormat(
            "Invalid magic number".to_owned(),
        ));
    }

    let num_atoms =
        u32::from_be_bytes(coords_bytes[8..12].try_into().map_err(|_| {
            CoordsError::InvalidFormat("Invalid atom count bytes".to_owned())
        })?) as usize;

    Ok(num_atoms)
}

// ============================================================================
// ASSEM01 binary serialization (assembly format with entity metadata)
// ============================================================================

use super::entity::{coords_to_entity_kind, MoleculeEntity, MoleculeType};

/// Serialize a list of entities to ASSEM01 binary format.
///
/// Format:
/// - 8 bytes: magic "ASSEM01\0"
/// - 4 bytes: entity_count (u32 BE)
/// - Per entity header (5 bytes each):
///   - 1 byte: `molecule_type` wire byte
///   - 4 bytes: `atom_count` (`u32` BE)
/// - Per atom (26 bytes, same as COORDS01):
///   - 12 bytes: x,y,z (`f32` BE)
///   - 1 byte: `chain_id`
///   - 3 bytes: `res_name`
///   - 4 bytes: `res_num` (`i32` BE)
///   - 4 bytes: `atom_name`
///   - 2 bytes: element symbol
///
/// # Errors
///
/// Currently infallible but returns `Result` for API consistency.
pub fn serialize_assembly(
    entities: &[MoleculeEntity],
) -> Result<Vec<u8>, CoordsError> {
    let total_atoms: usize =
        entities.iter().map(MoleculeEntity::atom_count).sum();
    let header_size = 8 + 4 + entities.len() * 5;
    let atom_size = total_atoms * 26;
    let mut buffer = Vec::with_capacity(header_size + atom_size);

    // Magic
    buffer.extend_from_slice(ASSEMBLY_MAGIC);

    // Entity count
    #[allow(clippy::cast_possible_truncation)] // entity count fits in u32
    buffer.extend_from_slice(&(entities.len() as u32).to_be_bytes());

    // Per-entity headers
    for entity in entities {
        buffer.push(entity.molecule_type.to_wire_byte());
        #[allow(clippy::cast_possible_truncation)] // atom count fits in u32
        buffer.extend_from_slice(&(entity.atom_count() as u32).to_be_bytes());
    }

    // Atom data (same layout as COORDS01)
    for entity in entities {
        let c = entity.to_coords();
        for i in 0..c.num_atoms {
            let atom = &c.atoms[i];
            buffer.extend_from_slice(&atom.x.to_be_bytes());
            buffer.extend_from_slice(&atom.y.to_be_bytes());
            buffer.extend_from_slice(&atom.z.to_be_bytes());
            buffer.push(c.chain_ids[i]);
            buffer.extend_from_slice(&c.res_names[i]);
            buffer.extend_from_slice(&c.res_nums[i].to_be_bytes());
            buffer.extend_from_slice(&c.atom_names[i]);
            let sym = c.elements.get(i).map_or("X", |e| e.symbol());
            let sym_bytes = sym.as_bytes();
            buffer.push(sym_bytes.first().copied().unwrap_or(b'X'));
            buffer.push(sym_bytes.get(1).copied().unwrap_or(0));
        }
    }

    Ok(buffer)
}

/// Parse entity headers from ASSEM01 binary, returning
/// (molecule_type, atom_count) pairs and the offset past all headers.
fn parse_entity_headers(
    bytes: &[u8],
    entity_count: usize,
) -> Result<(Vec<(MoleculeType, usize)>, usize), CoordsError> {
    let mut headers = Vec::with_capacity(entity_count);
    let mut offset = 12;
    for _ in 0..entity_count {
        let mol_type =
            MoleculeType::from_wire_byte(bytes[offset]).ok_or_else(|| {
                CoordsError::InvalidFormat(format!(
                    "Unknown molecule type byte: {}",
                    bytes[offset]
                ))
            })?;
        offset += 1;
        let atom_count = u32::from_be_bytes(
            bytes[offset..offset + 4].try_into().map_err(|_| {
                CoordsError::InvalidFormat(
                    "Invalid atom count in entity header".to_owned(),
                )
            })?,
        ) as usize;
        offset += 4;
        headers.push((mol_type, atom_count));
    }
    Ok((headers, offset))
}

/// Deserialize ASSEM01 binary format back to a list of entities.
///
/// # Errors
///
/// Returns `CoordsError::InvalidFormat` if the magic header, entity headers,
/// or atom data are malformed or truncated. Returns
/// `CoordsError::SerializationError` if individual atom fields cannot be
/// parsed.
pub fn deserialize_assembly(
    bytes: &[u8],
) -> Result<Vec<MoleculeEntity>, CoordsError> {
    if bytes.len() < 12 {
        return Err(CoordsError::InvalidFormat(
            "Data too short for ASSEM01 header".to_owned(),
        ));
    }

    let magic = &bytes[0..8];
    if magic != ASSEMBLY_MAGIC {
        return Err(CoordsError::InvalidFormat(
            "Invalid magic number for ASSEM01".to_owned(),
        ));
    }

    let entity_count =
        u32::from_be_bytes(bytes[8..12].try_into().map_err(|_| {
            CoordsError::InvalidFormat("Invalid entity count".to_owned())
        })?) as usize;

    let (entity_headers, headers_end) =
        parse_entity_headers(bytes, entity_count)?;

    let total_atoms: usize = entity_headers.iter().map(|(_, c)| c).sum();
    if bytes.len() < headers_end + total_atoms * 26 {
        return Err(CoordsError::InvalidFormat(
            "Data too short for atom data".to_owned(),
        ));
    }

    let mut cursor = &bytes[headers_end..];
    let mut entities = Vec::with_capacity(entity_count);

    for (entity_id, (mol_type, atom_count)) in
        entity_headers.into_iter().enumerate()
    {
        let (coords, rest) = read_atoms_to_coords(cursor, atom_count, true)?;
        cursor = rest;
        let kind = coords_to_entity_kind(mol_type, &coords);
        #[allow(clippy::cast_possible_truncation)] // entity ID fits in u32
        entities.push(MoleculeEntity {
            entity_id: entity_id as u32,
            molecule_type: mol_type,
            kind,
        });
    }

    Ok(entities)
}
