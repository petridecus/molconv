//! Structured polymer data types — chains, residues, atom sets.

use std::ops::Range;

use super::AtomSet;

/// A single residue within a polymer chain.
#[derive(Debug, Clone)]
pub struct Residue {
    /// 3-character residue name (e.g. b"ALA").
    pub name: [u8; 3],
    /// Residue sequence number.
    pub number: i32,
    /// Index range into the parent `PolymerData.atoms`.
    pub atom_range: Range<usize>,
}

/// A single polymer chain.
#[derive(Debug, Clone)]
pub struct PolymerChain {
    /// Single-character chain identifier (e.g. b'A').
    pub chain_id: u8,
    /// Residues belonging to this chain, in sequence order.
    pub residues: Vec<Residue>,
}

/// Structured polymer data — chains containing residues containing atoms.
#[derive(Debug, Clone)]
pub struct PolymerData {
    /// All atoms across every chain in this polymer.
    pub atoms: AtomSet,
    /// Polymer chains, each containing its own residues.
    pub chains: Vec<PolymerChain>,
}
