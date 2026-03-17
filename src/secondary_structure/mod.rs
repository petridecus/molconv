//! Secondary structure detection and classification.
//!
//! Provides Q3 classification (Helix, Sheet, Coil) with pluggable backends:
//! - `auto`: Fast CA-distance heuristic (no sidechain data needed)
//! - `dssp`: Kabsch-Sander hydrogen-bond based classification + string parsing

pub mod auto;
pub mod dssp;

use glam::Vec3;

/// Q3 secondary structure classification for a single residue.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SSType {
    /// Alpha helix.
    Helix,
    /// Beta sheet / strand.
    Sheet,
    /// Coil / loop (neither helix nor sheet).
    Coil,
}

impl SSType {
    /// Get the color for this SS type (RGB, 0-1 range).
    #[must_use]
    pub fn color(&self) -> [f32; 3] {
        match self {
            SSType::Helix => [0.9, 0.3, 0.5],
            SSType::Sheet => [0.95, 0.85, 0.3],
            SSType::Coil => [0.6, 0.85, 0.6],
        }
    }
}

/// Convert isolated 1-residue helix/sheet runs to Coil.
/// These are too short for ribbon rendering and would leave residues
/// with no backbone geometry.
#[must_use]
pub fn merge_short_segments(ss_types: &[SSType]) -> Vec<SSType> {
    let mut result = ss_types.to_vec();
    for i in 0..result.len() {
        if result[i] != SSType::Coil {
            let prev_same = i > 0 && result[i - 1] == result[i];
            let next_same = i + 1 < result.len() && result[i + 1] == result[i];
            if !prev_same && !next_same {
                result[i] = SSType::Coil;
            }
        }
    }
    result
}

/// Backbone atom positions for a single residue.
/// Used by the DSSP backend for hydrogen-bond energy calculation.
#[derive(Debug, Clone, Copy)]
pub struct BackboneResidue {
    /// Nitrogen position.
    pub n: Vec3,
    /// Alpha-carbon position.
    pub ca: Vec3,
    /// Carbonyl carbon position.
    pub c: Vec3,
    /// Carbonyl oxygen position.
    pub o: Vec3,
}

/// Input data for SS detection fallback in [`resolve`].
pub enum DetectionInput<'a> {
    /// Fast Cα-distance heuristic. Suitable for streaming/animation where
    /// speed matters more than accuracy.
    CaPositions(&'a [Vec3]),
    /// Kabsch-Sander hydrogen-bond analysis. More accurate but requires
    /// full backbone atoms (N, CA, C, O).
    Backbone(&'a [BackboneResidue]),
}

/// Resolve secondary structure assignments for a chain.
///
/// If `ss_override` is provided, uses it directly. Otherwise, detects SS
/// using the method specified by `fallback`. In both cases, isolated
/// 1-residue segments are merged to Coil via [`merge_short_segments`].
///
/// # Usage
///
/// Use [`DetectionInput::CaPositions`] during streaming (e.g. `RFDiffusion3`)
/// for fast updates, and [`DetectionInput::Backbone`] when the final
/// structure is available for more accurate DSSP-based assignment.
#[must_use]
#[allow(
    clippy::needless_pass_by_value,
    reason = "DetectionInput is a small enum of references"
)]
pub fn resolve(
    ss_override: Option<&[SSType]>,
    fallback: DetectionInput<'_>,
) -> Vec<SSType> {
    let raw = ss_override.map_or_else(
        || match fallback {
            DetectionInput::CaPositions(ca) => auto::detect(ca),
            DetectionInput::Backbone(res) => dssp::detect(res),
        },
        <[SSType]>::to_vec,
    );
    merge_short_segments(&raw)
}
