//! Chemical element enum and associated data (CPK colors, radii, etc.).

/// Chemical element for atoms in a molecular structure.
///
/// Covers biologically-relevant elements found in proteins, nucleic acids,
/// ligands, ions, and waters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Element {
    /// Hydrogen
    H,
    /// Carbon
    C,
    /// Nitrogen
    N,
    /// Oxygen
    O,
    /// Sulfur
    S,
    /// Phosphorus
    P,
    /// Selenium
    Se,
    /// Iron
    Fe,
    /// Zinc
    Zn,
    /// Magnesium
    Mg,
    /// Calcium
    Ca,
    /// Sodium
    Na,
    /// Chlorine
    Cl,
    /// Potassium
    K,
    /// Manganese
    Mn,
    /// Cobalt
    Co,
    /// Nickel
    Ni,
    /// Copper
    Cu,
    /// Bromine
    Br,
    /// Iodine
    I,
    /// Fluorine
    F,
    /// Unrecognized or unsupported element.
    Unknown,
}

impl Element {
    /// Parse element from a 1-2 character symbol string (case-insensitive).
    #[must_use]
    pub fn from_symbol(s: &str) -> Self {
        match s.trim().to_uppercase().as_str() {
            "H" => Element::H,
            "C" => Element::C,
            "N" => Element::N,
            "O" => Element::O,
            "S" => Element::S,
            "P" => Element::P,
            "SE" => Element::Se,
            "FE" => Element::Fe,
            "ZN" => Element::Zn,
            "MG" => Element::Mg,
            "CA" => Element::Ca,
            "NA" => Element::Na,
            "CL" => Element::Cl,
            "K" => Element::K,
            "MN" => Element::Mn,
            "CO" => Element::Co,
            "NI" => Element::Ni,
            "CU" => Element::Cu,
            "BR" => Element::Br,
            "I" => Element::I,
            "F" => Element::F,
            _ => Element::Unknown,
        }
    }

    /// Infer element from a standard protein atom name (e.g., "CA" -> C, "OG"
    /// -> O, "SD" -> S).
    ///
    /// For standard protein atoms, the first alphabetic character reliably
    /// identifies the element. This is the same heuristic used by
    /// `gpu.rs:atom_name_to_type_index`.
    #[must_use]
    pub fn from_atom_name(name: &str) -> Self {
        let name = name.trim();
        // Find first alphabetic character
        name.chars().find(|c| c.is_alphabetic()).map_or(
            Element::Unknown,
            |ch| match ch.to_ascii_uppercase() {
                'C' => Element::C,
                'N' => Element::N,
                'O' => Element::O,
                'S' => Element::S,
                'H' => Element::H,
                'P' => Element::P,
                _ => Element::Unknown,
            },
        )
    }

    /// Standard CPK coloring (Corey-Pauling-Koltun).
    #[must_use]
    pub fn cpk_color(&self) -> [f32; 3] {
        match self {
            Element::H => [1.0, 1.0, 1.0],     // White
            Element::C => [0.4, 0.4, 0.4],     // Dark gray
            Element::N => [0.2, 0.2, 1.0],     // Blue
            Element::O => [1.0, 0.2, 0.2],     // Red
            Element::S => [1.0, 0.85, 0.2],    // Yellow
            Element::P => [1.0, 0.5, 0.0],     // Orange
            Element::Se => [1.0, 0.63, 0.0],   // Orange-yellow
            Element::Fe => [0.56, 0.25, 0.08], // Rust brown
            Element::Zn => [0.49, 0.50, 0.69], // Slate blue
            Element::Mg | Element::Ca => [0.0, 0.55, 0.0], // Dark green
            Element::Na => [0.67, 0.36, 0.95], // Purple
            Element::Cl => [0.12, 0.94, 0.12], // Green
            Element::K => [0.56, 0.25, 0.83],  // Violet
            Element::Mn => [0.61, 0.48, 0.78], // Purple-gray
            Element::Co => [0.94, 0.56, 0.63], // Pink
            Element::Ni => [0.31, 0.82, 0.31], // Green
            Element::Cu => [0.78, 0.50, 0.20], // Copper
            Element::Br => [0.65, 0.16, 0.16], // Dark red
            Element::I => [0.58, 0.0, 0.58],   // Purple
            Element::F => [0.56, 0.88, 0.31],  // Yellow-green
            Element::Unknown => [0.7, 0.7, 0.7], // Light gray
        }
    }

    /// Covalent radius in angstroms (Cambridge CSD values).
    #[must_use]
    pub fn covalent_radius(&self) -> f32 {
        match self {
            Element::H => 0.31,
            Element::C => 0.76,
            Element::N => 0.71,
            Element::O => 0.66,
            Element::S => 1.05,
            Element::P => 1.07,
            Element::Se | Element::Br => 1.20,
            Element::Fe | Element::Cu => 1.32,
            Element::Zn => 1.22,
            Element::Mg => 1.41,
            Element::Ca => 1.76,
            Element::Na => 1.66,
            Element::Cl => 1.02,
            Element::K => 2.03,
            Element::Mn | Element::I => 1.39,
            Element::Co => 1.26,
            Element::Ni => 1.24,
            Element::F => 0.57,
            Element::Unknown => 0.77,
        }
    }

    /// Van der Waals radius in angstroms.
    #[must_use]
    pub fn vdw_radius(&self) -> f32 {
        match self {
            Element::H => 1.20,
            Element::C | Element::Unknown => 1.70,
            Element::N => 1.55,
            Element::O => 1.52,
            Element::S | Element::P => 1.80,
            Element::Se => 1.90,
            Element::Fe | Element::Mn | Element::Co => 2.00,
            Element::Zn => 1.39,
            Element::Mg => 1.73,
            Element::Ca => 2.31,
            Element::Na => 2.27,
            Element::Cl => 1.75,
            Element::K => 2.75,
            Element::Ni => 1.63,
            Element::Cu => 1.40,
            Element::Br => 1.85,
            Element::I => 1.98,
            Element::F => 1.47,
        }
    }

    /// Two-character symbol (padded with space if single char).
    #[must_use]
    pub fn symbol(&self) -> &'static str {
        match self {
            Element::H => "H",
            Element::C => "C",
            Element::N => "N",
            Element::O => "O",
            Element::S => "S",
            Element::P => "P",
            Element::Se => "Se",
            Element::Fe => "Fe",
            Element::Zn => "Zn",
            Element::Mg => "Mg",
            Element::Ca => "Ca",
            Element::Na => "Na",
            Element::Cl => "Cl",
            Element::K => "K",
            Element::Mn => "Mn",
            Element::Co => "Co",
            Element::Ni => "Ni",
            Element::Cu => "Cu",
            Element::Br => "Br",
            Element::I => "I",
            Element::F => "F",
            Element::Unknown => "X",
        }
    }
}
