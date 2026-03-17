//! Data types for volumetric density maps (MRC/CCP4 format).

use ndarray::Array3;

/// Parsed volumetric density map.
///
/// Grid data is stored in spatial XYZ order after axis reordering from the
/// file's column/row/section layout. The `data` array is indexed as
/// `data[[x, y, z]]`.
#[derive(Debug, Clone)]
pub struct DensityMap {
    /// Grid dimension along X after axis reordering.
    pub nx: usize,
    /// Grid dimension along Y after axis reordering.
    pub ny: usize,
    /// Grid dimension along Z after axis reordering.
    pub nz: usize,

    /// Grid start index along X.
    pub nxstart: i32,
    /// Grid start index along Y.
    pub nystart: i32,
    /// Grid start index along Z.
    pub nzstart: i32,

    /// Unit cell sampling intervals along X.
    pub mx: usize,
    /// Unit cell sampling intervals along Y.
    pub my: usize,
    /// Unit cell sampling intervals along Z.
    pub mz: usize,

    /// Unit cell dimensions a, b, c in Angstroms.
    pub cell_dims: [f32; 3],
    /// Unit cell angles alpha, beta, gamma in degrees.
    pub cell_angles: [f32; 3],

    /// Minimum density value.
    pub dmin: f32,
    /// Maximum density value.
    pub dmax: f32,
    /// Mean density value.
    pub dmean: f32,
    /// RMS deviation from mean density.
    pub rms: f32,

    /// Origin in Angstroms (MRC2014 words 50-52).
    pub origin: [f32; 3],

    /// Space group number.
    pub space_group: u32,

    /// 3D grid of density values, indexed as `data[[x, y, z]]`.
    pub data: Array3<f32>,
}

impl DensityMap {
    /// Angstroms per voxel along each axis: `[cell_a/mx, cell_b/my,
    /// cell_c/mz]`.
    #[must_use]
    pub fn voxel_size(&self) -> [f32; 3] {
        #[allow(clippy::cast_precision_loss)]
        // grid dimensions fit comfortably in f32
        [
            self.cell_dims[0] / self.mx as f32,
            self.cell_dims[1] / self.my as f32,
            self.cell_dims[2] / self.mz as f32,
        ]
    }

    /// Convert a grid index to Cartesian coordinates in Angstroms.
    ///
    /// Accounts for both `origin` (MRC2014 words 50-52) and
    /// `nxstart/nystart/nzstart`.
    #[must_use]
    pub fn grid_to_cartesian(
        &self,
        ix: usize,
        iy: usize,
        iz: usize,
    ) -> [f32; 3] {
        let vs = self.voxel_size();
        #[allow(clippy::cast_precision_loss)]
        // grid indices and start offsets fit in f32
        [
            (self.nxstart as f32 + ix as f32).mul_add(vs[0], self.origin[0]),
            (self.nystart as f32 + iy as f32).mul_add(vs[1], self.origin[1]),
            (self.nzstart as f32 + iz as f32).mul_add(vs[2], self.origin[2]),
        ]
    }

    /// Density threshold at a given sigma level: `dmean + sigma * rms`.
    #[must_use]
    pub fn sigma_level(&self, sigma: f32) -> f32 {
        sigma.mul_add(self.rms, self.dmean)
    }
}

/// Errors that can occur when parsing a density map.
#[derive(Debug, thiserror::Error)]
pub enum DensityError {
    /// The file header or data layout is invalid.
    #[error("invalid density map format: {0}")]
    InvalidFormat(String),

    /// The MRC data mode is not supported (only mode 2 / float32).
    #[error("unsupported MRC data mode: {0}")]
    UnsupportedMode(i32),

    /// An I/O error occurred while reading the map file.
    #[error(transparent)]
    Io(#[from] std::io::Error),
}
