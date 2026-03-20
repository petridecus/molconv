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
        [
            self.cell_dims[0] / self.mx as f32,
            self.cell_dims[1] / self.my as f32,
            self.cell_dims[2] / self.mz as f32,
        ]
    }

    /// Build the 3x3 deorthogonalization matrix for converting
    /// fractional unit cell coordinates to Cartesian Angstroms.
    ///
    /// For orthogonal cells (α=β=γ=90°) this is diagonal with (a,b,c).
    /// For non-orthogonal cells (e.g. hexagonal γ=120°) the off-diagonal
    /// terms rotate the fractional axes into Cartesian space.
    #[must_use]
    pub fn frac_to_cart_matrix(&self) -> [[f32; 3]; 3] {
        let [a, b, c] = self.cell_dims;
        let alpha = self.cell_angles[0].to_radians();
        let beta = self.cell_angles[1].to_radians();
        let gamma = self.cell_angles[2].to_radians();

        let cos_a = alpha.cos();
        let cos_b = beta.cos();
        let cos_g = gamma.cos();
        let sin_g = gamma.sin();

        let xi = cos_b.mul_add(-cos_g, cos_a) / sin_g;
        let sin_b = beta.sin();
        let zeta = sin_b.mul_add(sin_b, -(xi * xi)).max(0.0).sqrt();

        [
            [a, b * cos_g, c * cos_b],
            [0.0, b * sin_g, c * xi],
            [0.0, 0.0, c * zeta],
        ]
    }

    /// Convert a grid index to Cartesian coordinates in Angstroms.
    ///
    /// Converts grid indices to fractional unit cell coordinates, then
    /// applies the deorthogonalization matrix to get Cartesian positions.
    /// Accounts for `nxstart/nystart/nzstart` and `origin`.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn grid_to_cartesian(
        &self,
        ix: usize,
        iy: usize,
        iz: usize,
    ) -> [f32; 3] {
        self.grid_to_cartesian_f32(ix as f32, iy as f32, iz as f32)
    }

    /// Float-precision version of
    /// [`grid_to_cartesian`](Self::grid_to_cartesian).
    ///
    /// Accepts fractional grid positions for sub-voxel interpolation
    /// (e.g. from marching cubes edge interpolation).
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn grid_to_cartesian_f32(&self, gx: f32, gy: f32, gz: f32) -> [f32; 3] {
        let fx = (self.nxstart as f32 + gx) / self.mx as f32;
        let fy = (self.nystart as f32 + gy) / self.my as f32;
        let fz = (self.nzstart as f32 + gz) / self.mz as f32;

        let m = self.frac_to_cart_matrix();
        [
            m[0][0].mul_add(fx, m[0][1].mul_add(fy, m[0][2] * fz))
                + self.origin[0],
            m[1][1].mul_add(fy, m[1][2] * fz) + self.origin[1],
            m[2][2].mul_add(fz, self.origin[2]),
        ]
    }

    /// Convert Cartesian coordinates in Angstroms back to fractional
    /// grid indices.
    ///
    /// Inverse of
    /// [`grid_to_cartesian_f32`](Self::grid_to_cartesian_f32).
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn cartesian_to_grid(&self, cart: [f32; 3]) -> [f32; 3] {
        let cx = cart[0] - self.origin[0];
        let cy = cart[1] - self.origin[1];
        let cz = cart[2] - self.origin[2];

        let m = self.frac_to_cart_matrix();
        let fz = cz / m[2][2];
        let fy = m[1][2].mul_add(-fz, cy) / m[1][1];
        let fx = m[0][2].mul_add(-fz, m[0][1].mul_add(-fy, cx)) / m[0][0];

        [
            fx.mul_add(self.mx as f32, -(self.nxstart as f32)),
            fy.mul_add(self.my as f32, -(self.nystart as f32)),
            fz.mul_add(self.mz as f32, -(self.nzstart as f32)),
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
