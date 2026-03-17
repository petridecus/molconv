//! Coordinate interpolation utilities.

use glam::Vec3;

use crate::types::coords::{Coords, CoordsAtom};

/// Linear interpolation between two Coords instances.
#[must_use]
pub fn interpolate_coords(
    start: &Coords,
    end: &Coords,
    t: f32,
) -> Option<Coords> {
    if start.num_atoms != end.num_atoms {
        return None;
    }

    let t = t.clamp(0.0, 1.0);
    let one_minus_t = 1.0 - t;

    let atoms = start
        .atoms
        .iter()
        .zip(end.atoms.iter())
        .map(|(s, e)| CoordsAtom {
            x: one_minus_t.mul_add(s.x, e.x * t),
            y: one_minus_t.mul_add(s.y, e.y * t),
            z: one_minus_t.mul_add(s.z, e.z * t),
            occupancy: one_minus_t.mul_add(s.occupancy, e.occupancy * t),
            b_factor: one_minus_t.mul_add(s.b_factor, e.b_factor * t),
        })
        .collect();

    Some(Coords {
        num_atoms: start.num_atoms,
        atoms,
        chain_ids: start.chain_ids.clone(),
        res_names: start.res_names.clone(),
        res_nums: start.res_nums.clone(),
        atom_names: start.atom_names.clone(),
        elements: start.elements.clone(),
    })
}

/// Interpolate Coords with a collapse/expand effect through a collapse point.
#[must_use]
pub fn interpolate_coords_collapse<F>(
    start: &Coords,
    end: &Coords,
    t: f32,
    collapse_fn: F,
) -> Option<Coords>
where
    F: Fn(i32, u8) -> Vec3,
{
    if start.num_atoms != end.num_atoms {
        return None;
    }

    let t = t.clamp(0.0, 1.0);

    let atoms = start
        .atoms
        .iter()
        .zip(end.atoms.iter())
        .enumerate()
        .map(|(i, (s, e))| {
            let start_pos = Vec3::new(s.x, s.y, s.z);
            let end_pos = Vec3::new(e.x, e.y, e.z);
            let collapse_point =
                collapse_fn(start.res_nums[i], start.chain_ids[i]);

            let interpolated = if t < 0.5 {
                let phase_t = t * 2.0;
                start_pos.lerp(collapse_point, phase_t)
            } else {
                let phase_t = (t - 0.5) * 2.0;
                collapse_point.lerp(end_pos, phase_t)
            };

            CoordsAtom {
                x: interpolated.x,
                y: interpolated.y,
                z: interpolated.z,
                occupancy: (1.0 - t).mul_add(s.occupancy, e.occupancy * t),
                b_factor: (1.0 - t).mul_add(s.b_factor, e.b_factor * t),
            }
        })
        .collect();

    Some(Coords {
        num_atoms: start.num_atoms,
        atoms,
        chain_ids: start.chain_ids.clone(),
        res_names: start.res_names.clone(),
        res_nums: start.res_nums.clone(),
        atom_names: start.atom_names.clone(),
        elements: start.elements.clone(),
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::types::coords::Element;

    #[test]
    fn test_interpolate_coords() {
        let start = Coords {
            num_atoms: 2,
            atoms: vec![
                CoordsAtom {
                    x: 0.0,
                    y: 0.0,
                    z: 0.0,
                    occupancy: 1.0,
                    b_factor: 0.0,
                },
                CoordsAtom {
                    x: 1.0,
                    y: 0.0,
                    z: 0.0,
                    occupancy: 1.0,
                    b_factor: 0.0,
                },
            ],
            chain_ids: vec![b'A', b'A'],
            res_names: vec![*b"ALA", *b"ALA"],
            res_nums: vec![1, 1],
            atom_names: vec![*b"N   ", *b"CA  "],
            elements: vec![Element::Unknown; 2],
        };

        let end = Coords {
            num_atoms: 2,
            atoms: vec![
                CoordsAtom {
                    x: 0.0,
                    y: 10.0,
                    z: 0.0,
                    occupancy: 1.0,
                    b_factor: 0.0,
                },
                CoordsAtom {
                    x: 1.0,
                    y: 10.0,
                    z: 0.0,
                    occupancy: 1.0,
                    b_factor: 0.0,
                },
            ],
            chain_ids: vec![b'A', b'A'],
            res_names: vec![*b"ALA", *b"ALA"],
            res_nums: vec![1, 1],
            atom_names: vec![*b"N   ", *b"CA  "],
            elements: vec![Element::Unknown; 2],
        };

        let mid = interpolate_coords(&start, &end, 0.5).unwrap();
        assert!((mid.atoms[0].y - 5.0).abs() < 0.001);
        assert!((mid.atoms[1].y - 5.0).abs() < 0.001);

        let at_start = interpolate_coords(&start, &end, 0.0).unwrap();
        assert!((at_start.atoms[0].y - 0.0).abs() < 0.001);

        let at_end = interpolate_coords(&start, &end, 1.0).unwrap();
        assert!((at_end.atoms[0].y - 10.0).abs() < 0.001);
    }
}
