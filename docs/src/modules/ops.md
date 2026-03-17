# Ops — Transforms and Analysis

The `ops` module provides coordinate manipulation, structural
validation, and bond inference.

## `ops::transform` — Coordinate Operations

### Filtering

```rust,ignore
// Keep only standard amino acid atoms
let protein = protein_only(&coords);

// Keep only backbone atoms (N, CA, C)
let backbone = backbone_only(&coords);

// Keep only heavy atoms (non-hydrogen)
let heavy = heavy_atoms_only(&coords);

// Filter by residue name predicate
let filtered = filter_residues(&coords, |name| name == "ALA");
```

### Backbone extraction

```rust,ignore
// Extract CA chains as Vec<Vec<Vec3>>
let chains = extract_backbone_chains(&coords);

// Get all CA positions as flat Vec<Vec3>
let cas = extract_ca_positions(&coords);
```

### Alignment

Kabsch algorithm for optimal rigid-body superposition:

```rust,ignore
let (aligned, rmsd) = kabsch_alignment(&mobile, &reference);

// With uniform scaling
let (aligned, rmsd) = kabsch_alignment_with_scale(&mobile, &reference);

// Binary COORDS alignment (convenience wrapper)
let aligned_bytes = align_coords_bytes(&mobile_bytes, &reference_bytes)?;
```

### Interpolation

Smooth coordinate transitions for animation:

```rust,ignore
let interpolated = interpolate_coords(&start, &end, t);  // t in [0, 1]

// Collapse-expand: shrink to centroid then expand to target
let interp = interpolate_coords_collapse(&start, &end, t);
```

### Atom lookup

```rust,ignore
let pos = get_atom_position(&coords, chain_id, res_num, "CA");
let (pos, name) = get_closest_atom_for_residue(&coords, chain, res, point);
```

## `ops::validation` — Completeness Checks

```rust,ignore
let report = completeness_report(&coords);
let has_bb = has_complete_backbone(&coords);
let counts: AtomCounts = atom_counts(&coords);
```

## `ops::bond_inference` — Distance-Based Bonds

Infer covalent bonds from interatomic distances using element-specific
covalent radii:

```rust,ignore
let bonds: Vec<InferredBond> = infer_bonds(&coords, DEFAULT_TOLERANCE);
// InferredBond { atom_a, atom_b, order: BondOrder }
```

Used as a fallback for ligands and cofactors that lack explicit bond
tables.
