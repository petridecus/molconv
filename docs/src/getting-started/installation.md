# Installation

## As a Rust dependency

Add to your `Cargo.toml`:

```toml
[dependencies]
molconv = { git = "https://github.com/petridecus/molconv" }
```

## As a Python package

molconv provides optional Python bindings via PyO3/maturin.

```bash
# Build and install the wheel
cd crates/molconv
maturin develop --release --features python

# Verify
python -c "import molconv; print('OK')"
```

## Feature flags

| Feature  | Description |
|----------|-------------|
| `python` | Enable PyO3 bindings (requires `pyo3`, `numpy`) |

The default build includes no optional features — it's a pure Rust
library with no native dependencies beyond `glam` and `pdbtbx`.
