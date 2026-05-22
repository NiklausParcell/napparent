# napparent (tabular)

Rust workspace for **napparent-tabular** — apparent effect features on Apache Arrow batches.

```bash
cargo add napparent-tabular
```

| Path | Purpose |
|------|---------|
| [`napparent-tabular/`](napparent-tabular/) | Core crate (`cargo add napparent-tabular`) |
| [`napparent-tabular-py/`](napparent-tabular-py/) | Python bindings (PyArrow / maturin) |
| [`paper/`](paper/) | LaTeX theory for the tabular algorithm |
| [`scripts/test_napparent_tabular_bindings.py`](scripts/test_napparent_tabular_bindings.py) | Smoke test (also used in CI) |

```bash
cargo test -p napparent-tabular
maturin develop && python scripts/test_napparent_tabular_bindings.py
```
