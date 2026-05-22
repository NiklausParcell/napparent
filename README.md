# napparent (tabular)

Rust workspace for **napparent-tabular** — apparent effect features on Apache Arrow batches.

| Path | Purpose |
|------|---------|
| [`napparent-tabular/`](napparent-tabular/) | Core crate (`cargo add napparent-tabular`) |
| [`napparent-tabular-py/`](napparent-tabular-py/) | Python bindings (PyArrow / maturin) |
| [`paper/`](paper/) | LaTeX theory for the tabular algorithm |
| [`reference/`](reference/) | Python reference implementation |
| [`scripts/`](scripts/) | Smoke tests and dev helpers |

```bash
cargo test -p napparent-tabular
maturin develop && python scripts/test_napparent_tabular_bindings.py
```
