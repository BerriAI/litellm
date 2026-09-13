"""The e2e coverage registry: the denominator for e2e test coverage.

`schema.py` defines one validated row per customer-noticeable behavior (a "cell").
The `*.yaml` files hold the rows, one file per id-prefix. `registry.py` loads and
validates them; `collector.py` diffs the registry against the `@pytest.mark.covers`
markers on the live tests and reports coverage per module. See tests/e2e/CLAUDE.md
for the naming grammar.
"""
