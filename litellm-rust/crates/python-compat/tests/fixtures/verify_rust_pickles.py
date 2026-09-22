"""Check that CPython unpickles what `pickle::dumps` writes, to the value it was given.

PYTHON_COMPAT_RUST_PICKLES=/tmp/rust.tsv cargo test -p litellm-python-compat --test fixtures
python tests/fixtures/verify_rust_pickles.py /tmp/rust.tsv
"""

import ast
import pickle
import sys

failures = 0
rows = 0
with open(sys.argv[1], encoding="utf-8") as lines:
    for line in lines:
        data, expected = line.rstrip("\n").split("\t", 1)
        rows += 1
        actual = repr(pickle.loads(bytes.fromhex(data)))
        if actual != repr(ast.literal_eval(expected)):
            failures += 1
            sys.stdout.write(f"mismatch: expected {expected}, got {actual}\n")
sys.stdout.write(f"{rows} Rust pickles checked, {failures} mismatches\n")
sys.exit(1 if failures or not rows else 0)
