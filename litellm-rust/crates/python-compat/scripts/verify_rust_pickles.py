"""Check that CPython unpickles what `pickle::dumps` writes, to the value it was given.

    PYTHON_COMPAT_RUST_PICKLES=rust.tsv cargo test -p litellm-python-compat --test fixtures
    python scripts/verify_rust_pickles.py rust.tsv

The rows are plain data by construction, so this refuses to resolve any class rather than
handing file-controlled bytes to an unrestricted `pickle.loads`.
"""

import ast
import io
import pickle
import sys


class PlainDataUnpickler(pickle.Unpickler):
    """An unpickler with `GLOBAL`/`REDUCE` disabled, mirroring `pickle::loads` in Rust."""

    def find_class(self, module, name):
        raise pickle.UnpicklingError(f"refusing to resolve {module}.{name}")


def loads(data):
    return PlainDataUnpickler(io.BytesIO(data)).load()


def main(path):
    failures = 0
    rows = 0
    with open(path, encoding="utf-8") as lines:
        for line in lines:
            data, expected = line.rstrip("\n").split("\t", 1)
            rows += 1
            actual = repr(loads(bytes.fromhex(data)))
            if actual != repr(ast.literal_eval(expected)):
                failures += 1
                sys.stdout.write(f"mismatch: expected {expected}, got {actual}\n")
    sys.stdout.write(f"{rows} Rust pickles checked, {failures} mismatches\n")
    return 1 if failures or not rows else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
