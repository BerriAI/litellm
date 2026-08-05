"""Dump the LiteLLM proxy's OpenAPI schema to the path given as the only argument.

Run from the litellm repo root with the proxy dependencies installed:

    python terraform/provider/tools/dump_openapi.py openapi.json
"""

import json
import sys

from litellm.proxy.proxy_server import app


def main(out_path: str) -> None:
    with open(out_path, "w") as f:
        json.dump(app.openapi(), f)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python terraform/provider/tools/dump_openapi.py <out_path>", file=sys.stderr)
        sys.exit(2)
    main(sys.argv[1])
