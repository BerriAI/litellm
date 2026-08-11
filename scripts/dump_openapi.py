"""Dump the proxy's OpenAPI spec as deterministic JSON, read straight off the
FastAPI app object (no server boot, no database).

Reproducibility measures: `.env` loading is disabled and env knobs that brand
or reshape the spec (license, docs branding, root path) are dropped before
importing litellm, so output matches CI byte for byte; info.version is pinned
so release bumps don't churn the committed spec; descriptions are
cleandoc-normalized (Python 3.13 strips docstring indentation at compile time,
3.12 keeps it); keys are sorted.

--include-internal forces include_in_schema on every route; the dashboard type
generator (ui/litellm-dashboard/scripts/gen-api-types.mjs) uses it to type the
internal UI routes hidden from the served /openapi.json.
"""

import argparse
import inspect
import json
import os
from pathlib import Path
from typing import Final

SPEC_SHAPING_ENV_VARS: Final = ("DOCS_TITLE", "DOCS_DESCRIPTION", "LITELLM_LICENSE", "SERVER_ROOT_PATH")


def normalize(node: object) -> object:
    if isinstance(node, dict):
        return {
            k: inspect.cleandoc(v) if k == "description" and isinstance(v, str) else normalize(v)
            for k, v in node.items()
        }
    if isinstance(node, list):
        return [normalize(v) for v in node]
    return node


def main() -> None:
    parser: Final = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("--include-internal", action="store_true")
    args: Final = parser.parse_args()

    for var in SPEC_SHAPING_ENV_VARS:
        os.environ.pop(var, None)

    import dotenv

    dotenv.load_dotenv = lambda *_args, **_kwargs: False

    from fastapi.routing import APIRoute

    from litellm.proxy.proxy_server import app

    if args.include_internal:
        for route in app.routes:
            if isinstance(route, APIRoute):
                route.include_in_schema = True
    app.openapi_schema = None
    spec: Final = app.openapi()
    spec["info"]["version"] = "0.0.0"
    args.out.write_text(json.dumps(normalize(spec), sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
