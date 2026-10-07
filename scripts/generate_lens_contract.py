import argparse
import json
from pathlib import Path
from typing import Final

from pydantic import BaseModel, JsonValue

from litellm.proxy.lens.agent_context import Checkpoint
from litellm.proxy.lens.agent_review import Findings
from litellm.proxy.lens.agent_runtime import PythonAgentTurn
from litellm.proxy.lens.agent_workspace import EvidenceReply, EvidenceRequest, PythonRequest
from litellm.proxy.lens.analysis import Candidate, Clusters
from litellm.proxy.lens.models import (
    Claim,
    ExecutionContent,
    Extraction,
    ModelRequest,
    ModelResult,
    Progress,
    Result,
    Sample,
)
from litellm.proxy.lens.reconciliation import FindingGroups
from litellm.proxy.lens.release import PROTOCOL_VERSION

MODELS: Final[tuple[type[BaseModel], ...]] = (
    Claim,
    ExecutionContent,
    Extraction,
    ModelRequest,
    ModelResult,
    Progress,
    Result,
    Sample,
    Candidate,
    Clusters,
    Findings,
    EvidenceRequest,
    PythonRequest,
    EvidenceReply,
    PythonAgentTurn[Extraction],
    PythonAgentTurn[Findings],
    Checkpoint,
    FindingGroups,
)
TARGET: Final = Path(__file__).resolve().parents[1] / "litellm-rust/crates/lens/contract.json"


def draft_seven(value: JsonValue, names: bool = False) -> JsonValue:
    if isinstance(value, list):
        return [draft_seven(item) for item in value]
    if isinstance(value, dict):
        fields: Final = {
            "items" if name == "prefixItems" and not names else name: draft_seven(
                item, not names and name in ("properties", "definitions", "patternProperties")
            )
            for name, item in value.items()
            if names or (name != "title" and not (name == "default" and item is None))
        }
        return fields
    return value


def contract() -> str:
    schemas: Final = tuple(model.model_json_schema(ref_template="#/definitions/{model}") for model in MODELS)
    definitions: Final = {
        **{name: schema for document in schemas for name, schema in document.get("$defs", {}).items()},
        **{
            model.__name__: {key: value for key, value in schema.items() if key != "$defs"}
            for model, schema in zip(MODELS, schemas, strict=True)
        },
    }
    return (
        json.dumps(
            draft_seven(
                {
                    "$schema": "http://json-schema.org/draft-07/schema#",
                    "title": "LensProtocol",
                    "type": "object",
                    "definitions": definitions,
                    "x-lens-protocol-version": PROTOCOL_VERSION,
                }
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def main() -> None:
    parser: Final = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args: Final = parser.parse_args()
    generated: Final = contract()
    if args.check:
        if TARGET.read_text() != generated:
            raise SystemExit("Lens contracts changed; run python scripts/generate_lens_contract.py")
        return
    TARGET.write_text(generated)


if __name__ == "__main__":
    main()
