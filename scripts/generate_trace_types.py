# /// script
# requires-python = ">=3.10"
# dependencies = ["datamodel-code-generator==0.66.0", "ruff==0.15.3"]
# ///
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections.abc import Iterator, Mapping
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory
from types import MappingProxyType
from typing import Final

from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter

ROOT: Final = Path(__file__).resolve().parents[1]
TOOLING: Final = ROOT / "scripts/trace_codegen"
GENERATED: Final = ROOT / "litellm/tracing/generated"
ASSETS: Final = ROOT / "scripts/lens_assets"
SCHEMAS: Final = TypeAdapter(dict[str, dict[str, JsonValue]])


class Arguments(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    check: bool


class GeneratorConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    version: str
    options: tuple[str, ...]


class LensSource(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    repository: str
    revision: str
    schema_groups: Mapping[str, Mapping[str, str]]
    fixtures: Mapping[str, Mapping[str, str]]


def frozen_schemas(group: str, source: LensSource) -> Mapping[str, Mapping[str, JsonValue]]:
    return MappingProxyType(dict(frozen_schema(path, digest) for path, digest in source.schema_groups[group].items()))


def frozen_schema(relative_path: str, expected_digest: str) -> tuple[str, Mapping[str, JsonValue]]:
    path: Final = ASSETS / relative_path
    content: Final = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != expected_digest:
        raise ValueError(f"Lens contract checksum mismatch: {relative_path}")
    return path.stem, TypeAdapter(dict[str, JsonValue]).validate_json(content)


def definitions(schemas: Mapping[str, Mapping[str, JsonValue]]) -> Iterator[tuple[str, Mapping[str, JsonValue]]]:
    for name, schema in schemas.items():
        if name == "Tenant":
            continue
        yield from SCHEMAS.validate_python(schema.get("$defs", {})).items()
        yield name, {key: value for key, value in schema.items() if key not in ("$defs", "$schema")}


def generate(
    schemas: Mapping[str, Mapping[str, JsonValue]],
    mode: str,
    directory: Path,
    config: GeneratorConfig,
) -> Path:
    input_path: Final = directory / f"{mode}.json"
    output_path: Final = directory / f"{mode}.py"
    input_path.write_text(
        json.dumps(
            {
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "title": f"TraceWire{mode.title()}",
                "anyOf": [{"$ref": f"#/$defs/{name}"} for name in schemas if name != "Tenant"],
                "$defs": dict(definitions(schemas)),
            },
            indent=2,
        )
        + "\n"
    )
    specific: Final = (
        (
            "--output-model-type",
            "typing.TypedDict",
            "--additional-imports",
            (
                "collections.abc.Mapping,typing.Annotated,pydantic.Field,typing_extensions.ReadOnly,"
                "typing_extensions.NotRequired,typing_extensions"
            ),
        )
        if mode == "types"
        else (
            "--output-model-type",
            "pydantic_v2.BaseModel",
            "--base-class",
            "litellm.types.llms.base.LiteLLMBaseModel",
            "--enable-faux-immutability",
            "--additional-imports",
            "collections.abc.Mapping,typing.TypeAlias,pydantic.JsonValue",
        )
    )
    subprocess.run(
        (
            sys.executable,
            "-m",
            "datamodel_code_generator",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--custom-template-dir",
            str(TOOLING / "templates"),
            *config.options,
            *specific,
        ),
        check=True,
    )
    subprocess.run(
        (sys.executable, "-m", "ruff", "check", "--select", "I,F401", "--fix", str(output_path)),
        check=True,
        stdout=subprocess.DEVNULL,
    )
    subprocess.run(
        (sys.executable, "-m", "ruff", "format", "--line-length", "120", str(output_path)),
        check=True,
        stdout=subprocess.DEVNULL,
    )
    return output_path


def publish(path: Path, content: str, check: bool) -> bool:
    if path.exists() and path.read_text() == content:
        return True
    if check:
        sys.stderr.write(f"stale: {path.relative_to(ROOT)}\n")
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return True


def main() -> int:
    parser: Final = argparse.ArgumentParser(description="Regenerate Python HTTP contracts from pinned Lens schemas")
    parser.add_argument("--check", action="store_true", help="compare generated Python with committed files")
    args: Final = Arguments.model_validate(vars(parser.parse_args()))
    config: Final = GeneratorConfig.model_validate_json((TOOLING / "config.json").read_text())
    if version("datamodel-code-generator") != config.version:
        sys.stderr.write(f"requires datamodel-code-generator=={config.version}\n")
        return 1
    source: Final = LensSource.model_validate_json((ASSETS / "source.json").read_text())
    for path, entry in source.fixtures.items():
        if hashlib.sha256((ASSETS / path).read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError(f"Lens fixture checksum mismatch: {path}")
    domain: Final = frozen_schemas("types", source)
    requests: Final = frozen_schemas("requests", source)
    responses: Final = frozen_schemas("responses", source)
    clickhouse: Final = frozen_schemas("models", source)
    with TemporaryDirectory(prefix="trace-codegen-") as temporary:
        directory: Final = Path(temporary)
        types: Final = generate({**domain, "ReadQueryName": clickhouse["ReadQueryName"]}, "types", directory, config)
        models: Final = generate(
            {name: schema for name, schema in clickhouse.items() if name != "ReadQueryName"},
            "models",
            directory,
            config,
        )
        request_models: Final = generate(requests, "requests", directory, config)
        response_models: Final = generate(responses, "responses", directory, config)
        python_results: Final = (
            publish(GENERATED / "types.py", types.read_text(), args.check),
            publish(GENERATED / "models.py", models.read_text(), args.check),
            publish(GENERATED / "requests.py", request_models.read_text(), args.check),
            publish(GENERATED / "responses.py", response_models.read_text(), args.check),
        )
    return 0 if all(python_results) else 1


if __name__ == "__main__":
    sys.exit(main())
