# /// script
# requires-python = ">=3.10"
# dependencies = ["datamodel-code-generator==0.66.0", "ruff==0.15.3"]
# ///
from __future__ import annotations

import argparse
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
GENERATED: Final = ROOT / "litellm/rust_bridge/trace/generated"
SCHEMAS: Final = TypeAdapter(dict[str, dict[str, JsonValue]])


class Arguments(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    check: bool


class GeneratorConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    version: str
    options: tuple[str, ...]


def export(crate: str, extra_args: tuple[str, ...] = ()) -> Mapping[str, Mapping[str, JsonValue]]:
    result: Final = subprocess.run(
        (
            "cargo",
            "run",
            "--locked",
            "--manifest-path",
            str(ROOT / "litellm-rust/Cargo.toml"),
            "-p",
            f"litellm-{crate}",
            "--bin",
            f"export-{crate}-schema",
            "--features",
            "schema",
            *(("--", *extra_args) if extra_args else ()),
        ),
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    return MappingProxyType(SCHEMAS.validate_json(result.stdout))


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
            "--enable-faux-immutability",
            "--additional-imports",
            "collections.abc.Mapping,typing.TypeAlias",
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


def reconcile_schemas(expected: frozenset[Path], check: bool) -> bool:
    obsolete: Final = tuple(path for path in (TOOLING / "schemas").rglob("*.json") if path not in expected)
    if check:
        for path in obsolete:
            sys.stderr.write(f"obsolete: {path.relative_to(ROOT)}\n")
        return not obsolete
    for path in obsolete:
        path.unlink()
    return True


def main() -> int:
    parser: Final = argparse.ArgumentParser(description="Regenerate trace schemas and Python wire contracts")
    parser.add_argument("--check", action="store_true", help="compare fresh schemas and Python with committed files")
    args: Final = Arguments.model_validate(vars(parser.parse_args()))
    config: Final = GeneratorConfig.model_validate_json((TOOLING / "config.json").read_text())
    if version("datamodel-code-generator") != config.version:
        sys.stderr.write(f"requires datamodel-code-generator=={config.version}\n")
        return 1
    domain: Final = export("traces")
    requests: Final = export("traces", ("--requests",))
    clickhouse: Final = export("traces-clickhouse")
    exported: Final = tuple(schema_files(domain, clickhouse, requests))
    schema_results: Final = tuple(publish(path, content, args.check) for path, content in exported)
    schema_set_matches: Final = reconcile_schemas(frozenset(path for path, _ in exported), args.check)
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
        python_results: Final = (
            publish(GENERATED / "types.py", types.read_text(), args.check),
            publish(GENERATED / "models.py", models.read_text(), args.check),
            publish(GENERATED / "requests.py", request_models.read_text(), args.check),
        )
    return 0 if all((schema_set_matches, *schema_results, *python_results)) else 1


def schema_files(
    domain: Mapping[str, Mapping[str, JsonValue]],
    clickhouse: Mapping[str, Mapping[str, JsonValue]],
    requests: Mapping[str, Mapping[str, JsonValue]],
) -> Iterator[tuple[Path, str]]:
    for crate, schemas in (("traces", domain), ("traces-clickhouse", clickhouse), ("traces", requests)):
        for name, schema in schemas.items():
            yield TOOLING / "schemas" / crate / f"{name}.json", json.dumps(schema, indent=2, sort_keys=True) + "\n"


if __name__ == "__main__":
    sys.exit(main())
