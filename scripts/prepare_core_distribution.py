"""Select core metadata in an isolated source checkout before building."""

import argparse
import json
import re
from pathlib import Path
from typing import Final

import tomllib


def core_metadata(text: str) -> str:
    metadata: Final = tomllib.loads(text)
    if metadata["project"]["name"] != "litellm":
        raise ValueError("Prepare core from an unmodified litellm source checkout")
    dependencies: Final = metadata["tool"]["litellm"]["core"]["dependencies"]
    if not isinstance(dependencies, list) or not all(isinstance(value, str) for value in dependencies):
        raise ValueError("Core dependencies must be an explicit list of requirements")
    project: Final = re.search(r"(?ms)^\[project\]\n(.*?)(?=^\[|\Z)", text)
    if project is None:
        raise ValueError("Missing project metadata")
    name: Final = re.sub(r'^name\s*=\s*"litellm"', 'name = "litellm-core"', project.group(1), count=1, flags=re.M)
    body, count = re.subn(
        r"(?ms)^dependencies\s*=\s*\[.*?^\]",
        "dependencies = [\n" + "".join(f"    {json.dumps(value)},\n" for value in dependencies) + "]",
        name,
        count=1,
    )
    if count != 1:
        raise ValueError("Expected a multiline project dependency list")
    selected: Final = text[: project.start(1)] + body + text[project.end(1) :]
    legacy_extras: Final = metadata["tool"]["litellm"]["core"].get("legacy-only-extras", ())
    extras: Final = (
        "[project.optional-dependencies]\n"
        + "".join(
            f"{json.dumps(extra)} = {json.dumps(requirements)}\n"
            for extra, requirements in metadata["project"].get("optional-dependencies", {}).items()
            if extra not in legacy_extras
        )
        + "\n"
    )
    independent: Final = re.sub(r"(?ms)^\[project.optional-dependencies\]\n.*?(?=^\[|\Z)", lambda _: extras, selected)
    ui_path: Final = "litellm/proxy/_experimental/out"
    maturin: Final = dict(metadata["tool"]["maturin"])
    maturin["include"] = [path for path in maturin.get("include", ()) if path != f"{ui_path}/**"]
    maturin["exclude"] = list(dict.fromkeys([*maturin.get("exclude", ()), ui_path, f"{ui_path}/**"]))
    build_metadata: Final = (
        "[tool.maturin]\n" + "".join(f"{key} = {json.dumps(value)}\n" for key, value in maturin.items()) + "\n"
    )
    core_artifacts: Final = re.sub(r"(?ms)^\[tool.maturin\]\n.*?(?=^\[|\Z)", lambda _: build_metadata, independent)
    return core_artifacts.replace('"litellm[', '"litellm-core[')


def prepare_core_distribution(path: Path) -> None:
    path.write_text(core_metadata(path.read_text()))


if __name__ == "__main__":
    parser: Final = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pyproject", type=Path)
    prepare_core_distribution(parser.parse_args().pyproject)
