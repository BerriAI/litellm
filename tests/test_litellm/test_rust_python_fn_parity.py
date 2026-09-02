"""Enforce Rust <-> Python parity for registered SDK routes in litellm-rust.

litellm-rust is a port of the Python SDK. Registering a route (e.g. `ocr`,
`messages`) in PARITY_ROUTES asserts the whole trace maps 1:1:

1. the Python SDK entrypoint exists (e.g. `litellm.ocr`),
2. the Rust core entrypoint exists (`pub async fn <route>` in
   `crates/core/src/<route>/mod.rs`, per litellm-rust/ADDING_A_PROVIDER.md),
3. every top-level `pub fn` in each registered provider transformation file
   under that route resolves to a Python counterpart *by naming convention* —
   there is no per-function mapping file:

   - exact:            `map_ocr_params`   -> `map_ocr_params`
   - get_-strip:       `complete_url`     -> `get_complete_url` (clippy drops `get_`)
   - resolve_ -> get_: `resolve_api_key`  -> `get_api_key`
   - provider-infix:   `complete_anthropic_url` (in providers/anthropic/)
                       -> `complete_url` -> `get_complete_url`

Irregular functions are annotated in the Rust source directly above the fn:

    // python-parity: get_llm_provider   <- the Python twin's real name
    // rust-only: <reason>               <- no Python counterpart exists

Every other Rust file with top-level pub fns must be listed in
PENDING_REGISTRATION, so a new Rust module fails CI until its author either
registers it for parity or explicitly defers it.
"""

import importlib
import inspect
import re
from dataclasses import dataclass
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parents[2]
RUST_ROOT = REPO_ROOT / "litellm-rust"
RUST_CORE_SRC = RUST_ROOT / "crates" / "core" / "src"

PYTHON_PARITY_MARKER = "python-parity:"
RUST_ONLY_MARKER = "rust-only:"

PUB_FN_RE = re.compile(r"^pub (?:async )?fn ([A-Za-z0-9_]+)")


@dataclass(frozen=True)
class RouteParity:
    """One top-level SDK route under parity enforcement."""

    # (module, attr) of the Python SDK entrypoint, e.g. ("litellm", "ocr").
    python_entrypoint: tuple[str, str]
    # Whether crates/core/src/<route>/mod.rs must define `pub async fn <route>`.
    # False documents a known gap (entrypoint not yet ported to core).
    rust_core_entrypoint: bool
    # provider -> Python module holding that provider's transformation for this
    # route. Function names inside are matched by convention, not listed here.
    providers: dict[str, str]


PARITY_ROUTES: dict[str, RouteParity] = {
    "messages": RouteParity(
        python_entrypoint=("litellm.anthropic_interface.messages", "create"),
        rust_core_entrypoint=True,
        providers={
            "anthropic": "litellm.llms.anthropic.experimental_pass_through.messages.transformation",
            # azure_ai: pending — fn names need parity markers first
        },
    ),
    "ocr": RouteParity(
        python_entrypoint=("litellm", "ocr"),
        # Known gap: the OCR entrypoint still lives in ai-gateway, not core.
        rust_core_entrypoint=False,
        providers={
            "mistral": "litellm.llms.mistral.ocr.transformation",
            # azure_ai, vertex_ai: pending — fn names need parity markers first
        },
    ),
}

# Route-independent helpers checked file -> python module, same fn convention.
SHARED_HELPERS: dict[str, str] = {
    "crates/core/src/routing_utils/provider.rs": "litellm.litellm_core_utils.get_llm_provider_logic",
}

# Rust files with top-level pub fns not yet under parity enforcement. Removing
# an entry means registering it above; adding a new Rust module with top-level
# pub fns requires choosing a list.
PENDING_REGISTRATION = {
    "crates/core/src/chat_completions/conversation.rs",
    "crates/core/src/chat_completions/mod.rs",
    "crates/core/src/chat_completions/response_utils.rs",
    "crates/core/src/chat_completions/transformation.rs",
    "crates/core/src/error.rs",
    "crates/core/src/http_utils.rs",
    "crates/core/src/providers/azure_ai/messages/transformation.rs",
    "crates/core/src/providers/azure_ai/ocr/transformation.rs",
    "crates/core/src/providers/bedrock/aws_base.rs",
    "crates/core/src/providers/openai/realtime/transformation.rs",
    "crates/core/src/providers/vertex_ai/ocr/transformation.rs",
    "crates/core/src/responses/websocket.rs",
    "crates/core/src/router/strategy/simple_shuffle.rs",
}


def _rust_relpath(path: Path) -> str:
    return path.relative_to(RUST_ROOT).as_posix()


def _provider_from_relpath(relpath: str) -> str | None:
    parts = relpath.split("/")
    if "providers" in parts:
        return parts[parts.index("providers") + 1]
    return None


def _top_level_pub_fns(path: Path) -> list[tuple[str, list[str]]]:
    """(fn_name, marker_lines) for each column-0 pub fn in the file.

    marker_lines are the contiguous `//` / `#[...]` lines directly above the
    fn, where `// python-parity:` and `// rust-only:` annotations live.
    """
    lines = path.read_text().splitlines()
    results: list[tuple[str, list[str]]] = []
    for i, line in enumerate(lines):
        match = PUB_FN_RE.match(line)
        if match is None:
            continue
        markers: list[str] = []
        j = i - 1
        while j >= 0 and (lines[j].startswith("//") or lines[j].startswith("#[")):
            markers.append(lines[j])
            j -= 1
        results.append((match.group(1), markers))
    return results


def _explicit_python_name(markers: list[str]) -> str | None:
    for line in markers:
        if PYTHON_PARITY_MARKER in line:
            return line.split(PYTHON_PARITY_MARKER, 1)[1].strip()
    return None


def _is_rust_only(markers: list[str]) -> bool:
    return any(RUST_ONLY_MARKER in line for line in markers)


def _candidate_names(fn_name: str, provider: str | None) -> list[str]:
    base_names: list[str] = [fn_name]
    if provider:
        stripped = fn_name.replace(f"_{provider}", "").replace(f"{provider}_", "")
        if stripped and stripped != fn_name:
            base_names.append(stripped)
    candidates: list[str] = []
    for name in base_names:
        candidates.append(name)
        candidates.append(f"get_{name}")
        if name.startswith("resolve_"):
            candidates.append("get_" + name[len("resolve_") :])
    return list(dict.fromkeys(candidates))


def _python_surface_has(python_module: str, name: str) -> bool:
    module = importlib.import_module(python_module)
    if hasattr(module, name):
        return True
    for _, cls in inspect.getmembers(module, inspect.isclass):
        if cls.__module__ == module.__name__ and hasattr(cls, name):
            return True
    return False


def _assert_fns_have_python_counterparts(rust_relpath: str, python_module: str):
    rust_path = RUST_ROOT / rust_relpath
    provider = _provider_from_relpath(rust_relpath)
    failures: list[str] = []

    for fn_name, markers in _top_level_pub_fns(rust_path):
        if _is_rust_only(markers):
            continue
        explicit = _explicit_python_name(markers)
        candidates = [explicit] if explicit else _candidate_names(fn_name, provider)
        if not any(_python_surface_has(python_module, c) for c in candidates):
            failures.append(f"  {fn_name}: tried {candidates} in {python_module}")

    assert not failures, (
        f"{rust_relpath}: top-level pub fns without a Python counterpart:\n"
        + "\n".join(failures)
        + "\nFix by renaming to match the convention, adding a "
        f"`// {PYTHON_PARITY_MARKER} <name>` marker for an irregular twin, or "
        f"`// {RUST_ONLY_MARKER} <reason>` if no Python counterpart exists."
    )


def _registered_provider_files() -> dict[str, str]:
    files: dict[str, str] = {}
    for route, parity in PARITY_ROUTES.items():
        for provider, python_module in parity.providers.items():
            relpath = f"crates/core/src/providers/{provider}/{route}/transformation.rs"
            files[relpath] = python_module
    return files


def _route_mod_files() -> set[str]:
    return {
        f"crates/core/src/{route}/mod.rs"
        for route, parity in PARITY_ROUTES.items()
        if parity.rust_core_entrypoint
    }


def _rust_files_with_top_level_pub_fns() -> set[str]:
    files: set[str] = set()
    for path in sorted(RUST_CORE_SRC.rglob("*.rs")):
        if path.name == "tests.rs":
            continue
        if _top_level_pub_fns(path):
            files.add(_rust_relpath(path))
    return files


pytestmark = pytest.mark.skipif(
    not RUST_CORE_SRC.is_dir(),
    reason="litellm-rust sources not present (non-repo checkout)",
)


def test_every_rust_module_is_registered_or_pending():
    """A new Rust module with top-level pub fns must pick a list."""
    observed = _rust_files_with_top_level_pub_fns()
    covered = (
        set(_registered_provider_files())
        | _route_mod_files()
        | set(SHARED_HELPERS)
        | PENDING_REGISTRATION
    )

    unlisted = observed - covered
    assert not unlisted, (
        f"Rust modules with top-level pub fns that are neither registered for "
        f"parity testing nor listed as pending: {sorted(unlisted)}. Register "
        f"the route/provider in PARITY_ROUTES or SHARED_HELPERS (preferred), "
        f"or defer it in PENDING_REGISTRATION in {__file__}."
    )

    stale = covered - observed
    assert not stale, (
        f"entries for Rust files that do not exist or have no top-level pub "
        f"fns: {sorted(stale)}"
    )

    double_listed = (set(_registered_provider_files()) | set(SHARED_HELPERS)) & (
        PENDING_REGISTRATION
    )
    assert (
        not double_listed
    ), f"files both registered and pending: {sorted(double_listed)}"


@pytest.mark.parametrize("route", sorted(PARITY_ROUTES))
def test_route_entrypoints_exist(route: str):
    """The registered SDK entrypoint exists on both sides of the port."""
    parity = PARITY_ROUTES[route]

    python_module, attr = parity.python_entrypoint
    assert hasattr(
        importlib.import_module(python_module), attr
    ), f"route '{route}': Python entrypoint {python_module}.{attr} not found"

    if parity.rust_core_entrypoint:
        mod_rs = RUST_ROOT / "crates" / "core" / "src" / route / "mod.rs"
        fns = {name for name, _ in _top_level_pub_fns(mod_rs)}
        assert route in fns, (
            f"route '{route}': expected `pub async fn {route}` in "
            f"{_rust_relpath(mod_rs)} (per litellm-rust/ADDING_A_PROVIDER.md), "
            f"found {sorted(fns)}"
        )


@pytest.mark.parametrize(
    "rust_relpath,python_module",
    sorted(_registered_provider_files().items()) + sorted(SHARED_HELPERS.items()),
)
def test_registered_rust_fns_have_python_counterparts(
    rust_relpath: str, python_module: str
):
    _assert_fns_have_python_counterparts(rust_relpath, python_module)
