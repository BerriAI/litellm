"""Runs a test module's `Router(...)` calls once per backend.

Opt a module in with `pytestmark = pytest.mark.usefixtures("router_backend")` and import
`router_backend` from here. The Rust run pins `RUST_REQUIRED`, so a config the Rust backend
cannot serve fails loudly instead of quietly running on Python. `RUST_GAPS` lists the known
Rust-side failures as strict xfails: the backend's to-do list, which fails once a gap closes.
"""

from __future__ import annotations

from collections.abc import Generator, Mapping
from types import MappingProxyType
from typing import Final

import pytest

from litellm.router_backends.rust_router import NATIVE_ROUTER
from litellm.router_backends.selection import pinned_backend
from litellm.rust_bridge.configuration import Decision

RUST_GAPS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "test_set_deployment_num_retries_on_exception": "calls a PythonRouter internal the Rust backend replaces",
        "test_set_deployment_num_retries_does_not_override_existing": "calls a PythonRouter internal the Rust backend replaces",
        "test_deployment_without_num_retries": "calls a PythonRouter internal the Rust backend replaces",
        "test_request_level_num_retries_takes_precedence": "calls a PythonRouter internal the Rust backend replaces",
        "test_update_kwargs_does_not_fill_in_a_num_retries_default": "calls a PythonRouter internal the Rust backend replaces",
        "test_set_deployment_num_retries_with_string_value": "calls a PythonRouter internal the Rust backend replaces",
        "test_update_kwargs_preserves_an_explicit_zero": "calls a PythonRouter internal the Rust backend replaces",
        "test_async_function_with_retries_none_falls_back_to_zero": "calls a PythonRouter internal the Rust backend replaces",
        "test_async_function_with_retries_none_falls_back_to_router_default": "calls a PythonRouter internal the Rust backend replaces",
        "test_deployment_num_retries_applies_to_image_generation": "aimage_generation is not served yet",
        "test_request_num_retries_still_wins_on_image_generation": "aimage_generation is not served yet",
        "test_flag_off_hides_fallback_credentials": "dict fallback targets decline at construction",
        "test_flag_on_masks_fallback_credentials": "dict fallback targets decline at construction",
        "test_include_fallback_errors_propagates_through_router": "include_fallback_errors is not served yet",
        "test_set_response_headers_adds_model_group_to_streaming_wrapper": "calls a PythonRouter internal the Rust backend replaces",
    }
)


@pytest.fixture(params=("python", "rust"))
def router_backend(request: pytest.FixtureRequest) -> Generator[str]:
    backend: Final = str(request.param)
    if backend == "rust" and NATIVE_ROUTER.load() is None:
        pytest.skip("the native router is not built")
    if backend == "rust":
        gap: Final = next((reason for test, reason in RUST_GAPS.items() if f"::{test}[" in request.node.nodeid), None)
        if gap is not None:
            request.applymarker(pytest.mark.xfail(reason=gap, strict=True))
    with pinned_backend(Decision.PYTHON if backend == "python" else Decision.RUST_REQUIRED):
        yield backend
