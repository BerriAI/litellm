"""
Write-only secret invariants (LIT-2891 validation #2).

The single most important property of `/v2/agent-secrets`: a stored secret
value MUST never reappear on any GET response, ever. The defense is layered:

* **Type-level**: `AgentSecretResponse` has no `value` field. Even an
  accidental `**row` splat can't surface the ciphertext through the model.
* **Endpoint-level**: every `response_model` annotation references one of the
  value-free models, and `_row_to_response` reads named columns explicitly
  rather than splatting the whole Prisma row.
* **Schema-level**: `AgentSecretListResponse` only nests
  `AgentSecretResponse`, so a list response can't smuggle values either.

These tests pin all three so a future refactor can't quietly weaken the
contract. They deliberately avoid importing the endpoints module (which
would pull in the full proxy auth stack and its `orjson` dependency) — we
stress the type contract directly and grep the endpoint source as text.
"""

import pathlib

import pytest
from pydantic import ValidationError

from litellm.proxy.agent_settings_endpoints.types import (
    AgentSecretCreateRequest,
    AgentSecretListResponse,
    AgentSecretResponse,
    AgentSecretUpdateRequest,
)

_SECRETS_ENDPOINTS_PATH = (
    pathlib.Path(__file__).resolve().parents[4]
    / "litellm"
    / "proxy"
    / "agent_settings_endpoints"
    / "secrets_endpoints.py"
)


def _read_endpoints_source() -> str:
    # Read as text instead of importing the module — keeps these tests
    # collectable without the full proxy dependency tree (orjson, prisma,
    # etc.) installed.
    assert (
        _SECRETS_ENDPOINTS_PATH.exists()
    ), f"secrets_endpoints.py missing at {_SECRETS_ENDPOINTS_PATH}"
    return _SECRETS_ENDPOINTS_PATH.read_text()


def _strip_docstrings_and_comments(source: str) -> str:
    """Remove triple-quoted strings and `# ...` line comments.

    Used by the source-grep tests to avoid tripping on legitimate mentions
    of `value_enc` inside docstrings (e.g. "intentionally NOT read here")
    and inline explanatory comments. Crude regex is fine for our use —
    the source file is small and we're checking a denylist.
    """
    import re

    # Triple-quoted strings (greedy across lines).
    no_docstrings = re.sub(r'"""[\s\S]*?"""', "", source)
    no_docstrings = re.sub(r"'''[\s\S]*?'''", "", no_docstrings)
    # `# ...` line comments.
    no_comments = re.sub(r"#[^\n]*", "", no_docstrings)
    return no_comments


class TestResponseModelHasNoValueField:
    """The type system itself enforces validation #2."""

    def test_agent_secret_response_has_no_value_field(self):
        assert "value" not in AgentSecretResponse.model_fields
        assert "value_enc" not in AgentSecretResponse.model_fields

    def test_response_model_dump_never_contains_value(self):
        # Even if a future refactor added `value` to the row dict, the
        # response schema would silently drop it (Pydantic ignores extras
        # by default for BaseModel).
        resp = AgentSecretResponse(
            name="X",
            scope="all",
            type="env",
            file_path=None,
            created_at="now",
            updated_at="now",
        )
        dumped = resp.model_dump()
        assert "value" not in dumped
        assert "value_enc" not in dumped

    def test_list_response_only_nests_value_free_models(self):
        # A list response is a wrapper around AgentSecretResponse — assert
        # the nested type has the right shape.
        resp = AgentSecretListResponse(secrets=[])
        dumped = resp.model_dump()
        assert dumped == {"secrets": []}
        # The model_fields annotation must be List[AgentSecretResponse].
        secrets_field = AgentSecretListResponse.model_fields["secrets"]
        annotation = str(secrets_field.annotation)
        assert "AgentSecretResponse" in annotation

    def test_response_model_drops_value_extra_via_validation(self):
        # Round-trip through validate to confirm Pydantic strips an extra
        # `value` even if a caller tries to slip it past the type.
        raw = {
            "name": "X",
            "scope": "all",
            "type": "env",
            "file_path": None,
            "created_at": "t",
            "updated_at": "t",
            "value": "DO-NOT-LEAK",
        }
        resp = AgentSecretResponse.model_validate(raw)
        assert "value" not in resp.model_dump()
        assert "DO-NOT-LEAK" not in str(resp.model_dump())


class TestRequestModelsAcceptValue:
    """The flip side: write-only means values DO go in on POST/PUT."""

    def test_create_request_requires_value(self):
        with pytest.raises(ValidationError):
            AgentSecretCreateRequest(name="X", value="")  # too short
        with pytest.raises(ValidationError):
            AgentSecretCreateRequest(name="X")  # missing entirely

    def test_create_request_validates_name_charset(self):
        # We use names as env-var keys, so they must follow shell-safe
        # identifier rules. Reject names with dashes or starting with a digit.
        with pytest.raises(ValidationError):
            AgentSecretCreateRequest(name="bad-name", value="x")
        with pytest.raises(ValidationError):
            AgentSecretCreateRequest(name="9starting_digit", value="x")
        # Valid identifier passes.
        AgentSecretCreateRequest(name="OPENAI_API_KEY", value="x")

    def test_update_request_value_is_optional(self):
        # PUT is partial — UI may send only a scope change.
        body = AgentSecretUpdateRequest(scope="all")
        assert body.value is None

    def test_update_request_value_min_length_1(self):
        with pytest.raises(ValidationError):
            AgentSecretUpdateRequest(value="")


class TestEndpointSourceContainsNoValueLeak:
    """Belt-and-suspenders: scan the secrets endpoint source for accidental
    references that could leak `value_enc` onto a response. If a future PR
    introduces something like `response['value'] = decrypt(...)`, these
    tests catch it before it ships.
    """

    def test_no_decrypt_call_in_endpoints_module(self):
        source = _read_endpoints_source()
        # The secrets module must not import the decrypt helper — only the
        # VM config endpoints module needs it (for Test Connection).
        assert "decrypt_optional" not in source
        assert "decrypt_value_helper" not in source

    def test_endpoints_use_value_free_response_models(self):
        source = _read_endpoints_source()
        # Both response_model occurrences should reference the value-free
        # types — never some bare dict that could drift.
        assert "response_model=AgentSecretResponse" in source
        assert "response_model=AgentSecretListResponse" in source

    def test_row_to_response_does_not_read_value_enc(self):
        source = _read_endpoints_source()
        # Find the helper that maps DB rows to API responses. The chokepoint
        # is the GET path: it MUST NOT read value_enc. Write-side payload
        # builders (create/update) are allowed to mention it because they
        # write the ciphertext into the DB.
        helper_marker = "def _row_to_response(row:"
        helper_start = source.find(helper_marker)
        assert helper_start != -1, (
            "_row_to_response helper missing — refactor must not lose this "
            "chokepoint."
        )
        next_def = source.find("\ndef ", helper_start + len(helper_marker))
        helper_body = (
            source[helper_start:] if next_def == -1 else source[helper_start:next_def]
        )
        # Strip docstrings and `# ...` comments before checking. The helper's
        # docstring legitimately calls out "value_enc is intentionally NOT
        # read here", and we don't want that mention to trip the test.
        stripped = _strip_docstrings_and_comments(helper_body)
        assert "value_enc" not in stripped, (
            "_row_to_response references value_enc — the GET path must "
            "never touch the ciphertext column."
        )

    def test_no_response_dict_assigns_value(self):
        # Make sure no code path builds a response dict and adds `value` to
        # it. (`response_model` enforces this on FastAPI's side, but we
        # double-check the module doesn't construct such dicts directly.)
        source = _read_endpoints_source()
        # Disallow patterns like `"value": ...,` in response-shaped dicts.
        # Allow `body.value` (a request field). The simplest portable rule
        # is: the literal `"value":` must never appear in the source.
        assert (
            '"value":' not in source
        ), "secrets endpoint constructs a dict with a `value` key — possible leak."


@pytest.fixture(autouse=True)
def _no_op_fixture():
    yield
