"""
Regression tests for substring-collision bugs in the unified-resource-id extractors.

Background (LIT-3384, Cornell CIT): the managed-file ID format embeds a
`llm_output_file_model_id,<HASH>` field. The previous regex in
`extract_model_id_from_unified_id` (and the related `provider_resource_id`
patterns) was unanchored: `r"model_id,([^;]+)"`. Because `model_id` is a
substring of `llm_output_file_model_id`, the regex would capture the deployment
SHA-256 hash as the model_id and leak it into the team-model-access candidate
list, producing a spurious 403.

The fix anchors every field-name match to `(?:^|;)` so only a bare field
(separated by `;` or start-of-string) matches.
"""

import base64

import pytest

from litellm.llms.base_llm.managed_resources.utils import (
    extract_model_id_from_unified_id,
    extract_provider_resource_id_from_unified_id,
    extract_target_model_names_from_unified_id,
    extract_unified_uuid_from_unified_id,
    parse_unified_id,
)


# Deployment hash like the one that triggered LIT-3384.
DEPLOYMENT_HASH = (
    "8d0eaa7e6c6f54a425dfd0062cb6b0dc38093f36f0d15994e6ae491ccfc53ac4"
)


def _file_unified_id(
    target_model_names: str = "anthropic.batch.claude-4.5-haiku",
    output_file_id: str = "arn:aws:bedrock:us-east-1:1234:async-invoke/abc",
    model_id: str = DEPLOYMENT_HASH,
) -> str:
    """Build the unified-file-ID format produced by acreate_file:
    `litellm_proxy:<type>;unified_id,<uuid>;target_model_names,<names>;`
    `llm_output_file_id,<id>;llm_output_file_model_id,<model_id>`
    """
    return (
        "litellm_proxy:application/octet-stream"
        ";unified_id,c4843482-b176-4901-8292-7523fd0f2c6e"
        f";target_model_names,{target_model_names}"
        f";llm_output_file_id,{output_file_id}"
        f";llm_output_file_model_id,{model_id}"
    )


def _vector_store_unified_id() -> str:
    """Bare-key unified ID from generate_unified_id_string (vector_store path)."""
    return (
        "litellm_proxy:vector_store"
        ";unified_id,abc-123"
        ";target_model_names,gpt-4,gemini"
        ";resource_id,vs_xyz"
        ";model_id,real-model-id-123"
    )


class TestManagedFileUnifiedIdRegression:
    """LIT-3384 regression — `llm_output_file_model_id` must NOT be matched
    by the `model_id` regex.
    """

    def test_extract_model_id_does_not_leak_output_file_model_id_hash(self):
        unified = _file_unified_id()
        # Before the fix this returned DEPLOYMENT_HASH.
        assert extract_model_id_from_unified_id(unified) is None

    def test_parse_unified_id_does_not_set_model_id_on_managed_file(self):
        parsed = parse_unified_id(_file_unified_id())
        assert parsed is not None
        # Before the fix: parsed["model_id"] == DEPLOYMENT_HASH
        assert parsed["model_id"] is None
        # target_model_names still correctly extracted
        assert parsed["target_model_names"] == [
            "anthropic.batch.claude-4.5-haiku"
        ]

    def test_extract_provider_resource_id_does_not_leak_output_file_id(self):
        unified = _file_unified_id()
        # `llm_output_file_id,X` must not be matched by the bare `file_id,` regex.
        # Files use a dedicated extractor (`get_output_file_id_from_unified_file_id`).
        assert extract_provider_resource_id_from_unified_id(unified) is None

    def test_target_model_names_unaffected(self):
        unified = _file_unified_id()
        assert extract_target_model_names_from_unified_id(unified) == [
            "anthropic.batch.claude-4.5-haiku"
        ]

    def test_base64_encoded_managed_file_id_does_not_leak_hash(self):
        """Same check using the base64-encoded form (what auth_checks sees)."""
        decoded = _file_unified_id()
        b64 = base64.urlsafe_b64encode(decoded.encode()).decode().rstrip("=")
        assert extract_model_id_from_unified_id(b64) is None
        parsed = parse_unified_id(b64)
        assert parsed is not None
        assert parsed["model_id"] is None
        assert parsed["target_model_names"] == [
            "anthropic.batch.claude-4.5-haiku"
        ]


class TestBareKeyUnifiedIdNoRegression:
    """Bare-key unified IDs (vector_store path) must still extract correctly."""

    def test_extract_model_id_bare_key(self):
        assert (
            extract_model_id_from_unified_id(_vector_store_unified_id())
            == "real-model-id-123"
        )

    def test_extract_provider_resource_id_bare_key(self):
        assert (
            extract_provider_resource_id_from_unified_id(
                _vector_store_unified_id()
            )
            == "vs_xyz"
        )

    def test_parse_unified_id_bare_key(self):
        parsed = parse_unified_id(_vector_store_unified_id())
        assert parsed is not None
        assert parsed["model_id"] == "real-model-id-123"
        assert parsed["provider_resource_id"] == "vs_xyz"
        assert parsed["target_model_names"] == ["gpt-4", "gemini"]
        assert parsed["unified_uuid"] == "abc-123"


class TestAuthCandidateListIntegration:
    """End-to-end check: managed file IDs must not leak the deployment hash
    into the team-model-access candidate list.
    """

    def test_auth_candidates_do_not_include_deployment_hash(self):
        from litellm.proxy.auth.auth_utils import (
            _extract_models_from_managed_resource_id,
        )

        decoded = _file_unified_id()
        b64 = base64.urlsafe_b64encode(decoded.encode()).decode().rstrip("=")
        candidates = _extract_models_from_managed_resource_id(
            b64, resource_id_field="input_file_id"
        )
        assert DEPLOYMENT_HASH not in candidates, (
            f"Deployment hash leaked into team-access candidate list: {candidates}"
        )
        assert "anthropic.batch.claude-4.5-haiku" in candidates


class TestUnifiedUuidAnchored:
    """Anchor consistency check for the `unified_id,` field — no current
    substring collision risk, but the anchor must not introduce false negatives.
    """

    def test_unified_uuid_extracted(self):
        unified = _file_unified_id()
        assert (
            extract_unified_uuid_from_unified_id(unified)
            == "c4843482-b176-4901-8292-7523fd0f2c6e"
        )
