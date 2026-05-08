"""
Regression tests for the parsed-URL hostname match used to identify a
caller-supplied ``api_base`` as a known openai-compatible provider.

The previous shape (``if endpoint in api_base:``) used unanchored
substring search, which let a caller pass
``https://attacker.com/api.groq.com/openai/v1`` and have the proxy
return ``GROQ_API_KEY`` as the dynamic credential — exfiltrating the
server's real provider key to an attacker-controlled host on the
outbound request.
"""

import os
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.abspath("../../.."))

from litellm.litellm_core_utils.get_llm_provider_logic import (
    _endpoint_matches_api_base,
    get_llm_provider,
)


class TestEndpointMatchesApiBase:
    """Direct unit tests on the parsed-URL matcher."""

    @pytest.mark.parametrize(
        "endpoint, api_base",
        [
            # Bare hostname endpoint, exact host match.
            ("api.perplexity.ai", "https://api.perplexity.ai/v1"),
            # Endpoint includes a path; api_base path starts with it.
            ("api.groq.com/openai/v1", "https://api.groq.com/openai/v1"),
            # Endpoint with full URL scheme.
            ("https://api.cerebras.ai/v1", "https://api.cerebras.ai/v1/chat"),
            # Trailing-slash on registered endpoint must not break match.
            ("https://llm.chutes.ai/v1/", "https://llm.chutes.ai/v1/chat"),
            # Case-insensitive on hostname.
            ("api.groq.com/openai/v1", "https://API.GROQ.COM/openai/v1"),
        ],
    )
    def test_legitimate_provider_urls_match(self, endpoint, api_base):
        assert _endpoint_matches_api_base(endpoint, api_base) is True

    @pytest.mark.parametrize(
        "endpoint, api_base",
        [
            # Attacker host, registered endpoint smuggled into path.
            (
                "api.groq.com/openai/v1",
                "https://attacker.com/api.groq.com/openai/v1",
            ),
            # Attacker host, registered endpoint smuggled into a path segment.
            (
                "api.groq.com/openai/v1",
                "https://attacker.com/foo/api.groq.com/openai/v1",
            ),
            # Lookalike host that contains the registered host as a suffix label.
            (
                "api.groq.com/openai/v1",
                "https://api.groq.com.attacker.com/openai/v1",
            ),
            # Lookalike host with the registered host as a prefix.
            (
                "api.groq.com/openai/v1",
                "https://api.groq.com.evil.example/openai/v1",
            ),
            # Right host, wrong path — endpoint requires ``/openai/v1`` prefix.
            ("api.groq.com/openai/v1", "https://api.groq.com/v1"),
            # Path-segment lookalike: ``/openai/v10`` must not match ``/openai/v1``.
            ("api.groq.com/openai/v1", "https://api.groq.com/openai/v10"),
            # Userinfo / @-injection trick — the ``hostname`` after ``@`` is
            # what httpx connects to.
            (
                "api.groq.com/openai/v1",
                "https://api.groq.com@attacker.com/openai/v1",
            ),
        ],
    )
    def test_attacker_smuggling_does_not_match(self, endpoint, api_base):
        assert _endpoint_matches_api_base(endpoint, api_base) is False


class TestGetLlmProviderRejectsAttackerSmuggledApiBase:
    """
    End-to-end: ``get_llm_provider`` must NOT return the server's stored
    secret (e.g. ``GROQ_API_KEY``) for an api_base whose hostname is
    attacker-controlled, even when the registered endpoint string appears
    elsewhere in the URL.
    """

    def test_attacker_host_does_not_yield_groq_secret(self):
        # The function may either fall through (different provider) or
        # raise BadRequestError because the model can't be identified.
        # The invariant under test is that ``GROQ_API_KEY`` is never
        # looked up against an attacker-controlled hostname.
        import litellm

        with patch(
            "litellm.litellm_core_utils.get_llm_provider_logic.get_secret_str",
            return_value="server-real-groq-key",
        ) as mocked_secret:
            try:
                _, _, dynamic_api_key, _ = get_llm_provider(
                    model="some-model",
                    api_base="https://attacker.com/api.groq.com/openai/v1",
                )
                # If it returned, the dynamic key must not be the secret.
                assert dynamic_api_key != "server-real-groq-key"
            except litellm.exceptions.BadRequestError:
                # Acceptable outcome: provider unidentifiable, no secret
                # was returned.
                pass

        # Regardless of return / raise, the secret must never have been
        # read against this attacker-controlled api_base.
        groq_lookups = [
            call
            for call in mocked_secret.call_args_list
            if call.args and call.args[0] == "GROQ_API_KEY"
        ]
        assert groq_lookups == []

    def test_legitimate_groq_api_base_still_resolves(self):
        with patch(
            "litellm.litellm_core_utils.get_llm_provider_logic.get_secret_str",
            return_value="server-real-groq-key",
        ):
            _, provider, dynamic_api_key, _ = get_llm_provider(
                model="some-model",
                api_base="https://api.groq.com/openai/v1",
            )

        assert provider == "groq"
        assert dynamic_api_key == "server-real-groq-key"
