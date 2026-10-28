"""Virtual-key model allowlist enforcement on routes that bind the model from the URL.

Tracking issue: https://github.com/BerriAI/litellm/issues/48760

A virtual key scoped with ``models: ["allowed-model"]`` is correctly denied a model
named in the request body, but the Google ``generateContent`` and
``streamGenerateContent`` routes and the ``/openai/deployments/{model}/images/...``
routes bind the model from the URL and route on it, while the access check reads the
body model. A body model the key may use plus a denied model in the URL is served the
denied model. The open PR #41852 closes the chat, completions and embeddings routes,
not these.

These skeleton cases name the routes the fix must cover. They are skipped until the
fix lands; the implementation re-checks the resolved ``self.data["model"]`` at the
single request-processing choke point so every route is covered at once. See the
tracking issue for the full reproduction, the N-of-N matrix and the fix-control run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import pytest

pytestmark = pytest.mark.skip(
    reason="WIP, enforcing the model allowlist on URL-bound model routes, see issue #48760"
)


@dataclass(frozen=True, slots=True)
class _UrlRouteCase:
    name: str
    method: str
    path: str
    body_model: str
    served_denied: bool


# A key scoped to "allowed-model". Each case sends "denied-model" in the URL and
# "allowed-model" in the body. served_denied is the behaviour on current main that
# the fix must turn into a 403.
_CASES: Final = (
    _UrlRouteCase(
        "generate_content",
        "POST",
        "/v1beta/models/denied-model:generateContent",
        "allowed-model",
        True,
    ),
    _UrlRouteCase(
        "stream_generate_content",
        "POST",
        "/v1beta/models/denied-model:streamGenerateContent",
        "allowed-model",
        True,
    ),
    _UrlRouteCase(
        "images_generations_deployment",
        "POST",
        "/openai/deployments/denied-model/images/generations",
        "allowed-model",
        True,
    ),
    _UrlRouteCase(
        "images_edits_deployment",
        "POST",
        "/openai/deployments/denied-model/images/edits",
        "allowed-model",
        True,
    ),
)


@pytest.mark.parametrize("case", _CASES, ids=lambda case: case.name)
def test_url_route_model_outside_allowlist_is_denied(case: _UrlRouteCase) -> None:
    """A URL-bound model outside the key allowlist must be denied before dispatch.

    The served model, not just the body model, must be inside the key's allowed set.
    The request below sends an allowed body model and a denied URL model, and must
    return 403 key_model_access_denied with no upstream call.
    """
    raise NotImplementedError("fix and assertions pending, see issue #48760")
