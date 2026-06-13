"""Serialize the IR into a github_copilot ``/chat/completions`` request body.

v1's chain on the SDK path (main.py's big openai elif, "github_copilot" in
``openai_compatible_providers``) is ``GithubCopilotConfig.map_openai_params``
(OpenAIConfig's model-shape fork) + ``transform_request``
(OpenAIConfig.transform_request, which calls ``_transform_messages`` BEFORE
the base five-touch assembly). The body is therefore the openai_compat
assembly with ONE delta probed in-process at HEAD (seeded fake token dir, no
live OAuth):

- ``_transform_messages`` rewrites EVERY system message's ``role`` to
  ``assistant`` (content untouched) UNLESS the ambient
  ``litellm.disable_copilot_system_to_assistant`` global is True, in which
  case the messages ride unchanged. The flag threads through
  ``TranslationDeps`` (the DI rule — it is a litellm module global, not a
  ``litellm.constants`` leaf).

The IR hoists leading system text into ``request.system`` and the shared
``serialize_messages`` inverse re-emits them as leading
``{"role": "system", "content": text}`` entries; the shared openai guard
already falls back on every OTHER system shape (list-form content, a system
turn after the first, message ``name``), so the served set is exactly
"leading string-content system messages" — the rewrite flips their role.

The auth device-flow happens in ``validate_environment`` ABOVE the seam
(envelope scope, the xai-OAuth rule); this serializer reads no token files
and triggers no network (semgrep-enforced purity).
"""

from __future__ import annotations

from expression import Error, Result

from ...deps import TranslationDeps
from ...errors import TranslationError
from ...ir import Body, ChatRequest, PlainJson
from ..openai_compat.serialize import assemble_body
from . import params as p

_SerializeResult = Result[Body, TranslationError]


def serialize_request(request: ChatRequest, deps: TranslationDeps) -> _SerializeResult:
    reason = (
        p.unsupported_model_family(request.model)
        or p.unsupported_params(request)
        or p.unsupported_response_format(request)
    )
    if reason is not None:
        return Error(TranslationError.of_unsupported(reason))
    return assemble_body(request).map(
        lambda body: _with_copilot_system_rewrite(body, deps)
    )


def _with_copilot_system_rewrite(body: Body, deps: TranslationDeps) -> Body:
    if deps.disable_copilot_system_to_assistant:
        return body
    messages = body.get("messages")
    if not isinstance(messages, list):
        return body
    return {**body, "messages": [_system_to_assistant(message) for message in messages]}


def _system_to_assistant(message: PlainJson) -> PlainJson:
    if not isinstance(message, dict) or message.get("role") != "system":
        return message
    return {**message, "role": "assistant"}
