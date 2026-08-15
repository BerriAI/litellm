"""Payload shaping for the Generic Guardrail API.

Lets the config declare what the guardrail endpoint actually receives:
``send_images``, ``exclude_payload_fields``, ``max_messages``,
``max_text_chars`` and ``strip_patterns``.

Shaping is lossy by design, so every shaped payload carries a ``PayloadLoss``
describing what the endpoint could not see. The guardrail refuses to write back
mutations for those components: a provider that never received a text block (or
received a stripped / truncated one) must not be able to replace the caller's
original content with it.
"""

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import reduce
from typing import Final

from pydantic import JsonValue

from litellm._logging import verbose_proxy_logger
from litellm.proxy.guardrails._content_utils import (
    iter_messages_text,
    map_messages_image_urls,
    map_messages_text,
)
from litellm.types.proxy.guardrails.guardrail_hooks.generic_guardrail_api import (
    GenericGuardrailAPIRequest,
)

# Routing / correlation keys the endpoint needs to interpret any payload at all.
PROTECTED_PAYLOAD_FIELDS: Final = frozenset({"input_type", "litellm_call_id"})

# Ceiling on substitutions per pattern per text, so a pathological pattern on a
# huge transcript cannot stall the request path.
MAX_STRIP_SUBSTITUTIONS: Final = 64

# Python's re has no match timeout, and the substitution cap above bounds how many
# matches are replaced, not how long the engine spends finding one. A configured
# pattern that backtracks catastrophically would therefore burn the worker for as
# long as the caller's text lets it, so text past this size is not matched at all.
MAX_STRIP_INPUT_CHARS: Final = 100_000

# What a too-large block is replaced with. Dropping it is the fail-closed choice:
# sending it unstripped would leak exactly the content strip_patterns exists to
# remove, and it cannot be written back either, since it is marked lossy.
OVERSIZED_TEXT_PLACEHOLDER: Final = "[omitted: exceeds strip size limit]"

# Stands in for image data when send_images=False, mirroring the "[present]"
# convention used for headers whose value is not forwarded.
IMAGE_OMITTED_PLACEHOLDER: Final = "[omitted]"

_IMAGES_FIELD: Final = frozenset({"images"})


@dataclass(frozen=True, slots=True)
class PayloadPolicy:
    """Resolved payload-shaping configuration."""

    send_images: bool = True
    exclude_fields: frozenset[str] = frozenset()
    max_messages: int | None = None
    max_text_chars: int | None = None
    strip_patterns: tuple[re.Pattern[str], ...] = ()

    @property
    def shapes_text(self) -> bool:
        return self.max_text_chars is not None or bool(self.strip_patterns)

    @property
    def is_lossy(self) -> bool:
        """Whether any option keeps part of the request from reaching the guardrail."""
        return self.shapes_text or self.max_messages is not None or not self.send_images or bool(self.exclude_fields)


@dataclass(frozen=True, slots=True)
class PayloadLoss:
    """What the guardrail endpoint did not get to see."""

    altered_text_indices: frozenset[int] = frozenset()
    images_omitted: bool = False
    tools_omitted: bool = False


def resolve_exclude_fields(raw: Sequence[str] | None, *, guardrail_name: str | None) -> frozenset[str]:
    """Validate ``exclude_payload_fields`` against the request model.

    Unknown keys and routing-critical keys are dropped with a warning rather
    than raising, so a stale config never takes the proxy down.
    """
    if not raw:
        return frozenset()

    known: Final = frozenset(GenericGuardrailAPIRequest.model_fields)
    unknown: Final = tuple(field for field in raw if field not in known)
    if unknown:
        verbose_proxy_logger.warning(
            "Generic Guardrail API (%s): exclude_payload_fields contains unknown field(s) %s; ignoring them. "
            "Known fields: %s",
            guardrail_name,
            unknown,
            sorted(known),
        )

    protected: Final = tuple(field for field in raw if field in PROTECTED_PAYLOAD_FIELDS)
    if protected:
        verbose_proxy_logger.warning(
            "Generic Guardrail API (%s): exclude_payload_fields cannot drop routing-critical field(s) %s; "
            "they will still be sent.",
            guardrail_name,
            protected,
        )

    return frozenset(field for field in raw if field in known and field not in PROTECTED_PAYLOAD_FIELDS)


def compile_patterns(raw: Sequence[str] | None, *, option_name: str) -> tuple[re.Pattern[str], ...]:
    """Compile config-supplied regexes, raising on an invalid pattern.

    A bad regex is a config bug: failing at init surfaces it at proxy boot
    instead of silently disabling the option on the request path.
    """
    if not raw:
        return ()
    try:
        return tuple(re.compile(pattern) for pattern in raw)
    except re.error as e:
        raise ValueError(f"{option_name} contains an invalid regex: {e}") from e


def _strip_text(text: str, patterns: tuple[re.Pattern[str], ...]) -> str:
    return reduce(lambda acc, pattern: pattern.sub("", acc, count=MAX_STRIP_SUBSTITUTIONS), patterns, text)


def _shape_text(text: str, policy: PayloadPolicy) -> str:
    if policy.strip_patterns and len(text) > MAX_STRIP_INPUT_CHARS:
        verbose_proxy_logger.warning(
            "Generic Guardrail API: a %d character text block exceeds the %d character strip_patterns "
            "limit and was replaced with a placeholder instead of being matched.",
            len(text),
            MAX_STRIP_INPUT_CHARS,
        )
        return OVERSIZED_TEXT_PLACEHOLDER
    stripped: Final = _strip_text(text, policy.strip_patterns)
    return stripped[: policy.max_text_chars] if policy.max_text_chars is not None else stripped


def _retained_messages(messages: JsonValue, policy: PayloadPolicy) -> JsonValue:
    """The message window ``max_messages`` keeps, or None when there is nothing to window."""
    if not isinstance(messages, list) or not messages:
        return None
    if policy.max_messages is None:
        return messages
    return messages[-policy.max_messages :]


def _text_window_size(retained: JsonValue, policy: PayloadPolicy, total_texts: int) -> int:
    """How many trailing ``texts`` entries belong to the retained messages.

    ``texts`` is fragment-based (a multimodal turn contributes several entries, a
    tool-call-only turn contributes none) while ``max_messages`` counts messages,
    so applying the same number to both would leave the two views of the payload
    covering different parts of the conversation. Counting the fragments inside
    the retained messages keeps them aligned. Without structured messages to
    count (a response payload, an embedding call) the message count is the only
    bound available, so it is applied to the fragments directly.
    """
    if policy.max_messages is None:
        return total_texts
    if not isinstance(retained, list):
        return min(policy.max_messages, total_texts)
    fragments: Final = sum(1 for _ in iter_messages_text(retained))
    return min(fragments, total_texts)


def _shape_texts(
    texts: Sequence[str],
    policy: PayloadPolicy,
    window_size: int,
) -> tuple[JsonValue, frozenset[int]]:
    """Window and rewrite the flat text list, and report which indices changed.

    ``max_messages`` has to bound ``texts`` as well as ``structured_messages``:
    the translation handlers populate the flat list from the whole conversation,
    so windowing only the structured form would leave payload size proportional
    to session length, which is the thing the option exists to stop. Windowing
    shifts every position, so the caller treats the whole list as unusable for
    write-back (see ``merge_guardrailed_texts``).
    """
    windowed: Final = texts[len(texts) - window_size :] if window_size < len(texts) else texts
    shaped: Final = [_shape_text(text, policy) for text in windowed]  # mutable-ok: JSON texts is an array
    if len(windowed) != len(texts):
        return shaped, frozenset(range(len(texts)))
    return shaped, frozenset(index for index, text in enumerate(texts) if text != shaped[index])


def _shape_messages(retained: JsonValue, policy: PayloadPolicy) -> JsonValue:
    if not isinstance(retained, list):
        return None
    texted: Final = (
        map_messages_text(retained, lambda text: _shape_text(text, policy)) if policy.shapes_text else retained
    )
    # send_images=False has to cover the inline image parts too, otherwise the
    # base64 payload still leaves LiteLLM inside structured_messages. The part is
    # kept so the guardrail still knows an image was there.
    if policy.send_images:
        return texted
    return map_messages_image_urls(texted, lambda _: IMAGE_OMITTED_PLACEHOLDER)


def shape_payload(
    request: GenericGuardrailAPIRequest,
    policy: PayloadPolicy,
) -> tuple[Mapping[str, JsonValue], PayloadLoss]:
    """Return the JSON payload to POST plus the loss it represents.

    Shaping runs on the dumped payload rather than on the model: pydantic
    validates ``structured_messages`` list content lazily into a
    ``ValidatorIterator``, which is neither a list to walk nor safe to iterate
    twice, so the plain JSON form is the only reliable input here.
    """
    exclude: Final = policy.exclude_fields if policy.send_images else policy.exclude_fields | _IMAGES_FIELD
    excluded: Final = set(exclude) or None  # mutable-ok: model_dump(exclude=...) takes a set
    dumped: Final = request.model_dump(mode="json", exclude=excluded)

    texts: Final = request.texts
    texts_sent: Final = bool(texts) and "texts" not in exclude
    retained: Final = _retained_messages(dumped.get("structured_messages"), policy)
    window_size: Final = _text_window_size(retained, policy, len(texts or ()))
    shaped_texts, altered_indices = (
        _shape_texts(texts or (), policy, window_size) if texts_sent else (None, frozenset[int]())
    )
    shaped_messages: Final = _shape_messages(retained, policy)

    payload: Final = {  # mutable-ok: the POST body is a JSON object
        **dumped,
        **({"texts": shaped_texts} if texts_sent else {}),  # mutable-ok: JSON object
        **({"structured_messages": shaped_messages} if shaped_messages is not None else {}),  # mutable-ok: JSON
    }
    loss: Final = PayloadLoss(
        # An excluded text block was never seen at all, so no index may be rewritten.
        altered_text_indices=(altered_indices if texts_sent else frozenset(range(len(texts or ())))),
        images_omitted="images" in exclude,
        tools_omitted="tools" in exclude,
    )
    return payload, loss


def merge_guardrailed_texts(
    *,
    original: list[str],  # mutable-ok: the framework's write-back contract
    returned: list[str],  # mutable-ok: the framework's write-back contract
    loss: PayloadLoss,
    guardrail_name: str | None,
) -> list[str]:  # mutable-ok: GenericGuardrailAPIInputs["texts"] is list[str]
    """Accept guardrail-returned text only where the endpoint saw the original.

    Indices the payload altered (stripped / truncated) or never carried keep the
    caller's text, so lossy shaping can never splice a mangled prompt back into
    the request. Length mismatch breaks index alignment, so with any loss the
    whole rewrite is refused.
    """
    if not loss.altered_text_indices:
        return returned
    if len(returned) != len(original):
        verbose_proxy_logger.warning(
            "Generic Guardrail API (%s): ignoring returned texts. The payload was shaped "
            "(strip_patterns / max_text_chars / excluded texts) and the guardrail returned "
            "%d text(s) for %d sent, so they cannot be aligned.",
            guardrail_name,
            len(returned),
            len(original),
        )
        return original
    verbose_proxy_logger.warning(
        "Generic Guardrail API (%s): keeping original text for shaped index(es) %s; the guardrail "
        "never saw the full content at those positions.",
        guardrail_name,
        sorted(loss.altered_text_indices),
    )
    return [  # mutable-ok: write-back contract
        original[index] if index in loss.altered_text_indices else text for index, text in enumerate(returned)
    ]
