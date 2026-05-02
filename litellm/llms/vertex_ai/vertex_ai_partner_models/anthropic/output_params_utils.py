"""
Shared sanitization for ``output_config`` / ``output_format`` on Vertex AI
Claude. Lives in its own module so both the chat-completion transformation
(``transformation.py``) and the Messages pass-through transformation
(``experimental_pass_through/transformation.py``) can import it without
forming a cycle through the parent module's heavier imports.

CodeQL flagged the ``..transformation`` import path as a potential cyclic
import; extracting the helper into a leaf module resolves the warning and
keeps the parent module's import surface narrow.
"""

# Keys inside ``output_config`` that Vertex AI Claude does not accept.
#
# Historical note: ``effort`` was previously listed here based on the
# assumption that Vertex returned 400 "Extra inputs are not permitted" on
# adaptive-thinking effort knobs. Direct ``:rawPredict`` curls against
# ``us-east5`` on opus-4-7 (5-value enum: ``low|medium|high|xhigh|max``) and
# opus-4-6/sonnet-4-6 (4-value enum: ``low|medium|high|max``) confirmed
# Vertex AI Claude 4.6+ accepts ``output_config.effort`` with the same
# per-model gating as Anthropic direct. Stripping it here was hiding a knob
# Vertex actually supports — keeping ``reasoning_effort`` from landing on
# Vertex 4.6+ while it works on Anthropic direct. Removing ``effort`` from
# this allowlist relies on ``AnthropicConfig._apply_output_config`` to gate
# ``xhigh``/``max`` per model from the model map, which is the same
# validation Vertex applies server-side.
#
# Keep this list narrow — anything Vertex DOES accept (e.g. ``format`` for
# structured outputs, ``effort`` on 4.6+) must be preserved so callers can
# rely on Anthropic-native features.
VERTEX_UNSUPPORTED_OUTPUT_CONFIG_KEYS: frozenset = frozenset()


def sanitize_vertex_anthropic_output_params(data: dict) -> None:
    """
    Strip Vertex-unsupported keys from ``output_config`` /
    ``output_format`` in-place; forward whatever remains.

    Today ``VERTEX_UNSUPPORTED_OUTPUT_CONFIG_KEYS`` is empty (Vertex 4.6+
    accepts both ``format`` and ``effort``), so this helper is effectively
    a passthrough plus a defensive non-dict guard. The filtering scaffold
    is kept so that adding a new Vertex-unsupported key — should one
    surface — is a one-line change to the frozenset.

    Behavior:
      * ``output_config`` is filtered against
        ``VERTEX_UNSUPPORTED_OUTPUT_CONFIG_KEYS``; whatever survives is
        forwarded. If filtering empties the dict, it's removed entirely
        rather than sending an empty object downstream.
      * ``output_format`` is forwarded as-is (Vertex AI Claude accepts it).
      * Non-dict values for ``output_config`` are dropped to avoid sending
        malformed payloads downstream.
    """
    output_config = data.get("output_config")
    if output_config is None:
        return
    if not isinstance(output_config, dict):
        data.pop("output_config", None)
        return
    sanitized = {
        k: v
        for k, v in output_config.items()
        if k not in VERTEX_UNSUPPORTED_OUTPUT_CONFIG_KEYS
    }
    if sanitized:
        data["output_config"] = sanitized
    else:
        data.pop("output_config", None)
