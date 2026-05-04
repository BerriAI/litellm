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
# Add an entry only when a 400 "Extra inputs are not permitted" is
# reproducible against the live Vertex endpoint.
VERTEX_UNSUPPORTED_OUTPUT_CONFIG_KEYS: frozenset = frozenset()


def sanitize_vertex_anthropic_output_params(data: dict) -> None:
    """
    Strip Vertex-unsupported keys from ``output_config`` /
    ``output_format`` in-place; forward whatever remains.

    Behavior:
      * ``output_config`` containing only unsupported keys (e.g. ``effort``
        alone) is removed entirely so the request body has no empty dict.
      * ``output_config`` containing a mix of supported + unsupported keys
        has the unsupported subset filtered out and the rest forwarded.
      * ``output_config`` that is supported in full passes through unchanged.
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
