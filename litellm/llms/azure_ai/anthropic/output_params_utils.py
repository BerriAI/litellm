"""
Shared sanitization for ``output_config`` on Azure AI Foundry Claude requests.
Lives in its own module so both the chat-completion transformation
(``transformation.py``) and the Messages pass-through transformation
(``messages_transformation.py``) can import it without forming a cycle.
"""

from typing import Final


def _model_accepts_output_config_effort(model: str) -> bool:
    """Whether ``model`` accepts ``output_config.effort`` on Azure AI Foundry.

    Models that advertise ``supports_output_config`` (or any reasoning effort
    level) in the model map accept it; others (e.g. Haiku 4.5) return a 400
    "This model does not support the effort parameter." Imported lazily so
    this stays a leaf module.
    """
    from litellm.llms.anthropic.chat.transformation import AnthropicConfig

    return AnthropicConfig._model_supports_effort_param(model, "azure_ai")


def sanitize_azure_anthropic_output_params(
    data: dict,  # mutable-ok: request dict mutated in place, matches transform_request's contract
    model: str,
) -> None:
    """Strip Azure-unsupported keys from ``output_config`` in-place.

    Behavior:
      * ``output_config.effort`` is dropped for models that do not accept it
        (e.g. Haiku 4.5) and forwarded for those that do (Opus/Sonnet 4.6+).
      * ``output_config`` left empty after filtering is removed so the request
        body carries no empty dict.
      * Non-dict values for ``output_config`` are dropped to avoid sending
        malformed payloads downstream.
    """
    output_config: Final = data.get("output_config")
    if output_config is None:
        return
    if not isinstance(output_config, dict):
        data.pop("output_config", None)
        return

    drop_effort: Final = "effort" in output_config and not _model_accepts_output_config_effort(model)
    if drop_effort:
        from litellm._logging import verbose_logger

        verbose_logger.debug(
            "Dropping unsupported output_config.effort for azure_ai model=%s "
            "(no supports_output_config in the model map)",
            model,
        )

    sanitized: Final = (
        {k: v for k, v in output_config.items() if k != "effort"}  # mutable-ok: request dict mutated in place
        if drop_effort
        else output_config
    )
    if sanitized:
        data["output_config"] = sanitized  # rebind-ok: out-param store like siblings
    else:
        data.pop("output_config", None)
