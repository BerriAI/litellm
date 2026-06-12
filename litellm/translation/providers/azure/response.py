"""Azure OpenAI response parsing.

Azure's live response path is the SAME normalizer as openai:
``convert_to_model_response_object`` over the SDK dump (azure.py:358-375;
``AzureOpenAIConfig.transform_response`` raises NotImplementedError). The one
azure-only argument is ``convert_tool_call_to_json_mode=json_mode``, and
``json_mode`` is only ever set by the synthetic json-tool response_format
strategy -- which the azure serializer fails closed on -- so a v2-sent
request can never need the tool->content conversion and the parser is the
openai parser re-exported. Azure's response extras (choice-level
``content_filter_results``, top-level ``prompt_filter_results``) ride the
same provider_specific_fields / unknown-top-level-key paths v1 uses.
"""

from __future__ import annotations

from ..openai_compat.response import parse_response

__all__ = ("parse_response",)
