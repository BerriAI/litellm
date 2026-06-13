"""openrouter chat-completion response JSON -> IR ``ChatResponse``.

``OpenrouterConfig.transform_response`` is super() — the base GPT transform
over a FRESH ``ModelResponse`` (model=None), so the response model is the
BARE wire model (no ``openrouter/`` prefix; the xai R4 / deepseek pattern) —
PLUS one envelope post-step: ``usage.cost`` from the raw body is copied into
``_hidden_params["additional_headers"]["llm_provider-x-litellm-response-cost"]``
as a float (the cost-calculator feed). The body itself is untouched —
``usage.cost``/``cost_details`` ride the unknown-usage-key mirror verbatim
through BOTH sides (parity-pinned), so the shared openai parser is exact for
the dump and the hidden-params header is a SEAM/FORK OBLIGATION: the future
openrouter completion() fork must read ``usage.cost`` off the v2 body and
set the same header (pinned by
test_differential_openrouter_response.test_v1_cost_hidden_param_is_the_fork_obligation).
"""

from __future__ import annotations

from ..openai_compat.response import parse_response

__all__ = ("parse_response",)
