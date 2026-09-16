"""Explicit backend-form selection for SAP AI Core, encoded in the model string.

`get_llm_provider` strips only the leading `sap/`, so a `deployment/` segment survives on the
model name that reaches config selection (which sees `(model, provider)` and never `litellm_params`).
That makes the model string the only place a per-request backend choice can live:

    sap/<model>              -> orchestration (POST {url}/v2/completion), backward compatible
    sap/deployment/<model>   -> direct connect (POST {deployment_url}/invoke)
"""

from __future__ import annotations

from enum import Enum
from typing import Final


class SapBackendForm(str, Enum):
    ORCHESTRATION = "orchestration"
    DEPLOYMENT = "deployment"


_PROVIDER_PREFIX: Final = "sap/"
_DEPLOYMENT_PREFIX: Final = f"{SapBackendForm.DEPLOYMENT.value}/"


def split_sap_submode(model: str) -> tuple[SapBackendForm, str]:
    """Return the requested backend form and the bare model name.

    The request path strips the leading `sap/` via `get_llm_provider` before form selection, but
    capability probes (`get_supported_openai_params`, `model_group/info`) pass the model group name
    with the `sap/` provider prefix still attached. Tolerating it here keeps both paths agreeing on
    the form, so reasoning params are mapped identically whether or not the provider prefix survived.
    """
    unprefixed: Final = model[len(_PROVIDER_PREFIX) :] if model.startswith(_PROVIDER_PREFIX) else model
    if unprefixed.startswith(_DEPLOYMENT_PREFIX):
        return SapBackendForm.DEPLOYMENT, unprefixed[len(_DEPLOYMENT_PREFIX) :]
    return SapBackendForm.ORCHESTRATION, unprefixed
