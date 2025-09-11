# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from typing_extensions import TypedDict

from .dpo_hyperparameters_param import DpoHyperparametersParam

__all__ = ["DpoMethodParam"]


class DpoMethodParam(TypedDict, total=False):
    hyperparameters: DpoHyperparametersParam
    """The hyperparameters used for the DPO fine-tuning job."""
