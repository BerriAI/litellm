# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from typing_extensions import TypedDict

from .supervised_hyperparameters_param import SupervisedHyperparametersParam

__all__ = ["SupervisedMethodParam"]


class SupervisedMethodParam(TypedDict, total=False):
    hyperparameters: SupervisedHyperparametersParam
    """The hyperparameters used for the fine-tuning job."""
