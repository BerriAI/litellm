# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import Optional

from ..._models import BaseModel
from .supervised_hyperparameters import SupervisedHyperparameters

__all__ = ["SupervisedMethod"]


class SupervisedMethod(BaseModel):
    hyperparameters: Optional[SupervisedHyperparameters] = None
    """The hyperparameters used for the fine-tuning job."""
