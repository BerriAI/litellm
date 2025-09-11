# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import Optional

from ..._models import BaseModel
from .dpo_hyperparameters import DpoHyperparameters

__all__ = ["DpoMethod"]


class DpoMethod(BaseModel):
    hyperparameters: Optional[DpoHyperparameters] = None
    """The hyperparameters used for the DPO fine-tuning job."""
