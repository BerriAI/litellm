# File generated from our OpenAPI spec by Stainless.

from typing import List, Optional
from typing_extensions import Literal

from .._models import BaseModel
from .file_object import FileObject
from .fine_tune_event import FineTuneEvent

__all__ = ["FineTune", "Hyperparams"]


class Hyperparams(BaseModel):
    batch_size: int
    """The batch size to use for training.

    The batch size is the number of training examples used to train a single forward
    and backward pass.
    """

    learning_rate_multiplier: float
    """The learning rate multiplier to use for training."""

    n_epochs: int
    """The number of epochs to train the model for.

    An epoch refers to one full cycle through the training dataset.
    """

    prompt_loss_weight: float
    """The weight to use for loss on the prompt tokens."""

    classification_n_classes: Optional[int] = None
    """The number of classes to use for computing classification metrics."""

    classification_positive_class: Optional[str] = None
    """The positive class to use for computing classification metrics."""

    compute_classification_metrics: Optional[bool] = None
    """
    The classification metrics to compute using the validation dataset at the end of
    every epoch.
    """


class FineTune(BaseModel):
    id: str
    """The object identifier, which can be referenced in the API endpoints."""

    created_at: int
    """The Unix timestamp (in seconds) for when the fine-tuning job was created."""

    fine_tuned_model: Optional[str]
    """The name of the fine-tuned model that is being created."""

    hyperparams: Hyperparams
    """The hyperparameters used for the fine-tuning job.

    See the
    [fine-tuning guide](https://platform.openai.com/docs/guides/legacy-fine-tuning/hyperparameters)
    for more details.
    """

    model: str
    """The base model that is being fine-tuned."""

    object: Literal["fine-tune"]
    """The object type, which is always "fine-tune"."""

    organization_id: str
    """The organization that owns the fine-tuning job."""

    result_files: List[FileObject]
    """The compiled results files for the fine-tuning job."""

    status: str
    """
    The current status of the fine-tuning job, which can be either `created`,
    `running`, `succeeded`, `failed`, or `cancelled`.
    """

    training_files: List[FileObject]
    """The list of files used for training."""

    updated_at: int
    """The Unix timestamp (in seconds) for when the fine-tuning job was last updated."""

    validation_files: List[FileObject]
    """The list of files used for validation."""

    events: Optional[List[FineTuneEvent]] = None
    """
    The list of events that have been observed in the lifecycle of the FineTune job.
    """
