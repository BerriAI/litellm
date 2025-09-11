# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import Union
from typing_extensions import Annotated, TypeAlias

from ..._utils import PropertyInfo
from .batch_failed_webhook_event import BatchFailedWebhookEvent
from .batch_expired_webhook_event import BatchExpiredWebhookEvent
from .batch_cancelled_webhook_event import BatchCancelledWebhookEvent
from .batch_completed_webhook_event import BatchCompletedWebhookEvent
from .eval_run_failed_webhook_event import EvalRunFailedWebhookEvent
from .response_failed_webhook_event import ResponseFailedWebhookEvent
from .eval_run_canceled_webhook_event import EvalRunCanceledWebhookEvent
from .eval_run_succeeded_webhook_event import EvalRunSucceededWebhookEvent
from .response_cancelled_webhook_event import ResponseCancelledWebhookEvent
from .response_completed_webhook_event import ResponseCompletedWebhookEvent
from .response_incomplete_webhook_event import ResponseIncompleteWebhookEvent
from .fine_tuning_job_failed_webhook_event import FineTuningJobFailedWebhookEvent
from .fine_tuning_job_cancelled_webhook_event import FineTuningJobCancelledWebhookEvent
from .fine_tuning_job_succeeded_webhook_event import FineTuningJobSucceededWebhookEvent

__all__ = ["UnwrapWebhookEvent"]

UnwrapWebhookEvent: TypeAlias = Annotated[
    Union[
        BatchCancelledWebhookEvent,
        BatchCompletedWebhookEvent,
        BatchExpiredWebhookEvent,
        BatchFailedWebhookEvent,
        EvalRunCanceledWebhookEvent,
        EvalRunFailedWebhookEvent,
        EvalRunSucceededWebhookEvent,
        FineTuningJobCancelledWebhookEvent,
        FineTuningJobFailedWebhookEvent,
        FineTuningJobSucceededWebhookEvent,
        ResponseCancelledWebhookEvent,
        ResponseCompletedWebhookEvent,
        ResponseFailedWebhookEvent,
        ResponseIncompleteWebhookEvent,
    ],
    PropertyInfo(discriminator="type"),
]
