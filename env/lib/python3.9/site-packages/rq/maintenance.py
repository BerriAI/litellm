from typing import TYPE_CHECKING

from .queue import Queue
from .utils import as_text

if TYPE_CHECKING:
    from .worker import BaseWorker


def clean_intermediate_queue(worker: 'BaseWorker', queue: Queue) -> None:
    """
    Check whether there are any jobs stuck in the intermediate queue.

    A job may be stuck in the intermediate queue if a worker has successfully dequeued a job
    but was not able to push it to the StartedJobRegistry. This may happen in rare cases
    of hardware or network failure.

    We consider a job to be stuck in the intermediate queue if it doesn't exist in the StartedJobRegistry.
    """
    job_ids = [as_text(job_id) for job_id in queue.connection.lrange(queue.intermediate_queue_key, 0, -1)]
    for job_id in job_ids:
        if job_id not in queue.started_job_registry:
            job = queue.fetch_job(job_id)
            if job:
                worker.handle_job_failure(job, queue, exc_string='Job was stuck in intermediate queue.')
            queue.connection.lrem(queue.intermediate_queue_key, 1, job_id)
