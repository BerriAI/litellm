class SchedulerAlreadyRunningError(Exception):
    """Raised when attempting to start or configure the scheduler when it's already running."""

    def __str__(self):
        return "Scheduler is already running"


class SchedulerNotRunningError(Exception):
    """Raised when attempting to shutdown the scheduler when it's not running."""

    def __str__(self):
        return "Scheduler is not running"
