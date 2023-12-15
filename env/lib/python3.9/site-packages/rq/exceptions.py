class NoSuchJobError(Exception):
    pass


class DeserializationError(Exception):
    pass


class InvalidJobDependency(Exception):
    pass


class InvalidJobOperationError(Exception):
    pass


class InvalidJobOperation(Exception):
    pass


class DequeueTimeout(Exception):
    pass


class ShutDownImminentException(Exception):
    def __init__(self, msg, extra_info):
        self.extra_info = extra_info
        super().__init__(msg)


class TimeoutFormatError(Exception):
    pass


class AbandonedJobError(Exception):
    pass
