class LifespanFailure(Exception):
    """Raise when a lifespan failure event is sent by an application."""


class LifespanUnsupported(Exception):
    """Raise when lifespan events are not supported by an application."""


class UnexpectedMessage(Exception):
    """Raise when an unexpected message type is received during an ASGI cycle."""


class ConfigurationError(Exception):
    """Raise when an error occurs parsing configuration."""
