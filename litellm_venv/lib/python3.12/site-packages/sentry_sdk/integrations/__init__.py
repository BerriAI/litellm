from abc import ABC, abstractmethod
from threading import Lock

from sentry_sdk.utils import logger

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Callable
    from typing import Dict
    from typing import Iterator
    from typing import List
    from typing import Optional
    from typing import Set
    from typing import Type
    from typing import Union


_DEFAULT_FAILED_REQUEST_STATUS_CODES = frozenset(range(500, 600))


_installer_lock = Lock()

# Set of all integration identifiers we have attempted to install
_processed_integrations = set()  # type: Set[str]

# Set of all integration identifiers we have actually installed
_installed_integrations = set()  # type: Set[str]


def _generate_default_integrations_iterator(
    integrations,  # type: List[str]
    auto_enabling_integrations,  # type: List[str]
):
    # type: (...) -> Callable[[bool], Iterator[Type[Integration]]]

    def iter_default_integrations(with_auto_enabling_integrations):
        # type: (bool) -> Iterator[Type[Integration]]
        """Returns an iterator of the default integration classes:"""
        from importlib import import_module

        if with_auto_enabling_integrations:
            all_import_strings = integrations + auto_enabling_integrations
        else:
            all_import_strings = integrations

        for import_string in all_import_strings:
            try:
                module, cls = import_string.rsplit(".", 1)
                yield getattr(import_module(module), cls)
            except (DidNotEnable, SyntaxError) as e:
                logger.debug(
                    "Did not import default integration %s: %s", import_string, e
                )

    if isinstance(iter_default_integrations.__doc__, str):
        for import_string in integrations:
            iter_default_integrations.__doc__ += "\n- `{}`".format(import_string)

    return iter_default_integrations


_DEFAULT_INTEGRATIONS = [
    # stdlib/base runtime integrations
    "sentry_sdk.integrations.argv.ArgvIntegration",
    "sentry_sdk.integrations.atexit.AtexitIntegration",
    "sentry_sdk.integrations.dedupe.DedupeIntegration",
    "sentry_sdk.integrations.excepthook.ExcepthookIntegration",
    "sentry_sdk.integrations.logging.LoggingIntegration",
    "sentry_sdk.integrations.modules.ModulesIntegration",
    "sentry_sdk.integrations.stdlib.StdlibIntegration",
    "sentry_sdk.integrations.threading.ThreadingIntegration",
]

_AUTO_ENABLING_INTEGRATIONS = [
    "sentry_sdk.integrations.aiohttp.AioHttpIntegration",
    "sentry_sdk.integrations.anthropic.AnthropicIntegration",
    "sentry_sdk.integrations.ariadne.AriadneIntegration",
    "sentry_sdk.integrations.arq.ArqIntegration",
    "sentry_sdk.integrations.asyncpg.AsyncPGIntegration",
    "sentry_sdk.integrations.boto3.Boto3Integration",
    "sentry_sdk.integrations.bottle.BottleIntegration",
    "sentry_sdk.integrations.celery.CeleryIntegration",
    "sentry_sdk.integrations.chalice.ChaliceIntegration",
    "sentry_sdk.integrations.clickhouse_driver.ClickhouseDriverIntegration",
    "sentry_sdk.integrations.cohere.CohereIntegration",
    "sentry_sdk.integrations.django.DjangoIntegration",
    "sentry_sdk.integrations.falcon.FalconIntegration",
    "sentry_sdk.integrations.fastapi.FastApiIntegration",
    "sentry_sdk.integrations.flask.FlaskIntegration",
    "sentry_sdk.integrations.gql.GQLIntegration",
    "sentry_sdk.integrations.graphene.GrapheneIntegration",
    "sentry_sdk.integrations.httpx.HttpxIntegration",
    "sentry_sdk.integrations.huey.HueyIntegration",
    "sentry_sdk.integrations.huggingface_hub.HuggingfaceHubIntegration",
    "sentry_sdk.integrations.langchain.LangchainIntegration",
    "sentry_sdk.integrations.litestar.LitestarIntegration",
    "sentry_sdk.integrations.loguru.LoguruIntegration",
    "sentry_sdk.integrations.openai.OpenAIIntegration",
    "sentry_sdk.integrations.pymongo.PyMongoIntegration",
    "sentry_sdk.integrations.pyramid.PyramidIntegration",
    "sentry_sdk.integrations.quart.QuartIntegration",
    "sentry_sdk.integrations.redis.RedisIntegration",
    "sentry_sdk.integrations.rq.RqIntegration",
    "sentry_sdk.integrations.sanic.SanicIntegration",
    "sentry_sdk.integrations.sqlalchemy.SqlalchemyIntegration",
    "sentry_sdk.integrations.starlette.StarletteIntegration",
    "sentry_sdk.integrations.starlite.StarliteIntegration",
    "sentry_sdk.integrations.strawberry.StrawberryIntegration",
    "sentry_sdk.integrations.tornado.TornadoIntegration",
]

iter_default_integrations = _generate_default_integrations_iterator(
    integrations=_DEFAULT_INTEGRATIONS,
    auto_enabling_integrations=_AUTO_ENABLING_INTEGRATIONS,
)

del _generate_default_integrations_iterator


_MIN_VERSIONS = {
    "aiohttp": (3, 4),
    "anthropic": (0, 16),
    "ariadne": (0, 20),
    "arq": (0, 23),
    "asyncpg": (0, 23),
    "beam": (2, 12),
    "boto3": (1, 12),  # botocore
    "bottle": (0, 12),
    "celery": (4, 4, 7),
    "chalice": (1, 16, 0),
    "clickhouse_driver": (0, 2, 0),
    "django": (1, 8),
    "dramatiq": (1, 9),
    "falcon": (1, 4),
    "fastapi": (0, 79, 0),
    "flask": (0, 10),
    "gql": (3, 4, 1),
    "graphene": (3, 3),
    "grpc": (1, 32, 0),  # grpcio
    "huggingface_hub": (0, 22),
    "langchain": (0, 0, 210),
    "launchdarkly": (9, 8, 0),
    "openai": (1, 0, 0),
    "openfeature": (0, 7, 1),
    "quart": (0, 16, 0),
    "ray": (2, 7, 0),
    "requests": (2, 0, 0),
    "rq": (0, 6),
    "sanic": (0, 8),
    "sqlalchemy": (1, 2),
    "starlite": (1, 48),
    "strawberry": (0, 209, 5),
    "tornado": (6, 0),
    "typer": (0, 15),
    "unleash": (6, 0, 1),
}


def setup_integrations(
    integrations,
    with_defaults=True,
    with_auto_enabling_integrations=False,
    disabled_integrations=None,
):
    # type: (Sequence[Integration], bool, bool, Optional[Sequence[Union[type[Integration], Integration]]]) -> Dict[str, Integration]
    """
    Given a list of integration instances, this installs them all.

    When `with_defaults` is set to `True` all default integrations are added
    unless they were already provided before.

    `disabled_integrations` takes precedence over `with_defaults` and
    `with_auto_enabling_integrations`.
    """
    integrations = dict(
        (integration.identifier, integration) for integration in integrations or ()
    )

    logger.debug("Setting up integrations (with default = %s)", with_defaults)

    # Integrations that will not be enabled
    disabled_integrations = [
        integration if isinstance(integration, type) else type(integration)
        for integration in disabled_integrations or []
    ]

    # Integrations that are not explicitly set up by the user.
    used_as_default_integration = set()

    if with_defaults:
        for integration_cls in iter_default_integrations(
            with_auto_enabling_integrations
        ):
            if integration_cls.identifier not in integrations:
                instance = integration_cls()
                integrations[instance.identifier] = instance
                used_as_default_integration.add(instance.identifier)

    for identifier, integration in integrations.items():
        with _installer_lock:
            if identifier not in _processed_integrations:
                if type(integration) in disabled_integrations:
                    logger.debug("Ignoring integration %s", identifier)
                else:
                    logger.debug(
                        "Setting up previously not enabled integration %s", identifier
                    )
                    try:
                        type(integration).setup_once()
                    except DidNotEnable as e:
                        if identifier not in used_as_default_integration:
                            raise

                        logger.debug(
                            "Did not enable default integration %s: %s", identifier, e
                        )
                    else:
                        _installed_integrations.add(identifier)

                _processed_integrations.add(identifier)

    integrations = {
        identifier: integration
        for identifier, integration in integrations.items()
        if identifier in _installed_integrations
    }

    for identifier in integrations:
        logger.debug("Enabling integration %s", identifier)

    return integrations


def _check_minimum_version(integration, version, package=None):
    # type: (type[Integration], Optional[tuple[int, ...]], Optional[str]) -> None
    package = package or integration.identifier

    if version is None:
        raise DidNotEnable(f"Unparsable {package} version.")

    min_version = _MIN_VERSIONS.get(integration.identifier)
    if min_version is None:
        return

    if version < min_version:
        raise DidNotEnable(
            f"Integration only supports {package} {'.'.join(map(str, min_version))} or newer."
        )


class DidNotEnable(Exception):  # noqa: N818
    """
    The integration could not be enabled due to a trivial user error like
    `flask` not being installed for the `FlaskIntegration`.

    This exception is silently swallowed for default integrations, but reraised
    for explicitly enabled integrations.
    """


class Integration(ABC):
    """Baseclass for all integrations.

    To accept options for an integration, implement your own constructor that
    saves those options on `self`.
    """

    install = None
    """Legacy method, do not implement."""

    identifier = None  # type: str
    """String unique ID of integration type"""

    @staticmethod
    @abstractmethod
    def setup_once():
        # type: () -> None
        """
        Initialize the integration.

        This function is only called once, ever. Configuration is not available
        at this point, so the only thing to do here is to hook into exception
        handlers, and perhaps do monkeypatches.

        Inside those hooks `Integration.current` can be used to access the
        instance again.
        """
        pass
