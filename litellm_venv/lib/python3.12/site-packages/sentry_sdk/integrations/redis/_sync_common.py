import sentry_sdk
from sentry_sdk.consts import OP
from sentry_sdk.integrations.redis.consts import SPAN_ORIGIN
from sentry_sdk.integrations.redis.modules.caches import (
    _compile_cache_span_properties,
    _set_cache_data,
)
from sentry_sdk.integrations.redis.modules.queries import _compile_db_span_properties
from sentry_sdk.integrations.redis.utils import (
    _set_client_data,
    _set_pipeline_data,
)
from sentry_sdk.tracing import Span
from sentry_sdk.utils import capture_internal_exceptions

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any


def patch_redis_pipeline(
    pipeline_cls,
    is_cluster,
    get_command_args_fn,
    set_db_data_fn,
):
    # type: (Any, bool, Any, Callable[[Span, Any], None]) -> None
    old_execute = pipeline_cls.execute

    from sentry_sdk.integrations.redis import RedisIntegration

    def sentry_patched_execute(self, *args, **kwargs):
        # type: (Any, *Any, **Any) -> Any
        if sentry_sdk.get_client().get_integration(RedisIntegration) is None:
            return old_execute(self, *args, **kwargs)

        with sentry_sdk.start_span(
            op=OP.DB_REDIS,
            name="redis.pipeline.execute",
            origin=SPAN_ORIGIN,
        ) as span:
            with capture_internal_exceptions():
                set_db_data_fn(span, self)
                _set_pipeline_data(
                    span,
                    is_cluster,
                    get_command_args_fn,
                    False if is_cluster else self.transaction,
                    self.command_stack,
                )

            return old_execute(self, *args, **kwargs)

    pipeline_cls.execute = sentry_patched_execute


def patch_redis_client(cls, is_cluster, set_db_data_fn):
    # type: (Any, bool, Callable[[Span, Any], None]) -> None
    """
    This function can be used to instrument custom redis client classes or
    subclasses.
    """
    old_execute_command = cls.execute_command

    from sentry_sdk.integrations.redis import RedisIntegration

    def sentry_patched_execute_command(self, name, *args, **kwargs):
        # type: (Any, str, *Any, **Any) -> Any
        integration = sentry_sdk.get_client().get_integration(RedisIntegration)
        if integration is None:
            return old_execute_command(self, name, *args, **kwargs)

        cache_properties = _compile_cache_span_properties(
            name,
            args,
            kwargs,
            integration,
        )

        cache_span = None
        if cache_properties["is_cache_key"] and cache_properties["op"] is not None:
            cache_span = sentry_sdk.start_span(
                op=cache_properties["op"],
                name=cache_properties["description"],
                origin=SPAN_ORIGIN,
            )
            cache_span.__enter__()

        db_properties = _compile_db_span_properties(integration, name, args)

        db_span = sentry_sdk.start_span(
            op=db_properties["op"],
            name=db_properties["description"],
            origin=SPAN_ORIGIN,
        )
        db_span.__enter__()

        set_db_data_fn(db_span, self)
        _set_client_data(db_span, is_cluster, name, *args)

        value = old_execute_command(self, name, *args, **kwargs)

        db_span.__exit__(None, None, None)

        if cache_span:
            _set_cache_data(cache_span, self, cache_properties, value)
            cache_span.__exit__(None, None, None)

        return value

    cls.execute_command = sentry_patched_execute_command
