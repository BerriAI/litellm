"""
Prompt-management groups on the Rust router core (``_is_prompt_management_model`` and
``_prompt_management_factory``): a group of exactly one deployment whose model names a prompt
management integration (``langfuse/...``) or ``litellm_agent/...``. The request is not routed with
retries and fallbacks: the deployment is picked, its prompt rendered by the request's logging
object, and the rendered model called directly, or routed when it names a group.
"""

from collections.abc import Mapping
from typing import TYPE_CHECKING, Final

from litellm.router_rust_endpoints import PLANS
from litellm.router_rust_shaping import as_dict, update_kwargs_with_deployment

if TYPE_CHECKING:
    from litellm.router_rust_core import RustCore

_PROMPT_MANAGEMENT_PARAMS: Final = frozenset(
    {"bitbucket_config", "dotprompt_config", "prompt_id", "prompt_variables", "prompt_label", "prompt_version"}
)


def is_prompt_management_model(core: "RustCore", model: str) -> bool:
    """`_is_prompt_management_model`: one deployment, whose model's provider prefix is a known
    custom-logger-compatible callback."""
    model_list: Final = core._hosted._core_get_model_list(model_name=model)  # pyright: ignore[reportPrivateUsage]  # the core's host registry
    if model_list is None or len(model_list) != 1:
        return False
    litellm_model: Final = as_dict(model_list[0].get("litellm_params")).get("model")
    if not isinstance(litellm_model, str) or "/" not in litellm_model:
        return False
    known: Final = core._host.dynamic().known_custom_logger_compatible_callbacks  # pyright: ignore[reportPrivateUsage]  # the core's host
    return litellm_model.split("/")[0] in known


def _logging_and_kwargs(
    kwargs: dict[str, object],  # mutable-ok: the request kwargs the Python router mutates in place
) -> tuple[object, dict[str, object]]:  # mutable-ok: same
    """The request's logging object, else the one `function_setup` builds (with the kwargs it returns)."""
    import uuid

    from litellm.utils import Rules, function_setup, get_utc_datetime

    logging_obj: Final = kwargs.get("litellm_logging_obj")
    if logging_obj is not None:
        return logging_obj, kwargs
    return function_setup(  # pyright: ignore[reportUnknownVariableType]  # litellm's untyped helpers
        **{  # pyright: ignore[reportArgumentType]  # the request's keyword payload
            **kwargs,
            "original_function": "acompletion",
            "rules_obj": Rules(),
            "start_time": get_utc_datetime(),
            "litellm_call_id": kwargs.get("litellm_call_id") or str(uuid.uuid4()),
        }
    )


async def prompt_management_acompletion(
    core: "RustCore",
    model: str,
    messages: object,
    request: Mapping[str, object],
) -> object:
    """`_prompt_management_factory`, on the request as `acompletion` prepared it."""
    from litellm.utils import get_non_default_completion_params

    # The routing function is no request field: the routed call below drives its own plan.
    logging_obj, kwargs = _logging_and_kwargs(
        {key: value for key, value in request.items() if key != "original_function"}
    )
    caller_timeout: Final = kwargs.get("timeout") or kwargs.get("request_timeout")
    caller_stream_timeout: Final = kwargs.get("stream_timeout")
    specific_deployment: Final = kwargs.pop("specific_deployment", None)
    picked: Final = await core.select(
        model,
        kwargs,  # pyright: ignore[reportUnknownArgumentType]  # the request kwargs
        kind="pick",
        specific_deployment=bool(specific_deployment),
        messages=messages,
    )
    deployment: Final = as_dict(picked)
    update_kwargs_with_deployment(
        core._shaping,  # pyright: ignore[reportPrivateUsage]  # the core's own settings
        deployment,
        kwargs,  # pyright: ignore[reportUnknownArgumentType]  # the request kwargs
        cache=core._cache,  # pyright: ignore[reportPrivateUsage]  # the routing cache the shaping refunds into
        splits_stream_timeout=True,
    )
    params: Final = as_dict(deployment.get("litellm_params"))
    data: Final = dict(params)
    litellm_model: Final = data.get("model")
    is_litellm_agent_model: Final = isinstance(litellm_model, str) and litellm_model.startswith("litellm_agent/")
    prompt_id: Final = kwargs.get("prompt_id") or params.get("prompt_id")
    prompt_variables: Final = kwargs.get("prompt_variables") or params.get("prompt_variables")
    prompt_label: Final = kwargs.get("prompt_label") or params.get("prompt_label")
    if not is_litellm_agent_model and (prompt_id is None or not isinstance(prompt_id, str)):
        raise ValueError(f"Prompt ID is not set or not a string. Got={prompt_id}, type={type(prompt_id)}")
    if prompt_variables is not None and not isinstance(prompt_variables, dict):
        raise ValueError(
            f"Prompt variables is set but not a dictionary. Got={prompt_variables}, type={type(prompt_variables)}"
        )
    rendered_model, rendered_messages, optional_params = logging_obj.get_chat_completion_prompt(  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType, reportAttributeAccessIssue, reportOptionalMemberAccess]  # the request's Logging object
        model=litellm_model,  # pyright: ignore[reportArgumentType]  # a request value, passed as the Python router passes it
        messages=messages,  # pyright: ignore[reportArgumentType]  # a request value, passed as the Python router passes it
        non_default_params=get_non_default_completion_params(kwargs=kwargs),  # pyright: ignore[reportArgumentType]  # the request kwargs
        prompt_id=prompt_id,  # pyright: ignore[reportArgumentType]  # a request value, passed as the Python router passes it
        prompt_variables=prompt_variables,
        prompt_label=prompt_label,  # pyright: ignore[reportArgumentType]  # a request value, passed as the Python router passes it
        request_kwargs=kwargs,
        injected_for_every_deployment=True,
    )
    filtered_data: Final = {key: value for key, value in data.items() if key not in _PROMPT_MANAGEMENT_PARAMS}
    final: Final[dict[str, object]] = {**filtered_data, **kwargs, **optional_params}  # pyright: ignore[reportUnknownVariableType]  # mutable-ok: the call's kwargs
    final["model"] = rendered_model
    final["messages"] = rendered_messages
    final["litellm_logging_obj"] = logging_obj
    final["prompt_id"] = prompt_id
    final["prompt_variables"] = prompt_variables
    final["prompt_label"] = prompt_label
    target_list: Final = core._hosted._core_get_model_list(model_name=str(rendered_model))  # pyright: ignore[reportPrivateUsage, reportUnknownArgumentType]  # the core's host registry
    if target_list is None or len(target_list) == 0:  # a direct call to the rendered model
        return await core._host.invoke("acompletion")(**final)  # pyright: ignore[reportPrivateUsage, reportUnknownMemberType]  # the core's host
    timeouts: Final = {
        key: value
        for key, value in (
            ("timeout", caller_timeout if caller_timeout is not None else filtered_data.get("timeout")),
            (
                "stream_timeout",
                caller_stream_timeout if caller_stream_timeout is not None else filtered_data.get("stream_timeout"),
            ),
        )
        if value is not None
    }
    routed: Final = {key: value for key, value in final.items() if key not in ("timeout", "stream_timeout")}
    return await core._drive(PLANS["acompletion"], {**routed, **timeouts}, prepared_already=True)  # pyright: ignore[reportPrivateUsage]  # the core drives its own call
