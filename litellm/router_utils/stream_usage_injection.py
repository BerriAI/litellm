from collections.abc import Mapping, Sequence


def router_config_requests_stream_usage(
    request_kwargs: Mapping[str, object], configured_params: Sequence[Mapping[str, object]]
) -> bool:
    if request_kwargs.get("stream") is not True or "stream_options" in request_kwargs:
        return False
    return any(_includes_usage(params.get("stream_options")) for params in configured_params)


def _includes_usage(stream_options: object) -> bool:
    match stream_options:
        case {"include_usage": True}:
            return True
        case _:
            return False
