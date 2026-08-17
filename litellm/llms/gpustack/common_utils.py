from typing import Final
from urllib.parse import urlsplit, urlunsplit

from litellm.secret_managers.main import get_secret_str


def get_gpustack_api_base(api_base: str | None) -> str:
    resolved_api_base: Final = api_base or get_secret_str("GPUSTACK_API_BASE")
    if resolved_api_base is None:
        raise ValueError("api_base is required for GPUStack. Set it in the call or via GPUSTACK_API_BASE.")
    return resolved_api_base


def get_gpustack_endpoint(api_base: str | None, endpoint: str) -> str:
    parsed_api_base: Final = urlsplit(get_gpustack_api_base(api_base))
    normalized_path: Final = parsed_api_base.path.rstrip("/")
    normalized_endpoint: Final = endpoint.strip("/")
    endpoint_path: str
    if normalized_path.endswith(f"/{normalized_endpoint}"):
        endpoint_path = normalized_path
    elif normalized_path.endswith("/v1"):
        endpoint_path = f"{normalized_path}/{normalized_endpoint}"
    else:
        endpoint_path = f"{normalized_path}/v1/{normalized_endpoint}"
    return urlunsplit(parsed_api_base._replace(path=endpoint_path))


def get_gpustack_api_key(api_key: str | None) -> str | None:
    return api_key or get_secret_str("GPUSTACK_API_KEY")


def get_gpustack_headers(
    headers: dict[str, object],
    api_key: str | None,
    *,
    include_accept: bool = False,
) -> dict[str, object]:
    resolved_api_key: Final = get_gpustack_api_key(api_key)
    deduplicated_headers_by_name: Final = {
        header_name.lower(): (header_name, header_value) for header_name, header_value in headers.items()
    }
    deduplicated_headers: Final = {
        header_name: header_value for header_name, header_value in deduplicated_headers_by_name.values()
    }
    header_names: Final = set(deduplicated_headers_by_name)
    default_headers: Final = {
        **({"Content-Type": "application/json"} if "content-type" not in header_names else {}),
        **({"Accept": "application/json"} if include_accept and "accept" not in header_names else {}),
        **(
            {"Authorization": f"Bearer {resolved_api_key}"}
            if resolved_api_key is not None and "authorization" not in header_names
            else {}
        ),
    }
    return {**default_headers, **deduplicated_headers}


def strip_gpustack_model_prefix(model: str) -> str:
    if model.startswith("gpustack/"):
        return model.replace("gpustack/", "", 1)
    return model
