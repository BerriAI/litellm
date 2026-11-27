import re
from typing import Annotated, Final

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import ConfigDict, TypeAdapter

from litellm.constants import GNANI_BATCH_ROUTE_PREFIX
from litellm.litellm_core_utils.resource_ownership import is_proxy_admin
from litellm.litellm_core_utils.url_utils import join_api_base
from litellm.llms.gnani.batches.jobs import batch_api_base
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (
    pass_through_request,  # pyright: ignore[reportUnknownVariableType]  # the shared relay has a legacy untyped response
)
from litellm.proxy.pass_through_endpoints.passthrough_endpoint_router import PassthroughEndpointRouter
from litellm.types.proxy.auth.user_api_key_auth import UserAPIKeyAuth

router: Final = APIRouter(prefix=GNANI_BATCH_ROUTE_PREFIX, tags=["Gnani Batch Transcription", "pass-through"])
_CREDENTIALS: Final = PassthroughEndpointRouter()
_RESPONSE: Final = TypeAdapter(Response, config=ConfigDict(arbitrary_types_allowed=True))
_JOB_PATH: Final = re.compile(r"jobs/[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}(/start|/cancel|/files)?")


def _batch_methods(path: str) -> tuple[str, ...]:
    if path == "jobs":
        return ("GET", "POST")
    if path == "v2/jobs":
        return ("POST",)
    job: Final = _JOB_PATH.fullmatch(path)
    if job is None:
        return ()
    return ("POST",) if job[1] in ("/start", "/cancel") else ("GET",)


@router.get("/{endpoint:path}", response_model=dict[str, object])
@router.post(
    "/{endpoint:path}",
    response_model=dict[str, object],
    responses={
        201: {"model": dict[str, object], "description": "Job created"},
        202: {"model": dict[str, object], "description": "Job accepted for processing"},
    },
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {"schema": {"type": "object", "additionalProperties": True}},
                "multipart/form-data": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "config": {"type": "string", "description": "Gnani transcription configuration as JSON"},
                            "files": {"type": "array", "items": {"type": "string", "format": "binary"}},
                            "callback_url": {"type": "string", "format": "uri"},
                        },
                        "required": ["config", "files"],
                    },
                },
            },
        }
    },
)
async def gnani_batch(
    endpoint: str,
    request: Request,
    caller: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
) -> Response:
    """Create, start, inspect, list and cancel Gnani audio jobs on the proxy's provider account.

    Upload audio or submit cloud URLs to `jobs`. Use `v2/jobs` for presigned uploads.
    Set `config.with_diarization` for speaker labels and `callback_url` for job notifications.
    Fetch transcript links from `jobs/{job_id}/files`. Shared jobs require a proxy admin key.
    """
    methods: Final = _batch_methods(endpoint)
    if not methods:
        raise HTTPException(404, "Unknown Gnani batch endpoint")
    if request.method not in methods:
        raise HTTPException(405, "Unsupported Gnani batch method", headers={"Allow": ", ".join(methods)})
    if not is_proxy_admin(caller):
        raise HTTPException(403, "Only proxy admins can manage jobs on the shared Gnani account")
    credentials: Final = _CREDENTIALS.get_proxy_wide_credentials("gnani")
    if credentials is None:
        raise HTTPException(503, "Configure a Gnani pass-through deployment or set GNANI_API_KEY")
    multipart: Final = (
        request.headers.get("content-type", "").partition(";")[0].strip().lower() == "multipart/form-data"
    )
    forwarded: Final = ("content-type", "accept", "content-length") if multipart else ("content-type", "accept")
    headers: Final = {
        **{name: request.headers[name] for name in forwarded if name in request.headers},
        "X-API-Key-ID": credentials.api_key,
    }
    return _RESPONSE.validate_python(
        await pass_through_request(
            request=request,
            target=join_api_base(batch_api_base(credentials.api_base), endpoint),
            custom_headers=dict(headers),
            user_api_key_dict=caller,
            custom_llm_provider="gnani",
            upstream_headers_filter=lambda _: headers,
            raw_request_body=request.stream() if multipart else None,
            allow_redirects=False,
        )
    )
