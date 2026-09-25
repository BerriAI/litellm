"""
Bedrock Nova Reel video handler.

Implements the LiteLLM video surface for ``amazon.nova-reel-v1:0`` (and regional
inference-profile ids) on top of the Bedrock asynchronous invoke API:

- create:  POST {runtime}/async-invoke            (StartAsyncInvoke, SigV4-signed)
- status:  GET  {runtime}/async-invoke/{arn}      (GetAsyncInvoke, SigV4-signed)
- content: download ``output.mp4`` from the S3 output location via boto3

Mirrors the URL + signing conventions of
``litellm/llms/bedrock/embed/embedding.py`` (same async-invoke endpoints).
"""

from __future__ import annotations

import json
from collections.abc import Coroutine, Mapping, Sequence
from typing import TYPE_CHECKING, Final, TypeAlias
from urllib.parse import quote

import httpx

import litellm
from litellm._logging import verbose_logger
from litellm.llms.bedrock.videos.transformation import BedrockNovaReelVideoConfig
from litellm.secret_managers.main import get_secret
from litellm.types.llms.bedrock import (
    BedrockAsyncInvokeOutputDataConfig,
    BedrockAsyncInvokeS3OutputDataConfig,
    BedrockGetAsyncInvokeResponse,
)
from litellm.types.router import GenericLiteLLMParams
from litellm.types.videos.main import VideoObject
from litellm.types.videos.utils import decode_video_id_with_provider, extract_original_video_id

from ..base_aws_llm import (
    AWSPreparedRequest,
    BaseAWSLLM,
    Credentials,
    bedrock_bearer_token,
    pop_aws_auth_params,
)
from ..common_utils import BedrockError, redact_bedrock_headers_for_logging

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLogging

_LitellmParamsDict: TypeAlias = dict[str, object]
_ExtraHeadersDict: TypeAlias = dict[str, object]

DEFAULT_VIDEO_REGION: Final = "us-west-2"
NOVA_REEL_OUTPUT_FILENAME: Final = "output.mp4"
# S3 download client defaults when no per-call timeout is given (video_content
# threads its call timeout in; these bound the boto3 client otherwise).
DEFAULT_S3_CONNECT_TIMEOUT_S: Final = 5.0
DEFAULT_S3_READ_TIMEOUT_S: Final = 60.0


def _client_error_code(err: Exception) -> str:
    """AWS error Code from a botocore ClientError response, '' when absent."""
    response: Final = getattr(err, "response", None)
    if isinstance(response, Mapping):
        error: Final = response.get("Error")
        if isinstance(error, Mapping):
            code: Final = error.get("Code")
            if isinstance(code, str):
                return code
    return ""


def _client_error_http_status(err: Exception) -> int | None:
    """HTTPStatusCode from a botocore ClientError ResponseMetadata, None when absent."""
    response: Final = getattr(err, "response", None)
    if isinstance(response, Mapping):
        metadata: Final = response.get("ResponseMetadata")
        if isinstance(metadata, Mapping):
            status: Final = metadata.get("HTTPStatusCode")
            if isinstance(status, int):
                return status
    return None


def _sign_get_request(
    credentials: Credentials | None,
    url: str,
    headers: Mapping[str, str],
    aws_region_name: str,
    bearer_token: str | None = None,
) -> AWSPreparedRequest:
    try:
        from botocore.auth import SigV4Auth
        from botocore.awsrequest import AWSRequest
        from botocore.exceptions import NoCredentialsError
    except ImportError:
        raise ImportError("Missing boto3 to call bedrock. Run 'pip install boto3'.")

    request: Final = AWSRequest(
        method="GET",
        url=url,
        data=None,
        headers=(
            {  # mutable-ok: merged headers dict carries the bearer token
                **headers,
                "Authorization": f"Bearer {bearer_token}",
            }
            if bearer_token is not None
            else headers
        ),
    )
    if credentials is None and bearer_token is None:
        # Fail fast the same way the shared POST signer does (base_aws_llm.py):
        # an unsigned request would 403 at AWS with a far less actionable error.
        raise NoCredentialsError()
    if credentials is not None and bearer_token is None:
        SigV4Auth(credentials, "bedrock", aws_region_name).add_auth(request)
    return request.prepare()


def _region_from_invocation_arn(invocation_arn: str) -> str | None:
    """arn:aws:bedrock:{region}:{account}:async-invoke/{id} -> region."""
    parts: Final = invocation_arn.split(":")
    if len(parts) >= 4 and parts[0] == "arn":
        return parts[3] or None
    return None


def _parse_s3_uri(s3_uri: str) -> tuple[str, str]:
    """s3://bucket/optional/prefix/ -> (bucket, 'optional/prefix')."""
    if not s3_uri.startswith("s3://"):
        raise BedrockError(
            status_code=400,
            message=f"Invalid S3 output URI (expected s3://bucket/prefix): {s3_uri!r}",
        )
    trimmed: Final = s3_uri.rstrip("/")
    without_scheme: Final = trimmed[len("s3://") :]
    bucket, _, prefix = without_scheme.partition("/")
    if not bucket:
        raise BedrockError(status_code=400, message=f"Invalid S3 output URI: {s3_uri!r}")
    return bucket, prefix


def _s3_uri_from_output_config(
    output_config: BedrockAsyncInvokeOutputDataConfig | None,
) -> str | None:
    """s3Uri from the invocation's outputDataConfig.s3OutputDataConfig, if present."""
    if output_config is None:
        return None
    s3_config: Final[BedrockAsyncInvokeS3OutputDataConfig | None] = output_config.get("s3OutputDataConfig")
    if s3_config is None:
        return None
    s3_uri: Final[str | None] = s3_config.get("s3Uri")
    return s3_uri


def _params_to_dict(litellm_params: GenericLiteLLMParams | Mapping[str, object] | None) -> _LitellmParamsDict:
    """GenericLiteLLMParams is a pydantic model with dict-like access; copy to a real dict."""
    if litellm_params is None:
        return {}  # mutable-ok: empty params dict for None input
    if isinstance(litellm_params, dict):
        return dict(litellm_params)  # mutable-ok: aws_* keys are popped in place downstream
    if isinstance(litellm_params, GenericLiteLLMParams):
        return litellm_params.model_dump(exclude_none=True)
    return dict(litellm_params)  # mutable-ok: aws_* keys are popped in place downstream


def _as_generic_litellm_params(
    litellm_params: GenericLiteLLMParams | Mapping[str, object] | None,
) -> GenericLiteLLMParams:
    """Normalize handler-level litellm_params to GenericLiteLLMParams for the create transform.

    Only the clientRequestToken fallback inputs are read downstream
    (metadata.request_id, then the litellm_call_id extra field); aws_* credential
    keys are consumed from optional_params only, so nothing passed here
    double-processes credentials.
    """
    if isinstance(litellm_params, GenericLiteLLMParams):
        return litellm_params
    params: Final[GenericLiteLLMParams] = GenericLiteLLMParams()
    if litellm_params is not None:
        metadata: Final = litellm_params.get("metadata")
        if isinstance(metadata, dict):
            params.metadata = metadata  # pyright: ignore[reportAttributeAccessIssue]  # extra-allowed field
        call_id: Final = litellm_params.get("litellm_call_id")
        if isinstance(call_id, str) and call_id:
            params.litellm_call_id = call_id  # pyright: ignore[reportAttributeAccessIssue]  # @client extra field
    return params


class BedrockVideoGeneration(BaseAWSLLM):
    """
    Bedrock video generation handler for Amazon Nova Reel models.
    """

    def get_config_class(self) -> type[BedrockNovaReelVideoConfig]:
        return BedrockNovaReelVideoConfig

    def _load_credentials(
        self,
        optional_params: dict,  # mutable-ok: aws_* keys are popped in place
        aws_region_name: str | None = None,
        bearer_token: str | None = None,
    ) -> tuple[Credentials | None, str]:
        """Resolve SigV4 credentials + region the same way BedrockEmbedding does."""
        auth_params: Final = pop_aws_auth_params(optional_params)
        if aws_region_name is None:
            aws_region_name = optional_params.pop("aws_region_name", None)
        if aws_region_name is None:
            litellm_aws_region_name: Final = get_secret("AWS_REGION_NAME", None)
            if litellm_aws_region_name is not None and isinstance(litellm_aws_region_name, str):
                aws_region_name = litellm_aws_region_name
            standard_aws_region_name: Final = get_secret("AWS_REGION", None)
            if standard_aws_region_name is not None and isinstance(standard_aws_region_name, str):
                aws_region_name = standard_aws_region_name
        if aws_region_name is None:
            aws_region_name = DEFAULT_VIDEO_REGION

        credentials: Final[Credentials | None] = (
            None if bearer_token is not None else self.resolve_credentials(auth_params, aws_region_name)
        )
        return credentials, aws_region_name

    def _prepare_async_invoke_request(
        self,
        model: str,
        prompt: str,
        optional_params: _LitellmParamsDict,
        api_base: str | None,
        extra_headers: _ExtraHeadersDict | None,
        logging_obj: LiteLLMLogging | None,
        api_key: str | None = None,
        litellm_params: GenericLiteLLMParams | Mapping[str, object] | None = None,
    ) -> tuple[str, AWSPreparedRequest, bytes, _LitellmParamsDict]:
        """
        Returns (endpoint_url, prepped_request, body, data) for POST /async-invoke.

        litellm_params feeds only the clientRequestToken fallback
        (metadata.request_id); aws_* credentials are consumed from
        optional_params, never from litellm_params, so nothing is double-processed.
        """
        bearer_token: Final = bedrock_bearer_token(api_key)
        boto3_credentials_info: Final = self._get_boto_credentials_from_optional_params(
            optional_params, model, bearer_token=bearer_token
        )
        bedrock_provider: Final = self.get_bedrock_invoke_provider(model)
        model_id: Final = self.get_bedrock_model_id(
            model=model,
            provider=bedrock_provider,
            optional_params=optional_params,
        )
        _, proxy_endpoint_url = self.get_runtime_endpoint(
            api_base=api_base,
            aws_bedrock_runtime_endpoint=boto3_credentials_info.aws_bedrock_runtime_endpoint,
            aws_region_name=boto3_credentials_info.aws_region_name,
        )
        endpoint_url: Final = f"{proxy_endpoint_url.rstrip('/')}/async-invoke"

        config: Final = BedrockNovaReelVideoConfig()
        data, _, _ = config.transform_video_create_request(
            model=model_id,
            prompt=prompt,
            api_base=endpoint_url,
            video_create_optional_request_params=optional_params,
            litellm_params=_as_generic_litellm_params(litellm_params),
            headers={},  # mutable-ok: transform never reads headers for Nova Reel
        )
        # The transform returns the model name it was given; make sure the
        # envelope carries the resolved Bedrock model id.
        data["modelId"] = model_id

        body: Final = json.dumps(data).encode("utf-8")
        headers: Final[_ExtraHeadersDict] = {
            "Content-Type": "application/json",
            **(extra_headers or {}),
        }
        prepped: Final = self.get_request_headers(
            credentials=boto3_credentials_info.credentials,
            aws_region_name=boto3_credentials_info.aws_region_name,
            extra_headers=extra_headers,
            endpoint_url=endpoint_url,
            data=body,
            headers=headers,
            api_key=api_key,
        )

        if logging_obj is not None:
            logging_obj.pre_call(
                input=prompt,
                api_key="",
                additional_args={  # mutable-ok: logging payload dict built for this call
                    "complete_input_dict": data,
                    "api_base": endpoint_url,
                    # Redacted copy: pre_call forwards additional_args unmasked to
                    # logger_fn / log_pre_api_call callbacks; the signed headers
                    # must not leave the request path (prepped keeps them).
                    "headers": redact_bedrock_headers_for_logging(prepped.headers),
                },
            )
        return endpoint_url, prepped, body, data

    def _transform_create_response(
        self,
        model: str,
        response: httpx.Response,
        data: Mapping[str, object],
        logging_obj: LiteLLMLogging | None,
    ) -> VideoObject:
        if logging_obj is not None:
            logging_obj.post_call(
                input="",
                api_key="",
                original_response=response.text,
                additional_args={"complete_input_dict": data},  # mutable-ok: logging payload dict built for this call
            )
        # raise_for_status() already ran on both call paths, so any non-2xx is
        # covered; manual status_code checks here would only misclassify
        # legitimate 2xx variants (e.g. 202 from proxies) behind raise_for_status.
        config: Final = BedrockNovaReelVideoConfig()
        return config.transform_video_create_response(
            model=model,
            raw_response=response,
            logging_obj=logging_obj,
            request_data=data,
        )

    def video_generation(
        self,
        model: str,
        prompt: str,
        optional_params: _LitellmParamsDict,
        logging_obj: LiteLLMLogging | None,
        timeout: float | httpx.Timeout | None,
        avideo_generation: bool = False,
        client: httpx.Client | httpx.AsyncClient | None = None,
        api_base: str | None = None,
        extra_headers: _ExtraHeadersDict | None = None,
        api_key: str | None = None,
        litellm_params: GenericLiteLLMParams | Mapping[str, object] | None = None,
    ) -> VideoObject | Coroutine[object, object, VideoObject]:
        """Returns a VideoObject, or a coroutine resolving to one when avideo_generation is set."""
        if avideo_generation:
            return self.async_video_generation(
                model=model,
                prompt=prompt,
                optional_params=optional_params,
                logging_obj=logging_obj,
                timeout=timeout,
                client=client,
                api_base=api_base,
                extra_headers=extra_headers,
                api_key=api_key,
                litellm_params=litellm_params,
            )

        endpoint_url, prepped, body, data = self._prepare_async_invoke_request(
            model=model,
            prompt=prompt,
            optional_params=optional_params,
            api_base=api_base,
            extra_headers=extra_headers,
            logging_obj=logging_obj,
            api_key=api_key,
            litellm_params=litellm_params,
        )
        from litellm.llms.custom_httpx.http_handler import _get_httpx_client

        sync_client: Final = client if isinstance(client, httpx.Client) else _get_httpx_client()
        try:
            response: Final = sync_client.post(
                url=endpoint_url,
                headers=prepped.headers,
                content=body,
                timeout=timeout,
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as err:
            raise BedrockError(
                status_code=err.response.status_code,
                message=err.response.text,
                headers=err.response.headers,
                response=err.response,
            )
        except httpx.TimeoutException:
            raise BedrockError(status_code=408, message="Timeout error occurred.")
        return self._transform_create_response(model, response, data, logging_obj)

    async def async_video_generation(
        self,
        model: str,
        prompt: str,
        optional_params: _LitellmParamsDict,
        logging_obj: LiteLLMLogging | None,
        timeout: float | httpx.Timeout | None,
        client: httpx.Client | httpx.AsyncClient | None = None,
        api_base: str | None = None,
        extra_headers: _ExtraHeadersDict | None = None,
        api_key: str | None = None,
        litellm_params: GenericLiteLLMParams | Mapping[str, object] | None = None,
    ) -> VideoObject:
        from litellm.llms.custom_httpx.http_handler import get_async_httpx_client

        endpoint_url, prepped, body, data = self._prepare_async_invoke_request(
            model=model,
            prompt=prompt,
            optional_params=optional_params,
            api_base=api_base,
            extra_headers=extra_headers,
            logging_obj=logging_obj,
            api_key=api_key,
            litellm_params=litellm_params,
        )
        async_client: Final = (
            client
            if isinstance(client, httpx.AsyncClient)
            else get_async_httpx_client(
                llm_provider=litellm.LlmProviders.BEDROCK,
                params={"timeout": timeout},  # mutable-ok: per-call timeout kwargs for the shared client factory
            )
        )
        try:
            response: Final = await async_client.post(
                url=endpoint_url,
                headers=prepped.headers,
                content=body,
                timeout=timeout,
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as err:
            raise BedrockError(
                status_code=err.response.status_code,
                message=err.response.text,
                headers=err.response.headers,
                response=err.response,
            )
        except httpx.TimeoutException:
            raise BedrockError(status_code=408, message="Timeout error occurred.")
        return self._transform_create_response(model, response, data, logging_obj)

    def _status_request_parts(
        self,
        invocation_arn: str,
        optional_params: dict,  # mutable-ok: aws_* keys are popped in place
        api_base: str | None,
        api_key: str | None = None,
    ) -> tuple[str, AWSPreparedRequest, str]:
        """Returns (status_url, prepped_get_request, resolved_region) for GET /async-invoke/{arn}.

        The resolved region (explicit aws_region_name > ARN region > env > default) is
        returned so callers that follow up with an S3 download reuse the same region
        instead of re-resolving to the env/default one.
        """
        bearer_token: Final = bedrock_bearer_token(api_key)
        aws_region_name: str | None = optional_params.pop("aws_region_name", None)
        if aws_region_name is None:
            aws_region_name = _region_from_invocation_arn(invocation_arn)
        credentials, aws_region_name = self._load_credentials(
            optional_params,
            aws_region_name=aws_region_name,
            bearer_token=bearer_token,
        )
        _, proxy_endpoint_url = self.get_runtime_endpoint(
            api_base=api_base,
            aws_bedrock_runtime_endpoint=optional_params.pop("aws_bedrock_runtime_endpoint", None),
            aws_region_name=aws_region_name,
        )
        encoded_arn: Final = quote(invocation_arn, safe="")
        status_url: Final = f"{proxy_endpoint_url.rstrip('/')}/async-invoke/{encoded_arn}"
        prepped: Final = _sign_get_request(
            credentials=credentials,
            url=status_url,
            headers={"Content-Type": "application/json"},  # mutable-ok: SigV4 signs a plain headers dict
            aws_region_name=aws_region_name,
            bearer_token=bearer_token,
        )
        return status_url, prepped, aws_region_name

    def _decode_status_context(self, video_id: str) -> tuple[str, str]:
        """Returns (invocation_arn, model) encoded in the video id (single decode)."""
        decoded: Final = decode_video_id_with_provider(video_id)
        invocation_arn: Final[str] = decoded.get("video_id") or extract_original_video_id(video_id)
        if not invocation_arn:
            raise BedrockError(
                status_code=400,
                message=f"Could not extract a Bedrock invocation ARN from video id: {video_id!r}",
            )
        model: Final[str] = decoded.get("model_id") or "amazon.nova-reel-v1:0"
        return invocation_arn, model

    def _sync_get(self, prepped: AWSPreparedRequest, timeout: float | httpx.Timeout | None = None) -> httpx.Response:
        from litellm.llms.custom_httpx.http_handler import _get_httpx_client

        client: Final = _get_httpx_client()
        try:
            return client.get(url=prepped.url, headers=prepped.headers, timeout=timeout)
        except httpx.TimeoutException:
            raise BedrockError(status_code=408, message="Timeout error occurred.")

    async def _async_get(
        self,
        prepped: AWSPreparedRequest,
        timeout: float | httpx.Timeout | None = None,
    ) -> httpx.Response:
        from litellm.llms.custom_httpx.http_handler import get_async_httpx_client

        client: Final = get_async_httpx_client(
            llm_provider=litellm.LlmProviders.BEDROCK,
            params={"timeout": timeout},  # mutable-ok: per-call timeout kwargs for the shared client factory
        )
        try:
            return await client.get(url=prepped.url, headers=prepped.headers, timeout=timeout)
        except httpx.TimeoutException:
            raise BedrockError(status_code=408, message="Timeout error occurred.")

    def _map_status_response(
        self,
        response: httpx.Response,
        model: str,
        video_id: str,
        logging_obj: LiteLLMLogging | None,
    ) -> tuple[VideoObject, BedrockGetAsyncInvokeResponse]:
        if response.status_code != 200:
            raise BedrockError(
                status_code=response.status_code,
                message=f"Nova Reel get-async-invoke error: {response.text}",
                headers=response.headers,
                response=response,
            )
        raw: Final[BedrockGetAsyncInvokeResponse] = response.json()
        config: Final = BedrockNovaReelVideoConfig()
        video_obj = config.transform_video_status_retrieve_response(
            raw_response=response,
            logging_obj=logging_obj,
            model=model,
            video_id=video_id,
        )
        return video_obj, raw

    def video_status(
        self,
        video_id: str,
        litellm_params: GenericLiteLLMParams | Mapping[str, object] | None = None,
        logging_obj: LiteLLMLogging | None = None,
        api_base: str | None = None,
        api_key: str | None = None,
        astatus: bool = False,
        timeout: float | httpx.Timeout | None = None,
    ) -> VideoObject | Coroutine[object, object, VideoObject]:
        """Returns a VideoObject, or a coroutine resolving to one when astatus is set."""
        invocation_arn, model = self._decode_status_context(video_id)
        optional_params: Final[_LitellmParamsDict] = _params_to_dict(litellm_params)
        _, prepped, _ = self._status_request_parts(invocation_arn, optional_params, api_base, api_key=api_key)
        if astatus:
            return self._async_video_status(
                prepped=prepped, model=model, video_id=video_id, logging_obj=logging_obj, timeout=timeout
            )
        response: Final = self._sync_get(prepped, timeout=timeout)
        video_obj, _ = self._map_status_response(response, model, video_id, logging_obj)
        return video_obj

    async def _async_video_status(
        self,
        prepped: AWSPreparedRequest,
        model: str,
        video_id: str,
        logging_obj: LiteLLMLogging | None = None,
        timeout: float | httpx.Timeout | None = None,
    ) -> VideoObject:
        """Private async arm of video_status; only its internal dispatch calls it."""
        response: Final = await self._async_get(prepped, timeout=timeout)
        video_obj, _ = self._map_status_response(response, model, video_id, logging_obj)
        return video_obj

    def video_content(
        self,
        video_id: str,
        litellm_params: GenericLiteLLMParams | Mapping[str, object] | None = None,
        logging_obj: LiteLLMLogging | None = None,
        api_base: str | None = None,
        api_key: str | None = None,
        timeout: float | httpx.Timeout | None = None,
    ) -> bytes:
        """Download output.mp4 from the S3 output location once the job completed."""
        invocation_arn, model = self._decode_status_context(video_id)
        optional_params: Final[_LitellmParamsDict] = _params_to_dict(litellm_params)
        _, prepped, status_region = self._status_request_parts(
            invocation_arn, optional_params, api_base, api_key=api_key
        )
        response: Final = self._sync_get(prepped, timeout=timeout)
        video_obj, raw = self._map_status_response(response, model, video_id, logging_obj)
        if video_obj.status == "failed":
            failure_message: Final[str] = (
                (video_obj.error.get("message") if video_obj.error else None) or raw.get("failureMessage") or ""
            )
            raise BedrockError(
                status_code=502,
                message=f"Nova Reel invocation failed: {failure_message}",
            )
        if video_obj.status != "completed":
            # Client error about job state: 400-class, not a 500 that SDKs retry.
            raise BedrockError(
                status_code=400,
                message=(
                    "Nova Reel video generation is not complete yet "
                    f"(status={video_obj.status}). Check video_status() before downloading."
                ),
            )

        s3_uri: Final[str | None] = _s3_uri_from_output_config(raw.get("outputDataConfig"))
        if not s3_uri:
            # The raw get-async-invoke payload carries invocationArn (with the
            # account id); debug-log it, raise with only the observed key names.
            verbose_logger.debug("Nova Reel completed invocation without an S3 output location: %r", raw)
            output_config: Final = raw.get("outputDataConfig")
            observed_keys: Final = tuple(sorted(output_config.keys())) if isinstance(output_config, Mapping) else ()
            raise ValueError(
                f"No S3 output location on completed invocation (observed outputDataConfig keys: {observed_keys})"
            )

        bucket, prefix = _parse_s3_uri(s3_uri)
        # Nova writes output.mp4 into a per-invocation folder (v1:1 docs); older
        # v1:0 flows placed it directly under the configured prefix.
        key_candidates: Final = [  # mutable-ok: candidate S3 keys are tried in order
            f"{prefix}/{invocation_arn.rsplit('/', 1)[-1]}/{NOVA_REEL_OUTPUT_FILENAME}".lstrip("/"),
            f"{prefix}/{NOVA_REEL_OUTPUT_FILENAME}".lstrip("/"),
        ]
        return self._download_s3_object(
            bucket,
            key_candidates,
            litellm_params,
            raw,
            region_default=status_region,
            api_key=api_key,
            timeout=timeout,
        )

    def _download_s3_object(
        self,
        bucket: str,
        key_candidates: Sequence[str],
        litellm_params: GenericLiteLLMParams | Mapping[str, object] | None,
        raw: BedrockGetAsyncInvokeResponse,
        region_default: str | None = None,
        api_key: str | None = None,
        timeout: float | httpx.Timeout | None = None,
    ) -> bytes:
        """Download the output object from S3.

        region_default (the region the status request resolved: explicit
        aws_region_name > ARN region > env > default) is used unless the fresh
        litellm_params carry an explicit aws_region_name, which still wins.
        timeout (the video_content call timeout) bounds the boto3 client's
        read timeout; connect timeout is it or DEFAULT_S3_CONNECT_TIMEOUT_S,
        whichever is smaller.
        """
        try:
            import boto3
            from botocore.config import Config as BotocoreConfig
            from botocore.exceptions import BotoCoreError, ClientError
        except ImportError:
            raise ImportError("Missing boto3 to download Nova Reel output. Run 'pip install boto3'.")

        optional_params: Final[_LitellmParamsDict] = _params_to_dict(litellm_params)
        explicit_region: Final[str | None] = optional_params.pop("aws_region_name", None)
        bearer_token: Final[str | None] = bedrock_bearer_token(api_key)
        # Always resolve SigV4 credentials for S3: a Bedrock bearer token only
        # covers the async-invoke API, never S3 object access.
        credentials, region = self._load_credentials(
            optional_params,
            aws_region_name=(explicit_region if explicit_region is not None else region_default),
        )
        if bearer_token is not None and credentials is None:
            raise BedrockError(
                status_code=400,
                message=(
                    "Nova Reel video content download requires AWS SigV4 credentials with S3 read "
                    "access (aws_access_key_id/aws_secret_access_key or an ambient credential chain). "
                    "Bedrock bearer tokens only cover the Bedrock asynchronous invoke API and cannot "
                    "download objects from S3."
                ),
            )
        session_kwargs: Final[dict[str, str]] = {"region_name": region}  # mutable-ok: credential keys are added below
        if credentials is not None:
            session_kwargs["aws_access_key_id"] = credentials.access_key
            session_kwargs["aws_secret_access_key"] = credentials.secret_key
            if credentials.token:
                session_kwargs["aws_session_token"] = credentials.token
        read_timeout: Final[float] = float(timeout) if isinstance(timeout, (int, float)) else DEFAULT_S3_READ_TIMEOUT_S
        connect_timeout: Final[float] = min(DEFAULT_S3_CONNECT_TIMEOUT_S, read_timeout)
        client_config: Final = BotocoreConfig(connect_timeout=connect_timeout, read_timeout=read_timeout)
        session: Final = boto3.Session(**session_kwargs)
        s3_client: Final = session.client("s3", config=client_config)

        s3_uri: Final[str | None] = _s3_uri_from_output_config(raw.get("outputDataConfig"))
        errors: list[str] = []  # mutable-ok: error strings accumulate across candidate keys
        try:
            for key in key_candidates:
                body = None  # per-iteration resource; closed in finally
                try:
                    obj = s3_client.get_object(Bucket=bucket, Key=key)
                    body = obj["Body"]
                    return body.read()
                except ClientError as err:
                    # Annotated, not Final: basedpyright forbids Final assignment inside loops.
                    error_code: str = _client_error_code(err)
                    if error_code == "NoSuchKey" or _client_error_http_status(err) == 404:
                        # Only a missing key falls through to the next candidate.
                        errors.append(str(err))
                        continue
                    if error_code == "AccessDenied":
                        raise BedrockError(
                            status_code=403,
                            message=(f"Access denied downloading Nova Reel output from {s3_uri} (key {key!r}): {err}"),
                        ) from err
                    raise BedrockError(
                        status_code=502,
                        message=(
                            f"AWS error (code {error_code!r}) downloading Nova Reel output "
                            f"from {s3_uri} (key {key!r}): {err}"
                        ),
                    ) from err
                finally:
                    if body is not None:
                        body.close()
        except BotoCoreError as err:
            # ClientError never reaches this handler: it inherits Exception (not
            # BotoCoreError) and is handled per-key above. Anything else that is
            # a BotoCoreError (NoCredentialsError, EndpointConnectionError, ...)
            # aborts the download and maps to a 502. botocore messages carry
            # class name + failure reason, never credentials.
            raise BedrockError(
                status_code=502,
                message=f"Failed to download Nova Reel output from S3: {type(err).__name__}: {err}",
            )
        finally:
            s3_client.close()
        raise BedrockError(
            status_code=404,
            message=(
                "Nova Reel output video not found in the S3 output location "
                f"{s3_uri}. Tried keys: {tuple(key_candidates)}. Errors: {tuple(errors)}"
            ),
            headers={},  # mutable-ok: synthesized 404 carries no provider headers
        )
