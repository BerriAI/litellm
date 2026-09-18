"""
This file contains the handler for AWS Bedrock Nova Sonic realtime API.

This uses aws_sdk_bedrock_runtime for bidirectional streaming with Nova Sonic.
"""

import asyncio
import contextlib
import importlib.metadata
import json
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, MutableMapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, NoReturn, Protocol, runtime_checkable

from pydantic import JsonValue, TypeAdapter

import litellm
from litellm._logging import _redact_string, verbose_proxy_logger
from litellm.constants import (
    BEDROCK_REALTIME_COMMITTED_FAILURE_SCOPE_KEY,
    BEDROCK_REALTIME_PENDING_SESSION_UPDATE_SCOPE_KEY,
    BEDROCK_REALTIME_SDK_DISTRIBUTION,
    BEDROCK_REALTIME_SDK_SUPPORTED_RANGE,
    BEDROCK_REALTIME_SESSION_COMMITTED_SCOPE_KEY,
    REALTIME_SESSION_SUCCESS_LOGGED_KEY,
)
from litellm.litellm_core_utils.aws_partition import get_aws_dns_suffix
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLogging
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
from litellm.litellm_core_utils.realtime_streaming import DefaultLoggedRealTimeEventTypes
from litellm.types.llms.bedrock import AwsAuthParams
from litellm.types.llms.openai import OpenAIRealtimeEvents
from litellm.types.realtime import RealtimeResponseTransformInput

from ..base_aws_llm import BaseAWSLLM, run_aws_signing
from ..common_utils import BedrockError
from .transformation import BedrockRealtimeConfig

_CLIENT_MODALITIES_ADAPTER: Final[TypeAdapter["list[str] | None"]] = TypeAdapter(list[str] | None)
_CLIENT_MESSAGE_ADAPTER: Final[TypeAdapter[JsonValue]] = TypeAdapter(JsonValue)

_EMPTY_JSON_OBJECT: Final[Mapping[str, JsonValue]] = MappingProxyType({})

_BEDROCK_STREAM_ERROR_STATUS: Final[Mapping[str, int]] = MappingProxyType(
    {
        "AccessDeniedException": 403,
        "ConflictException": 400,
        "InternalServerException": 500,
        "ModelErrorException": 424,
        "ModelNotReadyException": 429,
        "ModelStreamErrorException": 424,
        "ModelTimeoutException": 408,
        "ResourceNotFoundException": 404,
        "ServiceQuotaExceededException": 400,
        "ServiceUnavailableException": 503,
        "ThrottlingException": 429,
        "ValidationException": 400,
    }
)


def _as_bedrock_error(error: BaseException) -> BaseException:
    status_code: Final = _BEDROCK_STREAM_ERROR_STATUS.get(type(error).__name__)
    if status_code is None:
        return error
    return BedrockError(status_code=status_code, message=f"{type(error).__name__}: {error}")


def _json_dict(value: JsonValue) -> dict[str, JsonValue]:
    return value if isinstance(value, dict) else {}


def _json_str(value: JsonValue) -> str | None:
    return value if isinstance(value, str) else None


def _should_log_event(openai_message: Mapping[str, object]) -> bool:
    logged_types: Final = (
        litellm.logged_real_time_event_types
        if litellm.logged_real_time_event_types is not None
        else DefaultLoggedRealTimeEventTypes
    )
    if logged_types == "*":
        return True
    return openai_message.get("type") in logged_types


class RealtimeClientWebSocket(Protocol):
    """The client-facing websocket surface the realtime bridge talks to."""

    scope: MutableMapping[str, object]  # mutable-ok: the ASGI scope is the per-connection state store

    async def receive_text(self) -> str: ...

    async def send_text(self, data: str) -> None: ...

    async def close(self, code: int = 1000, reason: str | None = None) -> None: ...


class BedrockInputStream(Protocol):
    async def send(self, event: object) -> None: ...

    async def close(self) -> None: ...


class BedrockPayloadPart(Protocol):
    @property
    def bytes_(self) -> bytes | None: ...


class BedrockOutputChunk(Protocol):
    @property
    def value(self) -> BedrockPayloadPart | None: ...


class BedrockOutputStream(Protocol):
    async def receive(self) -> BedrockOutputChunk | None: ...


class BedrockBidirectionalStream(Protocol):
    @property
    def input_stream(self) -> BedrockInputStream: ...

    async def await_output(self) -> tuple[object, BedrockOutputStream]: ...


@runtime_checkable
class ClosableBedrockRuntimeClient(Protocol):
    async def close(self) -> None: ...


def _installed_sdk_version() -> str | None:
    try:
        return importlib.metadata.version(BEDROCK_REALTIME_SDK_DISTRIBUTION)
    except importlib.metadata.PackageNotFoundError:
        return None


def _sdk_import_error(installed_version: str | None, cause: ImportError) -> ImportError:
    install_hint: Final = "pip install 'litellm[bedrock-realtime]'"
    requirement: Final = f"{BEDROCK_REALTIME_SDK_DISTRIBUTION}[awscrt]{BEDROCK_REALTIME_SDK_SUPPORTED_RANGE}"
    verbose_proxy_logger.error("Bedrock Realtime: SDK import failed (installed=%s): %s", installed_version, cause)
    if installed_version is None:
        return ImportError(f"Missing aws_sdk_bedrock_runtime: {install_hint} ({requirement})")
    return ImportError(
        f"{BEDROCK_REALTIME_SDK_DISTRIBUTION} {installed_version} is installed but Bedrock realtime needs "
        f"[awscrt]{BEDROCK_REALTIME_SDK_SUPPORTED_RANGE}: {install_hint}"
    )


async def _close_bedrock_client(bedrock_client: object) -> None:
    if not isinstance(bedrock_client, ClosableBedrockRuntimeClient):
        return
    with contextlib.suppress(Exception):
        await bedrock_client.close()
        verbose_proxy_logger.debug("Bedrock Realtime: closed SDK client")


@dataclass(frozen=True, slots=True)
class _BridgeOutcome:
    logged_events: tuple[OpenAIRealtimeEvents, ...]
    provider_failure: BaseException | None
    client_disconnected: bool


async def _client_messages(client_ws: RealtimeClientWebSocket, initial_message: str | None) -> AsyncIterator[str]:
    if initial_message is not None:
        yield initial_message
    while True:
        try:
            yield await client_ws.receive_text()
        except Exception as e:  # noqa: BLE001  # any receive failure means the client is gone
            verbose_proxy_logger.debug("Client to Bedrock forwarding ended: %s", e, exc_info=True)
            return


def _pending_session_update(scope: Mapping[str, object]) -> str | None:
    """A fallback attempt on the same websocket replays the session.update the failed attempt never acked."""
    if scope.get(BEDROCK_REALTIME_SESSION_COMMITTED_SCOPE_KEY) is True:
        committed_failure: Final = scope.get(BEDROCK_REALTIME_COMMITTED_FAILURE_SCOPE_KEY)
        raise BedrockError(
            status_code=400,
            message=(
                "Bedrock realtime session already committed to a provider stream; it cannot be replayed"
                + (f". The committed stream failed with: {committed_failure}" if committed_failure else "")
            ),
        )
    pending: Final = scope.get(BEDROCK_REALTIME_PENDING_SESSION_UPDATE_SCOPE_KEY)
    return pending if isinstance(pending, str) else None


def _raise_provider_failure(scope: MutableMapping[str, object], failure: BaseException) -> NoReturn:
    error: Final = _as_bedrock_error(failure)
    verbose_proxy_logger.error("Bedrock Realtime: provider stream failed: %s", _redact_string(str(error)))
    if scope.get(BEDROCK_REALTIME_SESSION_COMMITTED_SCOPE_KEY) is True:
        scope[BEDROCK_REALTIME_COMMITTED_FAILURE_SCOPE_KEY] = _redact_string(str(error))
    raise error from failure


def _parse_client_message(message: str) -> Mapping[str, JsonValue]:
    try:
        return _json_dict(_CLIENT_MESSAGE_ADAPTER.validate_json(message))
    except ValueError:
        return _EMPTY_JSON_OBJECT


async def _ack_session_update(
    client_ws: RealtimeClientWebSocket,
    bedrock_stream: BedrockBidirectionalStream,
    transformation_config: BedrockRealtimeConfig,
    model: str,
    logging_obj: LiteLLMLogging | None,
    parsed_client_message: Mapping[str, JsonValue],
) -> bool:
    """Ack the client's session.update once Bedrock accepted the stream; False means the client is gone."""
    await bedrock_stream.await_output()
    client_ws.scope.pop(BEDROCK_REALTIME_PENDING_SESSION_UPDATE_SCOPE_KEY, None)
    client_ws.scope[BEDROCK_REALTIME_SESSION_COMMITTED_SCOPE_KEY] = True  # rebind-ok: scope outlives the attempt
    if logging_obj is None:
        return True
    requested_modalities: Final = _CLIENT_MODALITIES_ADAPTER.validate_python(
        _json_dict(parsed_client_message.get("session")).get("modalities")
    )
    try:
        await client_ws.send_text(
            json.dumps(transformation_config.session_updated_event(model, logging_obj, requested_modalities))
        )
    except Exception as e:  # noqa: BLE001  # any send failure means the client is gone
        verbose_proxy_logger.debug("Client to Bedrock forwarding ended: %s", e, exc_info=True)
        return False
    return True


class BedrockRealtime(BaseAWSLLM):
    """Handler for Bedrock Nova Sonic realtime speech-to-speech API."""

    def __init__(self, sdk_version_lookup: Callable[[], str | None] = _installed_sdk_version):
        super().__init__()
        self._sdk_version_lookup: Final = sdk_version_lookup

    async def async_realtime(
        self,
        model: str,
        websocket: RealtimeClientWebSocket,
        logging_obj: LiteLLMLogging,
        api_base: str | None = None,
        api_key: str | None = None,
        timeout: float | None = None,
        aws_region_name: str | None = None,
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
        aws_session_token: str | None = None,
        aws_role_name: str | None = None,
        aws_session_name: str | None = None,
        aws_profile_name: str | None = None,
        aws_web_identity_token: str | None = None,
        aws_sts_endpoint: str | None = None,
        aws_bedrock_runtime_endpoint: str | None = None,
        aws_external_id: str | None = None,
        aws_session_tags: object = None,
        **kwargs: object,
    ):
        """
        Establish bidirectional streaming connection with Bedrock Nova Sonic.

        Args:
            model: Model ID (e.g., 'amazon.nova-sonic-v1:0')
            websocket: Client WebSocket connection
            logging_obj: LiteLLM logging object
            aws_region_name: AWS region
            Various AWS authentication parameters
        """
        try:
            from aws_sdk_bedrock_runtime.client import AsyncBedrockRuntimeClient
            from aws_sdk_bedrock_runtime.config import AsyncBedrockRuntimeConfig
            from aws_sdk_bedrock_runtime.models import InvokeModelWithBidirectionalStreamOperationInput
            from smithy_aws_core.identity import AWSCredentialsIdentity, StaticCredentialsResolver
            from smithy_http.aio.crt import AWSCRTHTTPClient
        except ImportError as e:
            raise _sdk_import_error(self._sdk_version_lookup(), e) from e

        pending_session_update: Final = _pending_session_update(websocket.scope)

        # Get AWS region
        if aws_region_name is None:
            optional_params: Final = {
                "aws_region_name": aws_region_name,
            }
            aws_region_name = self._get_aws_region_name(optional_params, model)

        # Get endpoint URL
        if api_base is not None:
            endpoint_uri = api_base
        elif aws_bedrock_runtime_endpoint is not None:
            endpoint_uri = aws_bedrock_runtime_endpoint
        else:
            endpoint_uri = f"https://bedrock-runtime.{aws_region_name}.{get_aws_dns_suffix(aws_region_name)}"

        verbose_proxy_logger.debug("Bedrock Realtime: Connecting to %s with model %s", endpoint_uri, model)

        auth_params: Final = AwsAuthParams(
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_session_token=aws_session_token,
            aws_session_name=aws_session_name,
            aws_profile_name=aws_profile_name,
            aws_role_name=aws_role_name,
            aws_web_identity_token=aws_web_identity_token,
            aws_sts_endpoint=aws_sts_endpoint,
            aws_external_id=aws_external_id,
            aws_session_tags=aws_session_tags,
        )
        credentials: Final = await run_aws_signing(self.resolve_credentials, auth_params, aws_region_name)
        if credentials is None:  # pyright: ignore[reportUnnecessaryComparison]  # boto3.Session() env fallback yields None
            raise BedrockError(
                status_code=401,
                message=(
                    "No AWS credentials found for Bedrock realtime. Set aws_* params in litellm_params "
                    "or configure credentials in the environment"
                ),
            )
        frozen_credentials: Final = await run_aws_signing(credentials.get_frozen_credentials)

        credentials_identity: Final = AWSCredentialsIdentity(
            access_key_id=frozen_credentials.access_key,
            secret_access_key=frozen_credentials.secret_key,
            session_token=frozen_credentials.token,
        )
        config: Final = await AsyncBedrockRuntimeConfig.resolve(
            endpoint_uri=endpoint_uri,
            region=aws_region_name,
            aws_credentials_identity_resolver=StaticCredentialsResolver(identity=credentials_identity),
            transport=AWSCRTHTTPClient(),
        )
        bedrock_client: Final = AsyncBedrockRuntimeClient(config=config)

        async def open_bidirectional_stream() -> BedrockBidirectionalStream:
            return await bedrock_client.invoke_model_with_bidirectional_stream(
                InvokeModelWithBidirectionalStreamOperationInput(model_id=model)
            )

        try:
            await self._run_session(websocket, open_bidirectional_stream, model, logging_obj, pending_session_update)
        finally:
            await _close_bedrock_client(bedrock_client)

    async def _run_session(
        self,
        websocket: RealtimeClientWebSocket,
        open_bidirectional_stream: Callable[[], Awaitable[BedrockBidirectionalStream]],
        model: str,
        logging_obj: LiteLLMLogging,
        pending_session_update: str | None,
    ) -> None:
        transformation_config: Final = BedrockRealtimeConfig()

        bedrock_stream: Final = await open_bidirectional_stream()

        verbose_proxy_logger.debug("Bedrock Realtime: Bidirectional stream established")

        if pending_session_update is None:
            await websocket.send_text(json.dumps(transformation_config.session_created_event(model, logging_obj)))
            verbose_proxy_logger.debug("Bedrock Realtime: sent session.created to client on connect")

        # Track state for transformation
        session_state: Final[RealtimeResponseTransformInput] = {
            "current_output_item_id": None,
            "current_response_id": None,
            "current_conversation_id": None,
            "current_delta_chunks": None,
            "current_item_chunks": None,
            "current_delta_type": None,
            "session_configuration_request": None,
        }

        outcome: Final = await self._bridge(
            websocket,
            bedrock_stream,
            transformation_config,
            model,
            session_state,
            logging_obj,
            initial_message=pending_session_update,
        )

        logged_events: Final = (
            *outcome.logged_events,
            *(
                leftover_event
                for leftover_event in transformation_config.leftover_usage_done_events()
                if _should_log_event(leftover_event)
            ),
        )
        if logged_events:
            GLOBAL_LOGGING_WORKER.ensure_initialized_and_enqueue(
                logging_obj.dispatch_success_handlers(
                    list(logged_events),  # mutable-ok: realtime spend logging requires a list result
                    prefer_async_handlers=True,
                )
            )
            logging_obj.model_call_details[REALTIME_SESSION_SUCCESS_LOGGED_KEY] = True

        if outcome.provider_failure is None:
            return
        if outcome.client_disconnected:
            verbose_proxy_logger.debug(
                "Bedrock Realtime: stream failed after the client disconnected: %s", outcome.provider_failure
            )
            return
        _raise_provider_failure(websocket.scope, outcome.provider_failure)

    async def _bridge(
        self,
        websocket: RealtimeClientWebSocket,
        bedrock_stream: BedrockBidirectionalStream,
        transformation_config: BedrockRealtimeConfig,
        model: str,
        session_state: RealtimeResponseTransformInput,
        logging_obj: LiteLLMLogging,
        initial_message: str | None,
    ) -> _BridgeOutcome:
        """Run both forwarding directions until the client leaves or either side fails."""
        logged: Final[list[OpenAIRealtimeEvents]] = []  # mutable-ok: events forwarded before a failure are still spend

        async def collect_logged_events() -> None:
            async for event in self._forward_bedrock_to_client(
                bedrock_stream, websocket, transformation_config, model, logging_obj, session_state
            ):
                logged.append(event)

        client_task: Final = asyncio.create_task(
            self._forward_client_to_bedrock(
                websocket, bedrock_stream, transformation_config, model, session_state, logging_obj, initial_message
            )
        )
        bedrock_task: Final = asyncio.create_task(collect_logged_events())

        await asyncio.wait((client_task, bedrock_task), return_when=asyncio.FIRST_COMPLETED)
        client_disconnected: Final = (
            client_task.done() and not client_task.cancelled() and client_task.exception() is None
        )
        client_task.cancel()
        bedrock_task.cancel()
        client_outcome, bedrock_outcome = await asyncio.gather(client_task, bedrock_task, return_exceptions=True)

        return _BridgeOutcome(
            logged_events=tuple(logged),
            provider_failure=(
                client_outcome
                if isinstance(client_outcome, Exception)
                else bedrock_outcome
                if isinstance(bedrock_outcome, Exception)
                else None
            ),
            client_disconnected=client_disconnected,
        )

    async def _forward_client_to_bedrock(
        self,
        client_ws: RealtimeClientWebSocket,
        bedrock_stream: BedrockBidirectionalStream,
        transformation_config: BedrockRealtimeConfig,
        model: str,
        session_state: RealtimeResponseTransformInput,
        logging_obj: LiteLLMLogging | None = None,
        initial_message: str | None = None,
    ) -> None:
        """Forward messages from client WebSocket to Bedrock stream.

        Returns once the client is gone; provider failures (input stream or readiness) propagate to the caller.
        """
        from aws_sdk_bedrock_runtime.models import (
            BidirectionalInputPayloadPart,
            InvokeModelWithBidirectionalStreamInputChunk,
        )

        def build_input_chunk(payload: bytes) -> object:
            return InvokeModelWithBidirectionalStreamInputChunk(value=BidirectionalInputPayloadPart(bytes_=payload))

        async def send_to_bedrock(bedrock_message: str) -> None:
            event: Final = build_input_chunk(bedrock_message.encode("utf-8"))
            await bedrock_stream.input_stream.send(event)
            verbose_proxy_logger.debug("Bedrock Realtime: Sent to Bedrock: %s", bedrock_message[:200])

        try:
            async for message in _client_messages(client_ws, initial_message):
                verbose_proxy_logger.debug("Bedrock Realtime: Received from client: %s", message[:200])
                parsed_client_message = _parse_client_message(message)
                is_session_update = _json_str(parsed_client_message.get("type")) == "session.update"
                if is_session_update:
                    client_ws.scope[BEDROCK_REALTIME_PENDING_SESSION_UPDATE_SCOPE_KEY] = (
                        message  # rebind-ok: scope outlives the attempt
                    )

                transformed_messages = transformation_config.transform_realtime_request(
                    message=message,
                    model=model,
                    session_configuration_request=session_state.get("session_configuration_request"),
                )
                for bedrock_message in transformed_messages:
                    await send_to_bedrock(bedrock_message)

                if is_session_update and not await _ack_session_update(
                    client_ws, bedrock_stream, transformation_config, model, logging_obj, parsed_client_message
                ):
                    break
        finally:
            for close_message in transformation_config.session_close_messages():
                with contextlib.suppress(Exception):
                    await send_to_bedrock(close_message)
            with contextlib.suppress(Exception):
                await bedrock_stream.input_stream.close()

    async def _forward_bedrock_to_client(
        self,
        bedrock_stream: BedrockBidirectionalStream,
        client_ws: RealtimeClientWebSocket,
        transformation_config: BedrockRealtimeConfig,
        model: str,
        logging_obj: LiteLLMLogging,
        session_state: RealtimeResponseTransformInput,
    ) -> AsyncIterator[OpenAIRealtimeEvents]:
        """Forward messages from Bedrock to the client, yielding the ones to record for spend logging.

        Provider failures propagate to the caller; the client websocket is only closed on a normal stream end.
        """

        async def send_to_client(message_json: str) -> bool:
            try:
                await client_ws.send_text(message_json)
            except Exception as e:  # noqa: BLE001  # any send failure means the client is gone
                verbose_proxy_logger.debug("Bedrock to client forwarding ended: %s", e, exc_info=True)
                return False
            verbose_proxy_logger.debug("Bedrock Realtime: Sent to client: %s", message_json[:200])
            return True

        output: Final = await bedrock_stream.await_output()
        while True:
            result = await output[1].receive()

            if result is None:
                verbose_proxy_logger.debug("Bedrock Realtime: Bedrock stream ended")
                with contextlib.suppress(Exception):
                    await client_ws.close()
                return

            payload_bytes = result.value.bytes_ if result.value else None
            if payload_bytes:
                bedrock_response = payload_bytes.decode("utf-8")
                verbose_proxy_logger.debug("Bedrock Realtime: Received from Bedrock: %s", bedrock_response[:200])

                # Transform Bedrock format to OpenAI format
                realtime_response_transform_input: RealtimeResponseTransformInput = {
                    "current_output_item_id": session_state.get("current_output_item_id"),
                    "current_response_id": session_state.get("current_response_id"),
                    "current_conversation_id": session_state.get("current_conversation_id"),
                    "current_delta_chunks": session_state.get("current_delta_chunks"),
                    "current_item_chunks": session_state.get("current_item_chunks"),
                    "current_delta_type": session_state.get("current_delta_type"),
                    "session_configuration_request": session_state.get("session_configuration_request"),
                }

                transformed_response = transformation_config.transform_realtime_response(
                    message=bedrock_response,
                    model=model,
                    logging_obj=logging_obj,
                    realtime_response_transform_input=realtime_response_transform_input,
                )

                # Update session state
                session_state.update(
                    {
                        "current_output_item_id": transformed_response.get("current_output_item_id"),
                        "current_response_id": transformed_response.get("current_response_id"),
                        "current_conversation_id": transformed_response.get("current_conversation_id"),
                        "current_delta_chunks": transformed_response.get("current_delta_chunks"),
                        "current_item_chunks": transformed_response.get("current_item_chunks"),
                        "current_delta_type": transformed_response.get("current_delta_type"),
                        "session_configuration_request": transformed_response.get("session_configuration_request"),
                    }
                )

                # Send transformed messages to client
                response_value = transformed_response["response"]
                openai_messages = response_value if isinstance(response_value, list) else (response_value,)
                for openai_message in openai_messages:
                    if not await send_to_client(json.dumps(openai_message)):
                        return
                    if _should_log_event(openai_message):
                        yield openai_message
