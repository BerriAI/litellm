# pyright: reportMissingModuleSource=false, reportUnknownMemberType=false
from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import grpc

from litellm.python_extension.generated.v1 import extension_host_pb2_grpc as pb_grpc
from litellm.types.guardrails import GuardrailEventHooks, Mode

from .adapters import RemoteCustomGuardrail, RemoteCustomLogger
from .cache_gateway import GatewayServices, InvocationCacheRegistry
from .client import PythonExtensionClient
from .config import ExtensionHostSettings, settings_from_config
from .manifest import ExtensionManifest, build_manifest


@dataclass(slots=True)
class ExtensionRuntime:
    settings: ExtensionHostSettings
    manifest: ExtensionManifest
    client: PythonExtensionClient
    cache_registry: InvocationCacheRegistry
    gateway_server: grpc.aio.Server | None = None

    def callback(self, entrypoint: str) -> RemoteCustomLogger | None:
        plugin_id = self.manifest.callback_ids.get(entrypoint)  # rebind-ok: invocation-scoped RPC state
        if plugin_id is None:
            return None
        return RemoteCustomLogger(self.client, plugin_id, self.cache_registry)

    def guardrail(
        self,
        entrypoint: str,
        name: str,
        event_hook: GuardrailEventHooks  # mutable-ok: LiteLLM compatibility payload
        | list[GuardrailEventHooks]
        | Mode
        | None = None,  # mutable-ok: LiteLLM compatibility payload
        default_on: bool = False,
        **kwargs: object,  # kwargs-ok: LiteLLM callback compatibility
    ) -> RemoteCustomGuardrail | None:
        plugin_id = self.manifest.guardrail_ids.get((entrypoint, name))  # rebind-ok: invocation-scoped RPC state
        if plugin_id is None:
            return None
        return RemoteCustomGuardrail(
            self.client,
            plugin_id,
            self.cache_registry,
            guardrail_name=name,
            event_hook=event_hook,
            default_on=default_on,
            extra_params=kwargs,
        )

    async def close(self) -> None:
        await self.client.close()
        if self.gateway_server is not None:
            await self.gateway_server.stop(grace=5)


_runtime: ExtensionRuntime | None = None  # rebind-ok: invocation-scoped RPC state


def get_extension_runtime() -> ExtensionRuntime | None:
    return _runtime


async def configure_extension_runtime(
    config: dict[str, object],  # mutable-ok: LiteLLM compatibility payload
) -> ExtensionRuntime | None:
    global _runtime  # noqa: PLW0603  # process-wide extension lifecycle singleton
    settings: Final = settings_from_config(config)
    if settings is None:
        if _runtime is not None:
            await _runtime.close()
            _runtime = None  # rebind-ok: invocation-scoped RPC state
        return None
    manifest: Final = build_manifest(config)
    cache_registry: Final = InvocationCacheRegistry()
    gateway_server: Final = await _start_gateway_services(settings, cache_registry)
    client: Final = PythonExtensionClient(settings, manifest)
    try:
        await client.start()
    except Exception:
        await client.close()
        if gateway_server is not None:
            await gateway_server.stop(grace=0)
        raise
    previous: Final = _runtime
    _runtime = ExtensionRuntime(  # rebind-ok: invocation-scoped RPC state
        settings, manifest, client, cache_registry, gateway_server
    )  # rebind-ok: invocation-scoped RPC state
    if previous is not None:
        if previous.manifest.revision_id != manifest.revision_id:
            await previous.client.retire(previous.manifest.revision_id)
        await previous.close()
    return _runtime


def remote_callback(entrypoint: str) -> RemoteCustomLogger | None:
    return _runtime.callback(entrypoint) if _runtime is not None else None


def remote_guardrail(
    entrypoint: str,
    name: str,
    event_hook: GuardrailEventHooks  # mutable-ok: LiteLLM compatibility payload
    | list[GuardrailEventHooks]
    | Mode
    | None = None,  # mutable-ok: LiteLLM compatibility payload
    default_on: bool = False,
    **kwargs: object,  # kwargs-ok: LiteLLM callback compatibility
) -> RemoteCustomGuardrail | None:
    if _runtime is None:
        return None
    return _runtime.guardrail(
        entrypoint,
        name,
        event_hook=event_hook,
        default_on=default_on,
        **kwargs,
    )


async def _start_gateway_services(
    settings: ExtensionHostSettings, registry: InvocationCacheRegistry
) -> grpc.aio.Server | None:
    if settings.gateway_listen is None:
        return None
    server: Final = grpc.aio.server()
    pb_grpc.add_GatewayServicesServicer_to_server(GatewayServices(settings.token, registry), server)
    target: Final = settings.gateway_listen.removeprefix("http://").removeprefix("https://")
    if server.add_insecure_port(target) == 0:
        raise RuntimeError(f"failed to bind GatewayServices to {settings.gateway_listen}")
    await server.start()
    return server
