# -*- coding: utf-8 -*-
# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import os
import dataclasses
import types
from typing import Any, cast
from collections.abc import Sequence

import google.ai.generativelanguage as glm

from google.auth import credentials as ga_credentials
from google.api_core import client_options as client_options_lib
from google.api_core import gapic_v1
from google.api_core import operations_v1

from google.generativeai import version

USER_AGENT = "genai-py"


@dataclasses.dataclass
class _ClientManager:
    client_config: dict[str, Any] = dataclasses.field(default_factory=dict)
    default_metadata: Sequence[tuple[str, str]] = ()
    discuss_client: glm.DiscussServiceClient | None = None
    discuss_async_client: glm.DiscussServiceAsyncClient | None = None
    model_client: glm.ModelServiceClient | None = None
    text_client: glm.TextServiceClient | None = None
    operations_client = None

    def configure(
        self,
        *,
        api_key: str | None = None,
        credentials: ga_credentials.Credentials | dict | None = None,
        # The user can pass a string to choose `rest` or `grpc` or 'grpc_asyncio'.
        # See `_transport_registry` in `DiscussServiceClientMeta`.
        # Since the transport classes align with the client classes it wouldn't make
        # sense to accept a `Transport` object here even though the client classes can.
        # We could accept a dict since all the `Transport` classes take the same args,
        # but that seems rare. Users that need it can just switch to the low level API.
        transport: str | None = None,
        client_options: client_options_lib.ClientOptions | dict | None = None,
        client_info: gapic_v1.client_info.ClientInfo | None = None,
        default_metadata: Sequence[tuple[str, str]] = (),
    ) -> None:
        """Captures default client configuration.

        If no API key has been provided (either directly, or on `client_options`) and the
        `GOOGLE_API_KEY` environment variable is set, it will be used as the API key.

        Note: Not all arguments are detailed below. Refer to the `*ServiceClient` classes in
        `google.ai.generativelanguage` for details on the other arguments.

        Args:
            transport: A string, one of: [`rest`, `grpc`, `grpc_asyncio`].
            api_key: The API-Key to use when creating the default clients (each service uses
                a separate client). This is a shortcut for `client_options={"api_key": api_key}`.
                If omitted, and the `GOOGLE_API_KEY` environment variable is set, it will be
                used.
            default_metadata: Default (key, value) metadata pairs to send with every request.
                when using `transport="rest"` these are sent as HTTP headers.
        """
        if isinstance(client_options, dict):
            client_options = client_options_lib.from_dict(client_options)
        if client_options is None:
            client_options = client_options_lib.ClientOptions()
        client_options = cast(client_options_lib.ClientOptions, client_options)
        had_api_key_value = getattr(client_options, "api_key", None)

        if had_api_key_value:
            if api_key is not None:
                raise ValueError("You can't set both `api_key` and `client_options['api_key']`.")
        else:
            if api_key is None:
                # If no key is provided explicitly, attempt to load one from the
                # environment.
                api_key = os.getenv("GOOGLE_API_KEY")

            client_options.api_key = api_key

        user_agent = f"{USER_AGENT}/{version.__version__}"
        if client_info:
            # Be respectful of any existing agent setting.
            if client_info.user_agent:
                client_info.user_agent += f" {user_agent}"
            else:
                client_info.user_agent = user_agent
        else:
            client_info = gapic_v1.client_info.ClientInfo(user_agent=user_agent)

        client_config = {
            "credentials": credentials,
            "transport": transport,
            "client_options": client_options,
            "client_info": client_info,
        }

        client_config = {key: value for key, value in client_config.items() if value is not None}

        self.client_config = client_config
        self.default_metadata = default_metadata
        self.discuss_client = None
        self.text_client = None
        self.model_client = None
        self.operations_client = None

    def make_client(self, cls):
        # Attempt to configure using defaults.
        if not self.client_config:
            configure()

        client = cls(**self.client_config)

        if not self.default_metadata:
            return client

        def keep(name, f):
            if name.startswith("_"):
                return False
            elif not isinstance(f, types.FunctionType):
                return False
            elif isinstance(f, classmethod):
                return False
            elif isinstance(f, staticmethod):
                return False
            else:
                return True

        def add_default_metadata_wrapper(f):
            def call(*args, metadata=(), **kwargs):
                metadata = list(metadata) + list(self.default_metadata)
                return f(*args, **kwargs, metadata=metadata)

            return call

        for name, value in cls.__dict__.items():
            if not keep(name, value):
                continue
            f = getattr(client, name)
            f = add_default_metadata_wrapper(f)
            setattr(client, name, f)

        return client

    def get_default_discuss_client(self) -> glm.DiscussServiceClient:
        if self.discuss_client is None:
            self.discuss_client = self.make_client(glm.DiscussServiceClient)
        return self.discuss_client

    def get_default_text_client(self) -> glm.TextServiceClient:
        if self.text_client is None:
            self.text_client = self.make_client(glm.TextServiceClient)
        return self.text_client

    def get_default_discuss_async_client(self) -> glm.DiscussServiceAsyncClient:
        if self.discuss_async_client is None:
            self.discuss_async_client = self.make_client(glm.DiscussServiceAsyncClient)
        return self.discuss_async_client

    def get_default_model_client(self) -> glm.ModelServiceClient:
        if self.model_client is None:
            self.model_client = self.make_client(glm.ModelServiceClient)
        return self.model_client

    def get_default_operations_client(self) -> operations_v1.OperationsClient:
        if self.operations_client is None:
            self.model_client = get_default_model_client()
            self.operations_client = self.model_client._transport.operations_client

        return self.operations_client


_client_manager = _ClientManager()


def configure(
    *,
    api_key: str | None = None,
    credentials: ga_credentials.Credentials | dict | None = None,
    # The user can pass a string to choose `rest` or `grpc` or 'grpc_asyncio'.
    # See `_transport_registry` in `DiscussServiceClientMeta`.
    # Since the transport classes align with the client classes it wouldn't make
    # sense to accept a `Transport` object here even though the client classes can.
    # We could accept a dict since all the `Transport` classes take the same args,
    # but that seems rare. Users that need it can just switch to the low level API.
    transport: str | None = None,
    client_options: client_options_lib.ClientOptions | dict | None = None,
    client_info: gapic_v1.client_info.ClientInfo | None = None,
    default_metadata: Sequence[tuple[str, str]] = (),
):
    """Captures default client configuration.

    If no API key has been provided (either directly, or on `client_options`) and the
    `GOOGLE_API_KEY` environment variable is set, it will be used as the API key.

    Note: Not all arguments are detailed below. Refer to the `*ServiceClient` classes in
    `google.ai.generativelanguage` for details on the other arguments.

    Args:
        transport: A string, one of: [`rest`, `grpc`, `grpc_asyncio`].
        api_key: The API-Key to use when creating the default clients (each service uses
            a separate client). This is a shortcut for `client_options={"api_key": api_key}`.
            If omitted, and the `GOOGLE_API_KEY` environment variable is set, it will be
            used.
        default_metadata: Default (key, value) metadata pairs to send with every request.
            when using `transport="rest"` these are sent as HTTP headers.
    """
    return _client_manager.configure(
        api_key=api_key,
        credentials=credentials,
        transport=transport,
        client_options=client_options,
        client_info=client_info,
        default_metadata=default_metadata,
    )


def get_default_discuss_client() -> glm.DiscussServiceClient:
    return _client_manager.get_default_discuss_client()


def get_default_text_client() -> glm.TextServiceClient:
    return _client_manager.get_default_text_client()


def get_default_operations_client() -> operations_v1.OperationsClient:
    return _client_manager.get_default_operations_client()


def get_default_discuss_async_client() -> glm.DiscussServiceAsyncClient:
    return _client_manager.get_default_discuss_async_client()


def get_default_model_client() -> glm.ModelServiceAsyncClient:
    return _client_manager.get_default_model_client()
