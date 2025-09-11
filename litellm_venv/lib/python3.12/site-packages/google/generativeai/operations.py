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

import functools
from typing import Iterator

from google.ai import generativelanguage as glm

from google.generativeai import client as client_lib
from google.generativeai.types import model_types
from google.api_core import operation as operation_lib

import tqdm.auto as tqdm


def list_operations(*, client=None) -> Iterator[CreateTunedModelOperation]:
    if client is None:
        client = client_lib.get_default_operations_client()

    # The client returns an iterator of Operation protos (`Iterator[google.longrunning.operations_pb2.Operation]`)
    # not a gapic Operation object (`google.api_core.operation.Operation`)
    operations = (
        CreateTunedModelOperation.from_proto(op, client)
        for op in client.list_operations(name="", filter_="")
    )

    return operations


def get_operation(name: str, *, client=None) -> CreateTunedModelOperation:
    if client is None:
        client = client_lib.get_default_operations_client()

    op = client.get_operation(name=name)
    return CreateTunedModelOperation.from_proto(op, client)


def delete_operation(name: str, *, client=None):
    """Raises:
    google.api_core.exceptions.MethodNotImplemented: Not implemented."""
    if client is None:
        client = client_lib.get_default_operations_client()

    return client.delete_operation(name=name)


class CreateTunedModelOperation(operation_lib.Operation):
    @classmethod
    def from_proto(cls, proto, client):
        """
        result = getattr(proto, 'result', None)
        if result is not None:
            if result.value == b'':
                del proto.result
        """

        return from_gapic(
            cls=CreateTunedModelOperation,
            operation=proto,
            operations_client=client,
            result_type=glm.TunedModel,
            metadata_type=glm.CreateTunedModelMetadata,
        )

    @classmethod
    def from_core_operation(
        cls,
        operation: operation_lib.Operation,
    ):
        polling = getattr(operation, "_polling", None)
        retry = getattr(operation, "_retry", None)
        if polling is not None:
            # google.api_core v 2.11
            kwargs = {"polling": polling}
        elif retry is not None:
            # google.api_core v 2.10
            kwargs = {"retry": retry}
        else:
            kwargs = {}
        return cls(
            operation=operation._operation,
            refresh=operation._refresh,
            cancel=operation._cancel,
            result_type=operation._result_type,
            metadata_type=operation._metadata_type,
            **kwargs,
        )

    @property
    def name(self) -> str:
        return self._operation.name

    def update(self):
        """Refresh the current statuses in metadata/result/error"""
        self._refresh_and_update()

    def wait_bar(self, **kwargs) -> Iterator[glm.CreateTunedModelMetadata]:
        """A tqdm wait bar, yields `Operation` statuses until complete.

        Args:
            **kwargs: passed through to `tqdm.auto.tqdm(..., **kwargs)`

        Yields:
            Operation statuses as `glm.CreateTunedModelMetadata` objects.
        """
        bar = tqdm.tqdm(total=self.metadata.total_steps, initial=0, **kwargs)

        # done() includes a `_refresh_and_update`
        while not self.done():
            metadata = self.metadata
            bar.update(self.metadata.completed_steps - bar.n)
            yield metadata
        metadata = self.metadata
        bar.update(self.metadata.completed_steps - bar.n)
        return self.result()

    def set_result(self, result: glm.TunedModel):
        result = model_types.decode_tuned_model(result)
        super().set_result(result)


def from_gapic(
    cls,
    *,
    operation,
    operations_client,
    result_type,
    metadata_type,
    grpc_metadata=None,
    **kwargs,
):
    """`google.api_core.operation.from_gapic`, patched to allow subclasses."""
    refresh = functools.partial(
        operations_client.get_operation, operation.name, metadata=grpc_metadata
    )
    cancel = functools.partial(
        operations_client.cancel_operation,
        operation.name,
        metadata=grpc_metadata,
    )
    return cls(operation, refresh, cancel, result_type, metadata_type, **kwargs)
