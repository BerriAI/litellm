# Copyright 2016 Google LLC
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

"""Wrap long-running operations returned from Google Cloud APIs."""

from typing import Dict

from google.longrunning import operations_pb2
from google.protobuf import json_format


_GOOGLE_APIS_PREFIX = "type.googleapis.com"

_TYPE_URL_MAP: Dict[str, type] = {}


def _compute_type_url(klass, prefix=_GOOGLE_APIS_PREFIX):
    """Compute a type URL for a klass.

    :type klass: type
    :param klass: class to be used as a factory for the given type

    :type prefix: str
    :param prefix: URL prefix for the type

    :rtype: str
    :returns: the URL, prefixed as appropriate
    """
    name = klass.DESCRIPTOR.full_name
    return "%s/%s" % (prefix, name)


def register_type(klass, type_url=None):
    """Register a klass as the factory for a given type URL.

    :type klass: :class:`type`
    :param klass: class to be used as a factory for the given type

    :type type_url: str
    :param type_url: (Optional) URL naming the type. If not provided,
                     infers the URL from the type descriptor.

    :raises ValueError: if a registration already exists for the URL.
    """
    if type_url is None:
        type_url = _compute_type_url(klass)
    if type_url in _TYPE_URL_MAP:
        if _TYPE_URL_MAP[type_url] is not klass:
            raise ValueError("Conflict: %s" % (_TYPE_URL_MAP[type_url],))

    _TYPE_URL_MAP[type_url] = klass


def _from_any(any_pb):
    """Convert an ``Any`` protobuf into the actual class.

    Uses the type URL to do the conversion.

    .. note::

        This assumes that the type URL is already registered.

    :type any_pb: :class:`google.protobuf.any_pb2.Any`
    :param any_pb: An any object to be converted.

    :rtype: object
    :returns: The instance (of the correct type) stored in the any
              instance.
    """
    klass = _TYPE_URL_MAP[any_pb.type_url]
    return klass.FromString(any_pb.value)


class Operation(object):
    """Representation of a Google API Long-Running Operation.

    .. _protobuf: https://github.com/googleapis/googleapis/blob/\
                  050400df0fdb16f63b63e9dee53819044bffc857/\
                  google/longrunning/operations.proto#L80
    .. _service: https://github.com/googleapis/googleapis/blob/\
                 050400df0fdb16f63b63e9dee53819044bffc857/\
                 google/longrunning/operations.proto#L38
    .. _JSON: https://cloud.google.com/speech/reference/rest/\
              v1beta1/operations#Operation

    This wraps an operation `protobuf`_ object and attempts to
    interact with the long-running operations `service`_ (specific
    to a given API). (Some services also offer a `JSON`_
    API that maps the same underlying data type.)

    :type name: str
    :param name: The fully-qualified path naming the operation.

    :type client: :class:`~google.cloud.client.Client`
    :param client: The client used to poll for the status of the operation.
                   If the operation was created via JSON/HTTP, the client
                   must own a :class:`~google.cloud._http.Connection`
                   to send polling requests. If created via protobuf, the
                   client must have a gRPC stub in the ``_operations_stub``
                   attribute.

    :type caller_metadata: dict
    :param caller_metadata: caller-assigned metadata about the operation
    """

    target = None
    """Instance assocated with the operations:  callers may set."""

    response = None
    """Response returned from completed operation.

    Only one of this and :attr:`error` can be populated.
    """

    error = None
    """Error that resulted from a failed (complete) operation.

    Only one of this and :attr:`response` can be populated.
    """

    metadata = None
    """Metadata about the current operation (as a protobuf).

    Code that uses operations must register the metadata types (via
    :func:`register_type`) to ensure that the metadata fields can be
    converted into the correct types.
    """

    _from_grpc = True

    def __init__(self, name, client, **caller_metadata):
        self.name = name
        self.client = client
        self.caller_metadata = caller_metadata.copy()
        self._complete = False

    @classmethod
    def from_pb(cls, operation_pb, client, **caller_metadata):
        """Factory:  construct an instance from a protobuf.

        :type operation_pb:
            :class:`~google.longrunning.operations_pb2.Operation`
        :param operation_pb: Protobuf to be parsed.

        :type client: object: must provide ``_operations_stub`` accessor.
        :param client: The client used to poll for the status of the operation.

        :type caller_metadata: dict
        :param caller_metadata: caller-assigned metadata about the operation

        :rtype: :class:`Operation`
        :returns: new instance, with attributes based on the protobuf.
        """
        result = cls(operation_pb.name, client, **caller_metadata)
        result._update_state(operation_pb)
        result._from_grpc = True
        return result

    @classmethod
    def from_dict(cls, operation, client, **caller_metadata):
        """Factory: construct an instance from a dictionary.

        :type operation: dict
        :param operation: Operation as a JSON object.

        :type client: :class:`~google.cloud.client.Client`
        :param client: The client used to poll for the status of the operation.

        :type caller_metadata: dict
        :param caller_metadata: caller-assigned metadata about the operation

        :rtype: :class:`Operation`
        :returns: new instance, with attributes based on the protobuf.
        """
        operation_pb = json_format.ParseDict(operation, operations_pb2.Operation())
        result = cls(operation_pb.name, client, **caller_metadata)
        result._update_state(operation_pb)
        result._from_grpc = False
        return result

    @property
    def complete(self):
        """Has the operation already completed?

        :rtype: bool
        :returns: True if already completed, else false.
        """
        return self._complete

    def _get_operation_rpc(self):
        """Polls the status of the current operation.

        Uses gRPC request to check.

        :rtype: :class:`~google.longrunning.operations_pb2.Operation`
        :returns: The latest status of the current operation.
        """
        request_pb = operations_pb2.GetOperationRequest(name=self.name)
        return self.client._operations_stub.GetOperation(request_pb)

    def _get_operation_http(self):
        """Checks the status of the current operation.

        Uses HTTP request to check.

        :rtype: :class:`~google.longrunning.operations_pb2.Operation`
        :returns: The latest status of the current operation.
        """
        path = "operations/%s" % (self.name,)
        api_response = self.client._connection.api_request(method="GET", path=path)
        return json_format.ParseDict(api_response, operations_pb2.Operation())

    def _get_operation(self):
        """Checks the status of the current operation.

        :rtype: :class:`~google.longrunning.operations_pb2.Operation`
        :returns: The latest status of the current operation.
        """
        if self._from_grpc:
            return self._get_operation_rpc()
        else:
            return self._get_operation_http()

    def _update_state(self, operation_pb):
        """Update the state of the current object based on operation.

        :type operation_pb:
            :class:`~google.longrunning.operations_pb2.Operation`
        :param operation_pb: Protobuf to be parsed.
        """
        if operation_pb.done:
            self._complete = True

        if operation_pb.HasField("metadata"):
            self.metadata = _from_any(operation_pb.metadata)

        result_type = operation_pb.WhichOneof("result")
        if result_type == "error":
            self.error = operation_pb.error
        elif result_type == "response":
            self.response = _from_any(operation_pb.response)

    def poll(self):
        """Check if the operation has finished.

        :rtype: bool
        :returns: A boolean indicating if the current operation has completed.
        :raises ValueError: if the operation
                 has already completed.
        """
        if self.complete:
            raise ValueError("The operation has completed.")

        operation_pb = self._get_operation()
        self._update_state(operation_pb)

        return self.complete
