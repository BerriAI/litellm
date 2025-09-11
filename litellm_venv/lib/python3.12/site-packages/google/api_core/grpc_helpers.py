# Copyright 2017 Google LLC
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

"""Helpers for :mod:`grpc`."""
from typing import Generic, Iterator, Optional, TypeVar

import collections
import functools
import warnings

import grpc

from google.api_core import exceptions
import google.auth
import google.auth.credentials
import google.auth.transport.grpc
import google.auth.transport.requests
import google.protobuf

PROTOBUF_VERSION = google.protobuf.__version__

# The grpcio-gcp package only has support for protobuf < 4
if PROTOBUF_VERSION[0:2] == "3.":  # pragma: NO COVER
    try:
        import grpc_gcp

        warnings.warn(
            """Support for grpcio-gcp is deprecated. This feature will be
            removed from `google-api-core` after January 1, 2024. If you need to
            continue to use this feature, please pin to a specific version of
            `google-api-core`.""",
            DeprecationWarning,
        )
        HAS_GRPC_GCP = True
    except ImportError:
        HAS_GRPC_GCP = False
else:
    HAS_GRPC_GCP = False


# The list of gRPC Callable interfaces that return iterators.
_STREAM_WRAP_CLASSES = (grpc.UnaryStreamMultiCallable, grpc.StreamStreamMultiCallable)

# denotes the proto response type for grpc calls
P = TypeVar("P")


def _patch_callable_name(callable_):
    """Fix-up gRPC callable attributes.

    gRPC callable lack the ``__name__`` attribute which causes
    :func:`functools.wraps` to error. This adds the attribute if needed.
    """
    if not hasattr(callable_, "__name__"):
        callable_.__name__ = callable_.__class__.__name__


def _wrap_unary_errors(callable_):
    """Map errors for Unary-Unary and Stream-Unary gRPC callables."""
    _patch_callable_name(callable_)

    @functools.wraps(callable_)
    def error_remapped_callable(*args, **kwargs):
        try:
            return callable_(*args, **kwargs)
        except grpc.RpcError as exc:
            raise exceptions.from_grpc_error(exc) from exc

    return error_remapped_callable


class _StreamingResponseIterator(Generic[P], grpc.Call):
    def __init__(self, wrapped, prefetch_first_result=True):
        self._wrapped = wrapped

        # This iterator is used in a retry context, and returned outside after init.
        # gRPC will not throw an exception until the stream is consumed, so we need
        # to retrieve the first result, in order to fail, in order to trigger a retry.
        try:
            if prefetch_first_result:
                self._stored_first_result = next(self._wrapped)
        except TypeError:
            # It is possible the wrapped method isn't an iterable (a grpc.Call
            # for instance). If this happens don't store the first result.
            pass
        except StopIteration:
            # ignore stop iteration at this time. This should be handled outside of retry.
            pass

    def __iter__(self) -> Iterator[P]:
        """This iterator is also an iterable that returns itself."""
        return self

    def __next__(self) -> P:
        """Get the next response from the stream.

        Returns:
            protobuf.Message: A single response from the stream.
        """
        try:
            if hasattr(self, "_stored_first_result"):
                result = self._stored_first_result
                del self._stored_first_result
                return result
            return next(self._wrapped)
        except grpc.RpcError as exc:
            # If the stream has already returned data, we cannot recover here.
            raise exceptions.from_grpc_error(exc) from exc

    # grpc.Call & grpc.RpcContext interface

    def add_callback(self, callback):
        return self._wrapped.add_callback(callback)

    def cancel(self):
        return self._wrapped.cancel()

    def code(self):
        return self._wrapped.code()

    def details(self):
        return self._wrapped.details()

    def initial_metadata(self):
        return self._wrapped.initial_metadata()

    def is_active(self):
        return self._wrapped.is_active()

    def time_remaining(self):
        return self._wrapped.time_remaining()

    def trailing_metadata(self):
        return self._wrapped.trailing_metadata()


# public type alias denoting the return type of streaming gapic calls
GrpcStream = _StreamingResponseIterator[P]


def _wrap_stream_errors(callable_):
    """Wrap errors for Unary-Stream and Stream-Stream gRPC callables.

    The callables that return iterators require a bit more logic to re-map
    errors when iterating. This wraps both the initial invocation and the
    iterator of the return value to re-map errors.
    """
    _patch_callable_name(callable_)

    @functools.wraps(callable_)
    def error_remapped_callable(*args, **kwargs):
        try:
            result = callable_(*args, **kwargs)
            # Auto-fetching the first result causes PubSub client's streaming pull
            # to hang when re-opening the stream, thus we need examine the hacky
            # hidden flag to see if pre-fetching is disabled.
            # https://github.com/googleapis/python-pubsub/issues/93#issuecomment-630762257
            prefetch_first = getattr(callable_, "_prefetch_first_result_", True)
            return _StreamingResponseIterator(
                result, prefetch_first_result=prefetch_first
            )
        except grpc.RpcError as exc:
            raise exceptions.from_grpc_error(exc) from exc

    return error_remapped_callable


def wrap_errors(callable_):
    """Wrap a gRPC callable and map :class:`grpc.RpcErrors` to friendly error
    classes.

    Errors raised by the gRPC callable are mapped to the appropriate
    :class:`google.api_core.exceptions.GoogleAPICallError` subclasses.
    The original `grpc.RpcError` (which is usually also a `grpc.Call`) is
    available from the ``response`` property on the mapped exception. This
    is useful for extracting metadata from the original error.

    Args:
        callable_ (Callable): A gRPC callable.

    Returns:
        Callable: The wrapped gRPC callable.
    """
    if isinstance(callable_, _STREAM_WRAP_CLASSES):
        return _wrap_stream_errors(callable_)
    else:
        return _wrap_unary_errors(callable_)


def _create_composite_credentials(
    credentials=None,
    credentials_file=None,
    default_scopes=None,
    scopes=None,
    ssl_credentials=None,
    quota_project_id=None,
    default_host=None,
):
    """Create the composite credentials for secure channels.

    Args:
        credentials (google.auth.credentials.Credentials): The credentials. If
            not specified, then this function will attempt to ascertain the
            credentials from the environment using :func:`google.auth.default`.
        credentials_file (str): A file with credentials that can be loaded with
            :func:`google.auth.load_credentials_from_file`. This argument is
            mutually exclusive with credentials.

            .. warning::
                Important: If you accept a credential configuration (credential JSON/File/Stream)
                from an external source for authentication to Google Cloud Platform, you must
                validate it before providing it to any Google API or client library. Providing an
                unvalidated credential configuration to Google APIs or libraries can compromise
                the security of your systems and data. For more information, refer to
                `Validate credential configurations from external sources`_.

            .. _Validate credential configurations from external sources:

            https://cloud.google.com/docs/authentication/external/externally-sourced-credentials
        default_scopes (Sequence[str]): A optional list of scopes needed for this
            service. These are only used when credentials are not specified and
            are passed to :func:`google.auth.default`.
        scopes (Sequence[str]): A optional list of scopes needed for this
            service. These are only used when credentials are not specified and
            are passed to :func:`google.auth.default`.
        ssl_credentials (grpc.ChannelCredentials): Optional SSL channel
            credentials. This can be used to specify different certificates.
        quota_project_id (str): An optional project to use for billing and quota.
        default_host (str): The default endpoint. e.g., "pubsub.googleapis.com".

    Returns:
        grpc.ChannelCredentials: The composed channel credentials object.

    Raises:
        google.api_core.DuplicateCredentialArgs: If both a credentials object and credentials_file are passed.
    """
    if credentials and credentials_file:
        raise exceptions.DuplicateCredentialArgs(
            "'credentials' and 'credentials_file' are mutually exclusive."
        )

    if credentials_file:
        credentials, _ = google.auth.load_credentials_from_file(
            credentials_file, scopes=scopes, default_scopes=default_scopes
        )
    elif credentials:
        credentials = google.auth.credentials.with_scopes_if_required(
            credentials, scopes=scopes, default_scopes=default_scopes
        )
    else:
        credentials, _ = google.auth.default(
            scopes=scopes, default_scopes=default_scopes
        )

    if quota_project_id and isinstance(
        credentials, google.auth.credentials.CredentialsWithQuotaProject
    ):
        credentials = credentials.with_quota_project(quota_project_id)

    request = google.auth.transport.requests.Request()

    # Create the metadata plugin for inserting the authorization header.
    metadata_plugin = google.auth.transport.grpc.AuthMetadataPlugin(
        credentials,
        request,
        default_host=default_host,
    )

    # Create a set of grpc.CallCredentials using the metadata plugin.
    google_auth_credentials = grpc.metadata_call_credentials(metadata_plugin)

    # if `ssl_credentials` is set, use `grpc.composite_channel_credentials` instead of
    # `grpc.compute_engine_channel_credentials` as the former supports passing
    # `ssl_credentials` via `channel_credentials` which is needed for mTLS.
    if ssl_credentials:
        # Combine the ssl credentials and the authorization credentials.
        # See https://grpc.github.io/grpc/python/grpc.html#grpc.composite_channel_credentials
        return grpc.composite_channel_credentials(
            ssl_credentials, google_auth_credentials
        )
    else:
        # Use grpc.compute_engine_channel_credentials in order to support Direct Path.
        # See https://grpc.github.io/grpc/python/grpc.html#grpc.compute_engine_channel_credentials
        # TODO(https://github.com/googleapis/python-api-core/issues/598):
        # Although `grpc.compute_engine_channel_credentials` returns channel credentials
        # outside of a Google Compute Engine environment (GCE), we should determine if
        # there is a way to reliably detect a GCE environment so that
        # `grpc.compute_engine_channel_credentials` is not called outside of GCE.
        return grpc.compute_engine_channel_credentials(google_auth_credentials)


def create_channel(
    target,
    credentials=None,
    scopes=None,
    ssl_credentials=None,
    credentials_file=None,
    quota_project_id=None,
    default_scopes=None,
    default_host=None,
    compression=None,
    attempt_direct_path: Optional[bool] = False,
    **kwargs,
):
    """Create a secure channel with credentials.

    Args:
        target (str): The target service address in the format 'hostname:port'.
        credentials (google.auth.credentials.Credentials): The credentials. If
            not specified, then this function will attempt to ascertain the
            credentials from the environment using :func:`google.auth.default`.
        scopes (Sequence[str]): A optional list of scopes needed for this
            service. These are only used when credentials are not specified and
            are passed to :func:`google.auth.default`.
        ssl_credentials (grpc.ChannelCredentials): Optional SSL channel
            credentials. This can be used to specify different certificates.
        credentials_file (str): A file with credentials that can be loaded with
            :func:`google.auth.load_credentials_from_file`. This argument is
            mutually exclusive with credentials.

            .. warning::
                Important: If you accept a credential configuration (credential JSON/File/Stream)
                from an external source for authentication to Google Cloud Platform, you must
                validate it before providing it to any Google API or client library. Providing an
                unvalidated credential configuration to Google APIs or libraries can compromise
                the security of your systems and data. For more information, refer to
                `Validate credential configurations from external sources`_.

            .. _Validate credential configurations from external sources:

            https://cloud.google.com/docs/authentication/external/externally-sourced-credentials
        quota_project_id (str): An optional project to use for billing and quota.
        default_scopes (Sequence[str]): Default scopes passed by a Google client
            library. Use 'scopes' for user-defined scopes.
        default_host (str): The default endpoint. e.g., "pubsub.googleapis.com".
        compression (grpc.Compression): An optional value indicating the
            compression method to be used over the lifetime of the channel.
        attempt_direct_path (Optional[bool]): If set, Direct Path will be attempted
            when the request is made. Direct Path is only available within a Google
            Compute Engine (GCE) environment and provides a proxyless connection
            which increases the available throughput, reduces latency, and increases
            reliability. Note:

            - This argument should only be set in a GCE environment and for Services
              that are known to support Direct Path.
            - If this argument is set outside of GCE, then this request will fail
              unless the back-end service happens to have configured fall-back to DNS.
            - If the request causes a `ServiceUnavailable` response, it is recommended
              that the client repeat the request with `attempt_direct_path` set to
              `False` as the Service may not support Direct Path.
            - Using `ssl_credentials` with `attempt_direct_path` set to `True` will
              result in `ValueError` as this combination  is not yet supported.

        kwargs: Additional key-word args passed to
            :func:`grpc_gcp.secure_channel` or :func:`grpc.secure_channel`.
            Note: `grpc_gcp` is only supported in environments with protobuf < 4.0.0.

    Returns:
        grpc.Channel: The created channel.

    Raises:
        google.api_core.DuplicateCredentialArgs: If both a credentials object and credentials_file are passed.
        ValueError: If `ssl_credentials` is set and `attempt_direct_path` is set to `True`.
    """

    # If `ssl_credentials` is set and `attempt_direct_path` is set to `True`,
    # raise ValueError as this is not yet supported.
    # See https://github.com/googleapis/python-api-core/issues/590
    if ssl_credentials and attempt_direct_path:
        raise ValueError("Using ssl_credentials with Direct Path is not supported")

    composite_credentials = _create_composite_credentials(
        credentials=credentials,
        credentials_file=credentials_file,
        default_scopes=default_scopes,
        scopes=scopes,
        ssl_credentials=ssl_credentials,
        quota_project_id=quota_project_id,
        default_host=default_host,
    )

    # Note that grpcio-gcp is deprecated
    if HAS_GRPC_GCP:  # pragma: NO COVER
        if compression is not None and compression != grpc.Compression.NoCompression:
            warnings.warn(
                "The `compression` argument is ignored for grpc_gcp.secure_channel creation.",
                DeprecationWarning,
            )
        if attempt_direct_path:
            warnings.warn(
                """The `attempt_direct_path` argument is ignored for grpc_gcp.secure_channel creation.""",
                DeprecationWarning,
            )
        return grpc_gcp.secure_channel(target, composite_credentials, **kwargs)

    if attempt_direct_path:
        target = _modify_target_for_direct_path(target)

    return grpc.secure_channel(
        target, composite_credentials, compression=compression, **kwargs
    )


def _modify_target_for_direct_path(target: str) -> str:
    """
    Given a target, return a modified version which is compatible with Direct Path.

    Args:
        target (str): The target service address in the format 'hostname[:port]' or
            'dns://hostname[:port]'.

    Returns:
        target (str): The target service address which is converted into a format compatible with Direct Path.
            If the target contains `dns:///` or does not contain `:///`, the target will be converted in
            a format compatible with Direct Path; otherwise the original target will be returned as the
            original target may already denote Direct Path.
    """

    # A DNS prefix may be included with the target to indicate the endpoint is living in the Internet,
    # outside of Google Cloud Platform.
    dns_prefix = "dns:///"
    # Remove "dns:///" if `attempt_direct_path` is set to True as
    # the Direct Path prefix `google-c2p:///` will be used instead.
    target = target.replace(dns_prefix, "")

    direct_path_separator = ":///"
    if direct_path_separator not in target:
        target_without_port = target.split(":")[0]
        # Modify the target to use Direct Path by adding the `google-c2p:///` prefix
        target = f"google-c2p{direct_path_separator}{target_without_port}"
    return target


_MethodCall = collections.namedtuple(
    "_MethodCall", ("request", "timeout", "metadata", "credentials", "compression")
)

_ChannelRequest = collections.namedtuple("_ChannelRequest", ("method", "request"))


class _CallableStub(object):
    """Stub for the grpc.*MultiCallable interfaces."""

    def __init__(self, method, channel):
        self._method = method
        self._channel = channel
        self.response = None
        """Union[protobuf.Message, Callable[protobuf.Message], exception]:
        The response to give when invoking this callable. If this is a
        callable, it will be invoked with the request protobuf. If it's an
        exception, the exception will be raised when this is invoked.
        """
        self.responses = None
        """Iterator[
            Union[protobuf.Message, Callable[protobuf.Message], exception]]:
        An iterator of responses. If specified, self.response will be populated
        on each invocation by calling ``next(self.responses)``."""
        self.requests = []
        """List[protobuf.Message]: All requests sent to this callable."""
        self.calls = []
        """List[Tuple]: All invocations of this callable. Each tuple is the
        request, timeout, metadata, compression, and credentials."""

    def __call__(
        self, request, timeout=None, metadata=None, credentials=None, compression=None
    ):
        self._channel.requests.append(_ChannelRequest(self._method, request))
        self.calls.append(
            _MethodCall(request, timeout, metadata, credentials, compression)
        )
        self.requests.append(request)

        response = self.response
        if self.responses is not None:
            if response is None:
                response = next(self.responses)
            else:
                raise ValueError(
                    "{method}.response and {method}.responses are mutually "
                    "exclusive.".format(method=self._method)
                )

        if callable(response):
            return response(request)

        if isinstance(response, Exception):
            raise response

        if response is not None:
            return response

        raise ValueError('Method stub for "{}" has no response.'.format(self._method))


def _simplify_method_name(method):
    """Simplifies a gRPC method name.

    When gRPC invokes the channel to create a callable, it gives a full
    method name like "/google.pubsub.v1.Publisher/CreateTopic". This
    returns just the name of the method, in this case "CreateTopic".

    Args:
        method (str): The name of the method.

    Returns:
        str: The simplified name of the method.
    """
    return method.rsplit("/", 1).pop()


class ChannelStub(grpc.Channel):
    """A testing stub for the grpc.Channel interface.

    This can be used to test any client that eventually uses a gRPC channel
    to communicate. By passing in a channel stub, you can configure which
    responses are returned and track which requests are made.

    For example:

    .. code-block:: python

        channel_stub = grpc_helpers.ChannelStub()
        client = FooClient(channel=channel_stub)

        channel_stub.GetFoo.response = foo_pb2.Foo(name='bar')

        foo = client.get_foo(labels=['baz'])

        assert foo.name == 'bar'
        assert channel_stub.GetFoo.requests[0].labels = ['baz']

    Each method on the stub can be accessed and configured on the channel.
    Here's some examples of various configurations:

    .. code-block:: python

        # Return a basic response:

        channel_stub.GetFoo.response = foo_pb2.Foo(name='bar')
        assert client.get_foo().name == 'bar'

        # Raise an exception:
        channel_stub.GetFoo.response = NotFound('...')

        with pytest.raises(NotFound):
            client.get_foo()

        # Use a sequence of responses:
        channel_stub.GetFoo.responses = iter([
            foo_pb2.Foo(name='bar'),
            foo_pb2.Foo(name='baz'),
        ])

        assert client.get_foo().name == 'bar'
        assert client.get_foo().name == 'baz'

        # Use a callable

        def on_get_foo(request):
            return foo_pb2.Foo(name='bar' + request.id)

        channel_stub.GetFoo.response = on_get_foo

        assert client.get_foo(id='123').name == 'bar123'
    """

    def __init__(self, responses=[]):
        self.requests = []
        """Sequence[Tuple[str, protobuf.Message]]: A list of all requests made
        on this channel in order. The tuple is of method name, request
        message."""
        self._method_stubs = {}

    def _stub_for_method(self, method):
        method = _simplify_method_name(method)
        self._method_stubs[method] = _CallableStub(method, self)
        return self._method_stubs[method]

    def __getattr__(self, key):
        try:
            return self._method_stubs[key]
        except KeyError:
            raise AttributeError

    def unary_unary(
        self,
        method,
        request_serializer=None,
        response_deserializer=None,
        _registered_method=False,
    ):
        """grpc.Channel.unary_unary implementation."""
        return self._stub_for_method(method)

    def unary_stream(
        self,
        method,
        request_serializer=None,
        response_deserializer=None,
        _registered_method=False,
    ):
        """grpc.Channel.unary_stream implementation."""
        return self._stub_for_method(method)

    def stream_unary(
        self,
        method,
        request_serializer=None,
        response_deserializer=None,
        _registered_method=False,
    ):
        """grpc.Channel.stream_unary implementation."""
        return self._stub_for_method(method)

    def stream_stream(
        self,
        method,
        request_serializer=None,
        response_deserializer=None,
        _registered_method=False,
    ):
        """grpc.Channel.stream_stream implementation."""
        return self._stub_for_method(method)

    def subscribe(self, callback, try_to_connect=False):
        """grpc.Channel.subscribe implementation."""
        pass

    def unsubscribe(self, callback):
        """grpc.Channel.unsubscribe implementation."""
        pass

    def close(self):
        """grpc.Channel.close implementation."""
        pass
