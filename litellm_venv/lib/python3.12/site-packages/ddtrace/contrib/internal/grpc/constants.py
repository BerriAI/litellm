import grpc


GRPC_PIN_MODULE_SERVER = grpc.Server
GRPC_PIN_MODULE_CLIENT = grpc.Channel
GRPC_METHOD_PATH_KEY = "grpc.method.path"
GRPC_METHOD_PACKAGE_SERVICE_KEY = "rpc.service"
GRPC_METHOD_PACKAGE_KEY = "grpc.method.package"
GRPC_METHOD_SERVICE_KEY = "grpc.method.service"
GRPC_METHOD_NAME_KEY = "grpc.method.name"
GRPC_METHOD_KIND_KEY = "grpc.method.kind"
GRPC_STATUS_CODE_KEY = "grpc.status.code"
GRPC_REQUEST_METADATA_PREFIX_KEY = "grpc.request.metadata."
GRPC_RESPONSE_METADATA_PREFIX_KEY = "grpc.response.metadata."
GRPC_HOST_KEY = "grpc.host"
GRPC_SPAN_KIND_KEY = "span.kind"
GRPC_SPAN_KIND_VALUE_CLIENT = "client"
GRPC_SPAN_KIND_VALUE_SERVER = "server"
GRPC_METHOD_KIND_UNARY = "unary"
GRPC_METHOD_KIND_CLIENT_STREAMING = "client_streaming"
GRPC_METHOD_KIND_SERVER_STREAMING = "server_streaming"
GRPC_METHOD_KIND_BIDI_STREAMING = "bidi_streaming"
GRPC_SERVICE_SERVER = "grpc-server"
GRPC_AIO_SERVICE_SERVER = "grpc-aio-server"
GRPC_SERVICE_CLIENT = "grpc-client"
GRPC_AIO_SERVICE_CLIENT = "grpc-aio-client"
