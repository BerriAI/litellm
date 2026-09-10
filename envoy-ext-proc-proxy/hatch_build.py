import os

from grpc_tools import protoc
from hatchling.builders.hooks.plugin.interface import BuildHookInterface

PROTO_ROOTS = [
    "proto/envoy/annotations/deprecation.proto",
    "proto/envoy/config/core/v3/address.proto",
    "proto/envoy/config/core/v3/backoff.proto",
    "proto/envoy/config/core/v3/base.proto",
    "proto/envoy/config/core/v3/extension.proto",
    "proto/envoy/config/core/v3/http_uri.proto",
    "proto/envoy/config/core/v3/socket_option.proto",
    "proto/envoy/extensions/filters/http/ext_proc/v3/processing_mode.proto",
    "proto/envoy/service/ext_proc/v3/external_processor.proto",
    "proto/envoy/type/v3/http_status.proto",
    "proto/envoy/type/v3/percent.proto",
    "proto/envoy/type/v3/semantic_version.proto",
    "proto/udpa/annotations/migrate.proto",
    "proto/udpa/annotations/status.proto",
    "proto/udpa/annotations/versioning.proto",
    "proto/validate/validate.proto",
    "proto/xds/annotations/v3/status.proto",
    "proto/xds/core/v3/context_params.proto",
]


class CustomBuildHook(BuildHookInterface):
    def initialize(self, version: str, build_data: dict[str, object]) -> None:
        # Run protoc code generation before building the wheel
        try:
            os.mkdir("src/envoy_ext_proc_proxy/protogen")
        except FileExistsError:
            pass

        proto_include = protoc._get_resource_file_name("grpc_tools", "_proto")
        exit_code = protoc.main(
            [
                "grpc_tools.protoc",
                "-I=proto",
                f"-I={proto_include}",
                "--python_out=src/envoy_ext_proc_proxy/protogen",
                "--grpc_python_out=src/envoy_ext_proc_proxy/protogen",
            ]
            + PROTO_ROOTS
        )
        if exit_code != 0:
            raise RuntimeError(f"protoc failed with exit code {exit_code}")
