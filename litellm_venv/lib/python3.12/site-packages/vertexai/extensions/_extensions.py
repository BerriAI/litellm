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
#
import json
from typing import Optional, Sequence, Union

from google.cloud.aiplatform import base
from google.cloud.aiplatform import initializer
from google.cloud.aiplatform import utils as aip_utils
from google.cloud.aiplatform_v1beta1 import types
from vertexai.reasoning_engines import _utils

from google.protobuf import struct_pb2

_LOGGER = base.Logger(__name__)

_AuthConfigOrJson = Union[_utils.JsonDict, types.AuthConfig]
_StructOrJson = Union[_utils.JsonDict, struct_pb2.Struct]
_RuntimeConfigOrJson = Union[_utils.JsonDict, types.RuntimeConfig]


_VERTEX_EXTENSION_HUB = {
    "code_interpreter": {
        "display_name": "Code Interpreter",
        "description": (
            "This extension generates and executes code in the specified language"
        ),
        "manifest": {
            "name": "code_interpreter_tool",
            "description": "Google Code Interpreter Extension",
            "api_spec": {
                "open_api_gcs_uri": (
                    "gs://vertex-extension-public/code_interpreter.yaml"
                ),
            },
            "auth_config": {
                "auth_type": "GOOGLE_SERVICE_ACCOUNT_AUTH",
                "google_service_account_config": {},
            },
        },
    },
    "vertex_ai_search": {
        "display_name": "Vertex AI Search",
        "description": "This extension generates and executes search queries",
        "manifest": {
            "name": "vertex_ai_search",
            "description": "Vertex AI Search Extension",
            "api_spec": {
                "open_api_gcs_uri": (
                    "gs://vertex-extension-public/vertex_ai_search.yaml"
                ),
            },
            "auth_config": {
                "auth_type": "GOOGLE_SERVICE_ACCOUNT_AUTH",
                "google_service_account_config": {},
            },
        },
    },
}


class Extension(base.VertexAiResourceNounWithFutureManager):
    """Represents a Vertex AI Extension resource."""

    client_class = aip_utils.ExtensionRegistryClientWithOverride
    _resource_noun = "extension"
    _getter_method = "get_extension"
    _list_method = "list_extensions"
    _delete_method = "delete_extension"
    _parse_resource_name_method = "parse_extension_path"
    _format_resource_name_method = "extension_path"

    def __init__(self, extension_name: str):
        """Retrieves an extension resource.

        Args:
            extension_name (str):
                Required. A fully-qualified resource name or ID such as
                "projects/123/locations/us-central1/extensions/456" or
                "456" when project and location are initialized or passed.
        """
        super().__init__(resource_name=extension_name)
        self.execution_api_client = initializer.global_config.create_client(
            client_class=aip_utils.ExtensionExecutionClientWithOverride,
        )
        self._gca_resource = self._get_gca_resource(resource_name=extension_name)
        self._api_spec = None
        self._operation_schemas = None

    @classmethod
    def create(
        cls,
        manifest: Union[_utils.JsonDict, types.ExtensionManifest],
        *,
        extension_name: Optional[str] = None,
        display_name: Optional[str] = None,
        description: Optional[str] = None,
        runtime_config: Optional[_RuntimeConfigOrJson] = None,
    ):
        """Creates a new Extension.

        Args:
            manifest (Union[dict[str, Any], ExtensionManifest]):
                Required. The manifest for the Extension to be created.
            extension_name (str):
                Optional. A fully-qualified extension resource name or extension
                ID such as "projects/123/locations/us-central1/extensions/456" or
                "456" when project and location are initialized or passed. If
                specifying the extension ID, it should be 4-63 characters, valid
                characters are lowercase letters, numbers and hyphens ("-"),
                and it should start with a number or a lower-case letter. If not
                provided, Vertex AI will generate a value for this ID.
            display_name (str):
                Optional. The user-defined name of the Extension.
                The name can be up to 128 characters long and can comprise any
                UTF-8 character.
            description (str):
                Optional. The description of the Extension.
            runtime_config (Union[dict[str, Any], RuntimeConfig]):
                Optional. Runtime config controlling the runtime behavior of
                this Extension. Defaults to None.

        Returns:
            Extension: The extension that was created.
        """
        sdk_resource = cls.__new__(cls)
        base.VertexAiResourceNounWithFutureManager.__init__(
            sdk_resource,
            resource_name=extension_name,
        )
        extension = types.Extension(
            name=extension_name,
            display_name=display_name or cls._generate_display_name(),
            description=description,
            manifest=_utils.to_proto(manifest, types.ExtensionManifest()),
        )
        if runtime_config:
            extension.runtime_config = _utils.to_proto(
                runtime_config,
                types.RuntimeConfig(),
            )
        operation_future = sdk_resource.api_client.import_extension(
            parent=initializer.global_config.common_location_path(),
            extension=extension,
        )
        _LOGGER.log_create_with_lro(cls, operation_future)
        created_extension = operation_future.result()
        _LOGGER.log_create_complete(
            cls,
            created_extension,
            cls._resource_noun,
            module_name="vertexai.preview.extensions",
        )
        # We use `._get_gca_resource(...)` instead of `created_extension` to
        # fully instantiate the attributes of the extension.
        sdk_resource._gca_resource = sdk_resource._get_gca_resource(
            resource_name=created_extension.name
        )
        sdk_resource.execution_api_client = initializer.global_config.create_client(
            client_class=aip_utils.ExtensionExecutionClientWithOverride,
        )
        sdk_resource._api_spec = None
        sdk_resource._operation_schemas = None
        return sdk_resource

    @property
    def resource_name(self) -> str:
        """Full qualified resource name for the extension."""
        return self._gca_resource.name

    def api_spec(self) -> _utils.JsonDict:
        """Returns the (Open)API Spec of the extension."""
        if self._api_spec is None:
            self._api_spec = _load_api_spec(self._gca_resource.manifest.api_spec)
        return self._api_spec

    def operation_schemas(self) -> Sequence[_utils.JsonDict]:
        """Returns the (Open)API schemas for each operation of the extension."""
        if self._operation_schemas is None:
            self._operation_schemas = [
                _utils.to_dict(op.function_declaration)
                for op in self._gca_resource.extension_operations
            ]
        return self._operation_schemas

    def execute(
        self,
        operation_id: str,
        operation_params: Optional[_StructOrJson] = None,
        runtime_auth_config: Optional[_AuthConfigOrJson] = None,
    ) -> Union[_utils.JsonDict, str]:
        """Executes an operation of the extension with the specified params.

        Args:
          operation_id (str):
              Required. The ID of the operation to be executed.
          operation_params (Union[dict[str, Any], Struct]):
              Optional. Parameters used for executing the operation. It should
              be in a form of map with param name as the key and actual param
              value as the value. E.g. if this operation requires a param
              "name" to be set to "abc", you can set this to {"name": "abc"}.
              Defaults to an empty dictionary.
          runtime_auth_config (Union[dict[str, Any], AuthConfig]):
              Optional. The Auth configuration to execute the operation.

        Returns:
            The result of executing the extension operation.
        """
        request = types.ExecuteExtensionRequest(
            name=self.resource_name,
            operation_id=operation_id,
            operation_params=operation_params,
        )
        if runtime_auth_config:
            request.runtime_auth_config = _utils.to_proto(
                runtime_auth_config,
                types.AuthConfig(),
            )
        response = self.execution_api_client.execute_extension(request)
        return _try_parse_execution_response(response)

    @classmethod
    def from_hub(
        cls,
        name: str,
        *,
        runtime_config: Optional[_RuntimeConfigOrJson] = None,
    ):
        """Creates a new Extension from the set of first party extensions.

        Args:
            name (str):
                Required. The name of the extension in the hub to be created.
                Supported values are "code_interpreter" and "vertex_ai_search".
            runtime_config (Union[dict[str, Any], RuntimeConfig]):
                Optional. Runtime config controlling the runtime behavior of
                the Extension. Defaults to None.

        Returns:
            Extension: The extension that was created.

        Raises:
            ValueError: If the `name` is not supported in the hub.
            ValueError: If the `runtime_config` is specified but inconsistent
            with the name (e.g. the name was "code_interpreter" but the
            runtime_config was based on "vertex_ai_search_runtime_config").
        """
        if runtime_config:
            runtime_config = _utils.to_proto(
                runtime_config,
                types.RuntimeConfig(),
            )
        if name == "code_interpreter":
            if runtime_config and not getattr(
                runtime_config,
                "code_interpreter_runtime_config",
                None,
            ):
                raise ValueError(
                    "code_interpreter_runtime_config is required for "
                    "code_interpreter extension"
                )
        elif name == "vertex_ai_search":
            if not runtime_config:
                raise ValueError(
                    "runtime_config is required for vertex_ai_search extension"
                )
            if runtime_config and not getattr(
                runtime_config,
                "vertex_ai_search_runtime_config",
                None,
            ):
                raise ValueError(
                    "vertex_ai_search_runtime_config is required for "
                    "vertex_ai_search extension"
                )
        else:
            raise ValueError(f"Unsupported 1P extension name: {name}")
        extension_info = _VERTEX_EXTENSION_HUB[name]
        return cls.create(
            display_name=extension_info["display_name"],
            description=extension_info["description"],
            manifest=extension_info["manifest"],
            runtime_config=runtime_config,
        )


def _try_parse_execution_response(
    response: types.ExecuteExtensionResponse,
) -> Union[_utils.JsonDict, str]:
    content: str = response.content
    try:
        content = json.loads(response.content)
    except:
        pass
    return content


def _load_api_spec(api_spec) -> _utils.JsonDict:
    """Loads the (Open)API Spec of the extension and converts it to JSON."""
    if api_spec.open_api_yaml:
        yaml = aip_utils.yaml_utils._maybe_import_yaml()
        return yaml.safe_load(api_spec.open_api_yaml)
    elif api_spec.open_api_gcs_uri:
        return aip_utils.yaml_utils.load_yaml(api_spec.open_api_gcs_uri)
    return {}
