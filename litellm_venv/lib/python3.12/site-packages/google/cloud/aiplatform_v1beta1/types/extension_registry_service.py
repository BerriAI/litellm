# -*- coding: utf-8 -*-
# Copyright 2024 Google LLC
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
from __future__ import annotations

from typing import MutableMapping, MutableSequence

import proto  # type: ignore

from google.cloud.aiplatform_v1beta1.types import extension as gca_extension
from google.cloud.aiplatform_v1beta1.types import operation
from google.protobuf import field_mask_pb2  # type: ignore


__protobuf__ = proto.module(
    package="google.cloud.aiplatform.v1beta1",
    manifest={
        "ImportExtensionRequest",
        "ImportExtensionOperationMetadata",
        "GetExtensionRequest",
        "UpdateExtensionRequest",
        "ListExtensionsRequest",
        "ListExtensionsResponse",
        "DeleteExtensionRequest",
    },
)


class ImportExtensionRequest(proto.Message):
    r"""Request message for
    [ExtensionRegistryService.ImportExtension][google.cloud.aiplatform.v1beta1.ExtensionRegistryService.ImportExtension].

    Attributes:
        parent (str):
            Required. The resource name of the Location to import the
            Extension in. Format:
            ``projects/{project}/locations/{location}``
        extension (google.cloud.aiplatform_v1beta1.types.Extension):
            Required. The Extension to import.
    """

    parent: str = proto.Field(
        proto.STRING,
        number=1,
    )
    extension: gca_extension.Extension = proto.Field(
        proto.MESSAGE,
        number=2,
        message=gca_extension.Extension,
    )


class ImportExtensionOperationMetadata(proto.Message):
    r"""Details of
    [ExtensionRegistryService.ImportExtension][google.cloud.aiplatform.v1beta1.ExtensionRegistryService.ImportExtension]
    operation.

    Attributes:
        generic_metadata (google.cloud.aiplatform_v1beta1.types.GenericOperationMetadata):
            The common part of the operation metadata.
    """

    generic_metadata: operation.GenericOperationMetadata = proto.Field(
        proto.MESSAGE,
        number=1,
        message=operation.GenericOperationMetadata,
    )


class GetExtensionRequest(proto.Message):
    r"""Request message for
    [ExtensionRegistryService.GetExtension][google.cloud.aiplatform.v1beta1.ExtensionRegistryService.GetExtension].

    Attributes:
        name (str):
            Required. The name of the Extension resource. Format:
            ``projects/{project}/locations/{location}/extensions/{extension}``
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


class UpdateExtensionRequest(proto.Message):
    r"""Request message for
    [ExtensionRegistryService.UpdateExtension][google.cloud.aiplatform.v1beta1.ExtensionRegistryService.UpdateExtension].

    Attributes:
        extension (google.cloud.aiplatform_v1beta1.types.Extension):
            Required. The Extension which replaces the
            resource on the server.
        update_mask (google.protobuf.field_mask_pb2.FieldMask):
            Required. Mask specifying which fields to update. Supported
            fields:

            ::

               * `display_name`
               * `description`
               * `tool_use_examples`
    """

    extension: gca_extension.Extension = proto.Field(
        proto.MESSAGE,
        number=1,
        message=gca_extension.Extension,
    )
    update_mask: field_mask_pb2.FieldMask = proto.Field(
        proto.MESSAGE,
        number=2,
        message=field_mask_pb2.FieldMask,
    )


class ListExtensionsRequest(proto.Message):
    r"""Request message for
    [ExtensionRegistryService.ListExtensions][google.cloud.aiplatform.v1beta1.ExtensionRegistryService.ListExtensions].

    Attributes:
        parent (str):
            Required. The resource name of the Location to list the
            Extensions from. Format:
            ``projects/{project}/locations/{location}``
        filter (str):
            Optional. The standard list filter. Supported fields: \*
            ``display_name`` \* ``create_time`` \* ``update_time``

            More detail in `AIP-160 <https://google.aip.dev/160>`__.
        page_size (int):
            Optional. The standard list page size.
        page_token (str):
            Optional. The standard list page token.
        order_by (str):
            Optional. A comma-separated list of fields to order by,
            sorted in ascending order. Use "desc" after a field name for
            descending. Supported fields:

            -  ``display_name``
            -  ``create_time``
            -  ``update_time``

            Example: ``display_name, create_time desc``.
    """

    parent: str = proto.Field(
        proto.STRING,
        number=1,
    )
    filter: str = proto.Field(
        proto.STRING,
        number=2,
    )
    page_size: int = proto.Field(
        proto.INT32,
        number=3,
    )
    page_token: str = proto.Field(
        proto.STRING,
        number=4,
    )
    order_by: str = proto.Field(
        proto.STRING,
        number=6,
    )


class ListExtensionsResponse(proto.Message):
    r"""Response message for
    [ExtensionRegistryService.ListExtensions][google.cloud.aiplatform.v1beta1.ExtensionRegistryService.ListExtensions]

    Attributes:
        extensions (MutableSequence[google.cloud.aiplatform_v1beta1.types.Extension]):
            List of Extension in the requested page.
        next_page_token (str):
            A token to retrieve the next page of results. Pass to
            [ListExtensionsRequest.page_token][google.cloud.aiplatform.v1beta1.ListExtensionsRequest.page_token]
            to obtain that page.
    """

    @property
    def raw_page(self):
        return self

    extensions: MutableSequence[gca_extension.Extension] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message=gca_extension.Extension,
    )
    next_page_token: str = proto.Field(
        proto.STRING,
        number=2,
    )


class DeleteExtensionRequest(proto.Message):
    r"""Request message for
    [ExtensionRegistryService.DeleteExtension][google.cloud.aiplatform.v1beta1.ExtensionRegistryService.DeleteExtension].

    Attributes:
        name (str):
            Required. The name of the Extension resource to be deleted.
            Format:
            ``projects/{project}/locations/{location}/extensions/{extension}``
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
