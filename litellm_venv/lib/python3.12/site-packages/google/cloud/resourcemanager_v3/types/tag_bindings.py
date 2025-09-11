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

__protobuf__ = proto.module(
    package="google.cloud.resourcemanager.v3",
    manifest={
        "TagBinding",
        "CreateTagBindingMetadata",
        "CreateTagBindingRequest",
        "DeleteTagBindingMetadata",
        "DeleteTagBindingRequest",
        "ListTagBindingsRequest",
        "ListTagBindingsResponse",
        "ListEffectiveTagsRequest",
        "ListEffectiveTagsResponse",
        "EffectiveTag",
    },
)


class TagBinding(proto.Message):
    r"""A TagBinding represents a connection between a TagValue and a
    cloud resource Once a TagBinding is created, the TagValue is
    applied to all the descendants of the Google Cloud resource.

    Attributes:
        name (str):
            Output only. The name of the TagBinding. This is a String of
            the form:
            ``tagBindings/{full-resource-name}/{tag-value-name}`` (e.g.
            ``tagBindings/%2F%2Fcloudresourcemanager.googleapis.com%2Fprojects%2F123/tagValues/456``).
        parent (str):
            The full resource name of the resource the TagValue is bound
            to. E.g.
            ``//cloudresourcemanager.googleapis.com/projects/123``
        tag_value (str):
            The TagValue of the TagBinding. Must be of the form
            ``tagValues/456``.
        tag_value_namespaced_name (str):
            The namespaced name for the TagValue of the TagBinding. Must
            be in the format
            ``{parent_id}/{tag_key_short_name}/{short_name}``.

            For methods that support TagValue namespaced name, only one
            of tag_value_namespaced_name or tag_value may be filled.
            Requests with both fields will be rejected.
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    parent: str = proto.Field(
        proto.STRING,
        number=2,
    )
    tag_value: str = proto.Field(
        proto.STRING,
        number=3,
    )
    tag_value_namespaced_name: str = proto.Field(
        proto.STRING,
        number=4,
    )


class CreateTagBindingMetadata(proto.Message):
    r"""Runtime operation information for creating a TagValue."""


class CreateTagBindingRequest(proto.Message):
    r"""The request message to create a TagBinding.

    Attributes:
        tag_binding (google.cloud.resourcemanager_v3.types.TagBinding):
            Required. The TagBinding to be created.
        validate_only (bool):
            Optional. Set to true to perform the
            validations necessary for creating the resource,
            but not actually perform the action.
    """

    tag_binding: "TagBinding" = proto.Field(
        proto.MESSAGE,
        number=1,
        message="TagBinding",
    )
    validate_only: bool = proto.Field(
        proto.BOOL,
        number=2,
    )


class DeleteTagBindingMetadata(proto.Message):
    r"""Runtime operation information for deleting a TagBinding."""


class DeleteTagBindingRequest(proto.Message):
    r"""The request message to delete a TagBinding.

    Attributes:
        name (str):
            Required. The name of the TagBinding. This is a String of
            the form: ``tagBindings/{id}`` (e.g.
            ``tagBindings/%2F%2Fcloudresourcemanager.googleapis.com%2Fprojects%2F123/tagValues/456``).
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


class ListTagBindingsRequest(proto.Message):
    r"""The request message to list all TagBindings for a parent.

    Attributes:
        parent (str):
            Required. The full resource name of a
            resource for which you want to list existing
            TagBindings. E.g.
            "//cloudresourcemanager.googleapis.com/projects/123".
        page_size (int):
            Optional. The maximum number of TagBindings
            to return in the response. The server allows a
            maximum of 300 TagBindings to return. If
            unspecified, the server will use 100 as the
            default.
        page_token (str):
            Optional. A pagination token returned from a previous call
            to ``ListTagBindings`` that indicates where this listing
            should continue from.
    """

    parent: str = proto.Field(
        proto.STRING,
        number=1,
    )
    page_size: int = proto.Field(
        proto.INT32,
        number=2,
    )
    page_token: str = proto.Field(
        proto.STRING,
        number=3,
    )


class ListTagBindingsResponse(proto.Message):
    r"""The ListTagBindings response.

    Attributes:
        tag_bindings (MutableSequence[google.cloud.resourcemanager_v3.types.TagBinding]):
            A possibly paginated list of TagBindings for
            the specified resource.
        next_page_token (str):
            Pagination token.

            If the result set is too large to fit in a single response,
            this token is returned. It encodes the position of the
            current result cursor. Feeding this value into a new list
            request with the ``page_token`` parameter gives the next
            page of the results.

            When ``next_page_token`` is not filled in, there is no next
            page and the list returned is the last page in the result
            set.

            Pagination tokens have a limited lifetime.
    """

    @property
    def raw_page(self):
        return self

    tag_bindings: MutableSequence["TagBinding"] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message="TagBinding",
    )
    next_page_token: str = proto.Field(
        proto.STRING,
        number=2,
    )


class ListEffectiveTagsRequest(proto.Message):
    r"""The request message to ListEffectiveTags

    Attributes:
        parent (str):
            Required. The full resource name of a
            resource for which you want to list the
            effective tags. E.g.
            "//cloudresourcemanager.googleapis.com/projects/123".
        page_size (int):
            Optional. The maximum number of effective
            tags to return in the response. The server
            allows a maximum of 300 effective tags to return
            in a single page. If unspecified, the server
            will use 100 as the default.
        page_token (str):
            Optional. A pagination token returned from a previous call
            to ``ListEffectiveTags`` that indicates from where this
            listing should continue.
    """

    parent: str = proto.Field(
        proto.STRING,
        number=1,
    )
    page_size: int = proto.Field(
        proto.INT32,
        number=2,
    )
    page_token: str = proto.Field(
        proto.STRING,
        number=3,
    )


class ListEffectiveTagsResponse(proto.Message):
    r"""The response of ListEffectiveTags.

    Attributes:
        effective_tags (MutableSequence[google.cloud.resourcemanager_v3.types.EffectiveTag]):
            A possibly paginated list of effective tags
            for the specified resource.
        next_page_token (str):
            Pagination token.

            If the result set is too large to fit in a single response,
            this token is returned. It encodes the position of the
            current result cursor. Feeding this value into a new list
            request with the ``page_token`` parameter gives the next
            page of the results.

            When ``next_page_token`` is not filled in, there is no next
            page and the list returned is the last page in the result
            set.

            Pagination tokens have a limited lifetime.
    """

    @property
    def raw_page(self):
        return self

    effective_tags: MutableSequence["EffectiveTag"] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message="EffectiveTag",
    )
    next_page_token: str = proto.Field(
        proto.STRING,
        number=2,
    )


class EffectiveTag(proto.Message):
    r"""An EffectiveTag represents a tag that applies to a resource during
    policy evaluation. Tags can be either directly bound to a resource
    or inherited from its ancestor. EffectiveTag contains the name and
    namespaced_name of the tag value and tag key, with additional fields
    of ``inherited`` to indicate the inheritance status of the effective
    tag.

    Attributes:
        tag_value (str):
            Resource name for TagValue in the format ``tagValues/456``.
        namespaced_tag_value (str):
            The namespaced name of the TagValue. Can be in the form
            ``{organization_id}/{tag_key_short_name}/{tag_value_short_name}``
            or
            ``{project_id}/{tag_key_short_name}/{tag_value_short_name}``
            or
            ``{project_number}/{tag_key_short_name}/{tag_value_short_name}``.
        tag_key (str):
            The name of the TagKey, in the format ``tagKeys/{id}``, such
            as ``tagKeys/123``.
        namespaced_tag_key (str):
            The namespaced name of the TagKey. Can be in the form
            ``{organization_id}/{tag_key_short_name}`` or
            ``{project_id}/{tag_key_short_name}`` or
            ``{project_number}/{tag_key_short_name}``.
        tag_key_parent_name (str):
            The parent name of the tag key. Must be in the format
            ``organizations/{organization_id}`` or
            ``projects/{project_number}``
        inherited (bool):
            Indicates the inheritance status of a tag
            value attached to the given resource. If the tag
            value is inherited from one of the resource's
            ancestors, inherited will be true. If false,
            then the tag value is directly attached to the
            resource, inherited will be false.
    """

    tag_value: str = proto.Field(
        proto.STRING,
        number=1,
    )
    namespaced_tag_value: str = proto.Field(
        proto.STRING,
        number=2,
    )
    tag_key: str = proto.Field(
        proto.STRING,
        number=3,
    )
    namespaced_tag_key: str = proto.Field(
        proto.STRING,
        number=4,
    )
    tag_key_parent_name: str = proto.Field(
        proto.STRING,
        number=6,
    )
    inherited: bool = proto.Field(
        proto.BOOL,
        number=5,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
