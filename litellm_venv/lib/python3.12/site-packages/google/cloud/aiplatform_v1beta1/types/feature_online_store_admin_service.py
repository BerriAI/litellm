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

from google.cloud.aiplatform_v1beta1.types import (
    feature_online_store as gca_feature_online_store,
)
from google.cloud.aiplatform_v1beta1.types import feature_view as gca_feature_view
from google.cloud.aiplatform_v1beta1.types import (
    feature_view_sync as gca_feature_view_sync,
)
from google.cloud.aiplatform_v1beta1.types import operation
from google.protobuf import field_mask_pb2  # type: ignore


__protobuf__ = proto.module(
    package="google.cloud.aiplatform.v1beta1",
    manifest={
        "CreateFeatureOnlineStoreRequest",
        "GetFeatureOnlineStoreRequest",
        "ListFeatureOnlineStoresRequest",
        "ListFeatureOnlineStoresResponse",
        "UpdateFeatureOnlineStoreRequest",
        "DeleteFeatureOnlineStoreRequest",
        "CreateFeatureViewRequest",
        "GetFeatureViewRequest",
        "ListFeatureViewsRequest",
        "ListFeatureViewsResponse",
        "UpdateFeatureViewRequest",
        "DeleteFeatureViewRequest",
        "CreateFeatureOnlineStoreOperationMetadata",
        "UpdateFeatureOnlineStoreOperationMetadata",
        "CreateFeatureViewOperationMetadata",
        "UpdateFeatureViewOperationMetadata",
        "SyncFeatureViewRequest",
        "SyncFeatureViewResponse",
        "GetFeatureViewSyncRequest",
        "ListFeatureViewSyncsRequest",
        "ListFeatureViewSyncsResponse",
    },
)


class CreateFeatureOnlineStoreRequest(proto.Message):
    r"""Request message for
    [FeatureOnlineStoreAdminService.CreateFeatureOnlineStore][google.cloud.aiplatform.v1beta1.FeatureOnlineStoreAdminService.CreateFeatureOnlineStore].

    Attributes:
        parent (str):
            Required. The resource name of the Location to create
            FeatureOnlineStores. Format:
            ``projects/{project}/locations/{location}``
        feature_online_store (google.cloud.aiplatform_v1beta1.types.FeatureOnlineStore):
            Required. The FeatureOnlineStore to create.
        feature_online_store_id (str):
            Required. The ID to use for this FeatureOnlineStore, which
            will become the final component of the FeatureOnlineStore's
            resource name.

            This value may be up to 60 characters, and valid characters
            are ``[a-z0-9_]``. The first character cannot be a number.

            The value must be unique within the project and location.
    """

    parent: str = proto.Field(
        proto.STRING,
        number=1,
    )
    feature_online_store: gca_feature_online_store.FeatureOnlineStore = proto.Field(
        proto.MESSAGE,
        number=2,
        message=gca_feature_online_store.FeatureOnlineStore,
    )
    feature_online_store_id: str = proto.Field(
        proto.STRING,
        number=3,
    )


class GetFeatureOnlineStoreRequest(proto.Message):
    r"""Request message for
    [FeatureOnlineStoreAdminService.GetFeatureOnlineStore][google.cloud.aiplatform.v1beta1.FeatureOnlineStoreAdminService.GetFeatureOnlineStore].

    Attributes:
        name (str):
            Required. The name of the FeatureOnlineStore
            resource.
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


class ListFeatureOnlineStoresRequest(proto.Message):
    r"""Request message for
    [FeatureOnlineStoreAdminService.ListFeatureOnlineStores][google.cloud.aiplatform.v1beta1.FeatureOnlineStoreAdminService.ListFeatureOnlineStores].

    Attributes:
        parent (str):
            Required. The resource name of the Location to list
            FeatureOnlineStores. Format:
            ``projects/{project}/locations/{location}``
        filter (str):
            Lists the FeatureOnlineStores that match the filter
            expression. The following fields are supported:

            -  ``create_time``: Supports ``=``, ``!=``, ``<``, ``>``,
               ``<=``, and ``>=`` comparisons. Values must be in RFC
               3339 format.
            -  ``update_time``: Supports ``=``, ``!=``, ``<``, ``>``,
               ``<=``, and ``>=`` comparisons. Values must be in RFC
               3339 format.
            -  ``labels``: Supports key-value equality and key presence.

            Examples:

            -  ``create_time > "2020-01-01" OR update_time > "2020-01-01"``
               FeatureOnlineStores created or updated after 2020-01-01.
            -  ``labels.env = "prod"`` FeatureOnlineStores with label
               "env" set to "prod".
        page_size (int):
            The maximum number of FeatureOnlineStores to
            return. The service may return fewer than this
            value. If unspecified, at most 100
            FeatureOnlineStores will be returned. The
            maximum value is 100; any value greater than 100
            will be coerced to 100.
        page_token (str):
            A page token, received from a previous
            [FeatureOnlineStoreAdminService.ListFeatureOnlineStores][google.cloud.aiplatform.v1beta1.FeatureOnlineStoreAdminService.ListFeatureOnlineStores]
            call. Provide this to retrieve the subsequent page.

            When paginating, all other parameters provided to
            [FeatureOnlineStoreAdminService.ListFeatureOnlineStores][google.cloud.aiplatform.v1beta1.FeatureOnlineStoreAdminService.ListFeatureOnlineStores]
            must match the call that provided the page token.
        order_by (str):
            A comma-separated list of fields to order by, sorted in
            ascending order. Use "desc" after a field name for
            descending. Supported Fields:

            -  ``create_time``
            -  ``update_time``
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
        number=5,
    )


class ListFeatureOnlineStoresResponse(proto.Message):
    r"""Response message for
    [FeatureOnlineStoreAdminService.ListFeatureOnlineStores][google.cloud.aiplatform.v1beta1.FeatureOnlineStoreAdminService.ListFeatureOnlineStores].

    Attributes:
        feature_online_stores (MutableSequence[google.cloud.aiplatform_v1beta1.types.FeatureOnlineStore]):
            The FeatureOnlineStores matching the request.
        next_page_token (str):
            A token, which can be sent as
            [ListFeatureOnlineStoresRequest.page_token][google.cloud.aiplatform.v1beta1.ListFeatureOnlineStoresRequest.page_token]
            to retrieve the next page. If this field is omitted, there
            are no subsequent pages.
    """

    @property
    def raw_page(self):
        return self

    feature_online_stores: MutableSequence[
        gca_feature_online_store.FeatureOnlineStore
    ] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message=gca_feature_online_store.FeatureOnlineStore,
    )
    next_page_token: str = proto.Field(
        proto.STRING,
        number=2,
    )


class UpdateFeatureOnlineStoreRequest(proto.Message):
    r"""Request message for
    [FeatureOnlineStoreAdminService.UpdateFeatureOnlineStore][google.cloud.aiplatform.v1beta1.FeatureOnlineStoreAdminService.UpdateFeatureOnlineStore].

    Attributes:
        feature_online_store (google.cloud.aiplatform_v1beta1.types.FeatureOnlineStore):
            Required. The FeatureOnlineStore's ``name`` field is used to
            identify the FeatureOnlineStore to be updated. Format:
            ``projects/{project}/locations/{location}/featureOnlineStores/{feature_online_store}``
        update_mask (google.protobuf.field_mask_pb2.FieldMask):
            Field mask is used to specify the fields to be overwritten
            in the FeatureOnlineStore resource by the update. The fields
            specified in the update_mask are relative to the resource,
            not the full request. A field will be overwritten if it is
            in the mask. If the user does not provide a mask then only
            the non-empty fields present in the request will be
            overwritten. Set the update_mask to ``*`` to override all
            fields.

            Updatable fields:

            -  ``big_query_source``
            -  ``bigtable``
            -  ``labels``
            -  ``sync_config``
    """

    feature_online_store: gca_feature_online_store.FeatureOnlineStore = proto.Field(
        proto.MESSAGE,
        number=1,
        message=gca_feature_online_store.FeatureOnlineStore,
    )
    update_mask: field_mask_pb2.FieldMask = proto.Field(
        proto.MESSAGE,
        number=2,
        message=field_mask_pb2.FieldMask,
    )


class DeleteFeatureOnlineStoreRequest(proto.Message):
    r"""Request message for
    [FeatureOnlineStoreAdminService.DeleteFeatureOnlineStore][google.cloud.aiplatform.v1beta1.FeatureOnlineStoreAdminService.DeleteFeatureOnlineStore].

    Attributes:
        name (str):
            Required. The name of the FeatureOnlineStore to be deleted.
            Format:
            ``projects/{project}/locations/{location}/featureOnlineStores/{feature_online_store}``
        force (bool):
            If set to true, any FeatureViews and Features
            for this FeatureOnlineStore will also be
            deleted. (Otherwise, the request will only work
            if the FeatureOnlineStore has no FeatureViews.)
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    force: bool = proto.Field(
        proto.BOOL,
        number=2,
    )


class CreateFeatureViewRequest(proto.Message):
    r"""Request message for
    [FeatureOnlineStoreAdminService.CreateFeatureView][google.cloud.aiplatform.v1beta1.FeatureOnlineStoreAdminService.CreateFeatureView].

    Attributes:
        parent (str):
            Required. The resource name of the FeatureOnlineStore to
            create FeatureViews. Format:
            ``projects/{project}/locations/{location}/featureOnlineStores/{feature_online_store}``
        feature_view (google.cloud.aiplatform_v1beta1.types.FeatureView):
            Required. The FeatureView to create.
        feature_view_id (str):
            Required. The ID to use for the FeatureView, which will
            become the final component of the FeatureView's resource
            name.

            This value may be up to 60 characters, and valid characters
            are ``[a-z0-9_]``. The first character cannot be a number.

            The value must be unique within a FeatureOnlineStore.
        run_sync_immediately (bool):
            Immutable. If set to true, one on demand sync will be run
            immediately, regardless whether the
            [FeatureView.sync_config][google.cloud.aiplatform.v1beta1.FeatureView.sync_config]
            is configured or not.
    """

    parent: str = proto.Field(
        proto.STRING,
        number=1,
    )
    feature_view: gca_feature_view.FeatureView = proto.Field(
        proto.MESSAGE,
        number=2,
        message=gca_feature_view.FeatureView,
    )
    feature_view_id: str = proto.Field(
        proto.STRING,
        number=3,
    )
    run_sync_immediately: bool = proto.Field(
        proto.BOOL,
        number=4,
    )


class GetFeatureViewRequest(proto.Message):
    r"""Request message for
    [FeatureOnlineStoreAdminService.GetFeatureView][google.cloud.aiplatform.v1beta1.FeatureOnlineStoreAdminService.GetFeatureView].

    Attributes:
        name (str):
            Required. The name of the FeatureView resource. Format:
            ``projects/{project}/locations/{location}/featureOnlineStores/{feature_online_store}/featureViews/{feature_view}``
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


class ListFeatureViewsRequest(proto.Message):
    r"""Request message for
    [FeatureOnlineStoreAdminService.ListFeatureViews][google.cloud.aiplatform.v1beta1.FeatureOnlineStoreAdminService.ListFeatureViews].

    Attributes:
        parent (str):
            Required. The resource name of the FeatureOnlineStore to
            list FeatureViews. Format:
            ``projects/{project}/locations/{location}/featureOnlineStores/{feature_online_store}``
        filter (str):
            Lists the FeatureViews that match the filter expression. The
            following filters are supported:

            -  ``create_time``: Supports ``=``, ``!=``, ``<``, ``>``,
               ``>=``, and ``<=`` comparisons. Values must be in RFC
               3339 format.
            -  ``update_time``: Supports ``=``, ``!=``, ``<``, ``>``,
               ``>=``, and ``<=`` comparisons. Values must be in RFC
               3339 format.
            -  ``labels``: Supports key-value equality as well as key
               presence.

            Examples:

            -  ``create_time > \"2020-01-31T15:30:00.000000Z\" OR update_time > \"2020-01-31T15:30:00.000000Z\"``
               --> FeatureViews created or updated after
               2020-01-31T15:30:00.000000Z.
            -  ``labels.active = yes AND labels.env = prod`` -->
               FeatureViews having both (active: yes) and (env: prod)
               labels.
            -  ``labels.env: *`` --> Any FeatureView which has a label
               with 'env' as the key.
        page_size (int):
            The maximum number of FeatureViews to return.
            The service may return fewer than this value. If
            unspecified, at most 1000 FeatureViews will be
            returned. The maximum value is 1000; any value
            greater than 1000 will be coerced to 1000.
        page_token (str):
            A page token, received from a previous
            [FeatureOnlineStoreAdminService.ListFeatureViews][google.cloud.aiplatform.v1beta1.FeatureOnlineStoreAdminService.ListFeatureViews]
            call. Provide this to retrieve the subsequent page.

            When paginating, all other parameters provided to
            [FeatureOnlineStoreAdminService.ListFeatureViews][google.cloud.aiplatform.v1beta1.FeatureOnlineStoreAdminService.ListFeatureViews]
            must match the call that provided the page token.
        order_by (str):
            A comma-separated list of fields to order by, sorted in
            ascending order. Use "desc" after a field name for
            descending.

            Supported fields:

            -  ``feature_view_id``
            -  ``create_time``
            -  ``update_time``
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
        number=5,
    )


class ListFeatureViewsResponse(proto.Message):
    r"""Response message for
    [FeatureOnlineStoreAdminService.ListFeatureViews][google.cloud.aiplatform.v1beta1.FeatureOnlineStoreAdminService.ListFeatureViews].

    Attributes:
        feature_views (MutableSequence[google.cloud.aiplatform_v1beta1.types.FeatureView]):
            The FeatureViews matching the request.
        next_page_token (str):
            A token, which can be sent as
            [ListFeatureViewsRequest.page_token][google.cloud.aiplatform.v1beta1.ListFeatureViewsRequest.page_token]
            to retrieve the next page. If this field is omitted, there
            are no subsequent pages.
    """

    @property
    def raw_page(self):
        return self

    feature_views: MutableSequence[gca_feature_view.FeatureView] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message=gca_feature_view.FeatureView,
    )
    next_page_token: str = proto.Field(
        proto.STRING,
        number=2,
    )


class UpdateFeatureViewRequest(proto.Message):
    r"""Request message for
    [FeatureOnlineStoreAdminService.UpdateFeatureView][google.cloud.aiplatform.v1beta1.FeatureOnlineStoreAdminService.UpdateFeatureView].

    Attributes:
        feature_view (google.cloud.aiplatform_v1beta1.types.FeatureView):
            Required. The FeatureView's ``name`` field is used to
            identify the FeatureView to be updated. Format:
            ``projects/{project}/locations/{location}/featureOnlineStores/{feature_online_store}/featureViews/{feature_view}``
        update_mask (google.protobuf.field_mask_pb2.FieldMask):
            Field mask is used to specify the fields to be overwritten
            in the FeatureView resource by the update. The fields
            specified in the update_mask are relative to the resource,
            not the full request. A field will be overwritten if it is
            in the mask. If the user does not provide a mask then only
            the non-empty fields present in the request will be
            overwritten. Set the update_mask to ``*`` to override all
            fields.

            Updatable fields:

            -  ``labels``
            -  ``serviceAgentType``
    """

    feature_view: gca_feature_view.FeatureView = proto.Field(
        proto.MESSAGE,
        number=1,
        message=gca_feature_view.FeatureView,
    )
    update_mask: field_mask_pb2.FieldMask = proto.Field(
        proto.MESSAGE,
        number=2,
        message=field_mask_pb2.FieldMask,
    )


class DeleteFeatureViewRequest(proto.Message):
    r"""Request message for
    [FeatureOnlineStoreAdminService.DeleteFeatureViews][].

    Attributes:
        name (str):
            Required. The name of the FeatureView to be deleted. Format:
            ``projects/{project}/locations/{location}/featureOnlineStores/{feature_online_store}/featureViews/{feature_view}``
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


class CreateFeatureOnlineStoreOperationMetadata(proto.Message):
    r"""Details of operations that perform create FeatureOnlineStore.

    Attributes:
        generic_metadata (google.cloud.aiplatform_v1beta1.types.GenericOperationMetadata):
            Operation metadata for FeatureOnlineStore.
    """

    generic_metadata: operation.GenericOperationMetadata = proto.Field(
        proto.MESSAGE,
        number=1,
        message=operation.GenericOperationMetadata,
    )


class UpdateFeatureOnlineStoreOperationMetadata(proto.Message):
    r"""Details of operations that perform update FeatureOnlineStore.

    Attributes:
        generic_metadata (google.cloud.aiplatform_v1beta1.types.GenericOperationMetadata):
            Operation metadata for FeatureOnlineStore.
    """

    generic_metadata: operation.GenericOperationMetadata = proto.Field(
        proto.MESSAGE,
        number=1,
        message=operation.GenericOperationMetadata,
    )


class CreateFeatureViewOperationMetadata(proto.Message):
    r"""Details of operations that perform create FeatureView.

    Attributes:
        generic_metadata (google.cloud.aiplatform_v1beta1.types.GenericOperationMetadata):
            Operation metadata for FeatureView Create.
    """

    generic_metadata: operation.GenericOperationMetadata = proto.Field(
        proto.MESSAGE,
        number=1,
        message=operation.GenericOperationMetadata,
    )


class UpdateFeatureViewOperationMetadata(proto.Message):
    r"""Details of operations that perform update FeatureView.

    Attributes:
        generic_metadata (google.cloud.aiplatform_v1beta1.types.GenericOperationMetadata):
            Operation metadata for FeatureView Update.
    """

    generic_metadata: operation.GenericOperationMetadata = proto.Field(
        proto.MESSAGE,
        number=1,
        message=operation.GenericOperationMetadata,
    )


class SyncFeatureViewRequest(proto.Message):
    r"""Request message for
    [FeatureOnlineStoreAdminService.SyncFeatureView][google.cloud.aiplatform.v1beta1.FeatureOnlineStoreAdminService.SyncFeatureView].

    Attributes:
        feature_view (str):
            Required. Format:
            ``projects/{project}/locations/{location}/featureOnlineStores/{feature_online_store}/featureViews/{feature_view}``
    """

    feature_view: str = proto.Field(
        proto.STRING,
        number=1,
    )


class SyncFeatureViewResponse(proto.Message):
    r"""Respose message for
    [FeatureOnlineStoreAdminService.SyncFeatureView][google.cloud.aiplatform.v1beta1.FeatureOnlineStoreAdminService.SyncFeatureView].

    Attributes:
        feature_view_sync (str):
            Format:
            ``projects/{project}/locations/{location}/featureOnlineStores/{feature_online_store}/featureViews/{feature_view}/featureViewSyncs/{feature_view_sync}``
    """

    feature_view_sync: str = proto.Field(
        proto.STRING,
        number=1,
    )


class GetFeatureViewSyncRequest(proto.Message):
    r"""Request message for
    [FeatureOnlineStoreAdminService.GetFeatureViewSync][google.cloud.aiplatform.v1beta1.FeatureOnlineStoreAdminService.GetFeatureViewSync].

    Attributes:
        name (str):
            Required. The name of the FeatureViewSync resource. Format:
            ``projects/{project}/locations/{location}/featureOnlineStores/{feature_online_store}/featureViews/{feature_view}/featureViewSyncs/{feature_view_sync}``
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


class ListFeatureViewSyncsRequest(proto.Message):
    r"""Request message for
    [FeatureOnlineStoreAdminService.ListFeatureViewSyncs][google.cloud.aiplatform.v1beta1.FeatureOnlineStoreAdminService.ListFeatureViewSyncs].

    Attributes:
        parent (str):
            Required. The resource name of the FeatureView to list
            FeatureViewSyncs. Format:
            ``projects/{project}/locations/{location}/featureOnlineStores/{feature_online_store}/featureViews/{feature_view}``
        filter (str):
            Lists the FeatureViewSyncs that match the filter expression.
            The following filters are supported:

            -  ``create_time``: Supports ``=``, ``!=``, ``<``, ``>``,
               ``>=``, and ``<=`` comparisons. Values must be in RFC
               3339 format.

            Examples:

            -  ``create_time > \"2020-01-31T15:30:00.000000Z\"`` -->
               FeatureViewSyncs created after
               2020-01-31T15:30:00.000000Z.
        page_size (int):
            The maximum number of FeatureViewSyncs to
            return. The service may return fewer than this
            value. If unspecified, at most 1000
            FeatureViewSyncs will be returned. The maximum
            value is 1000; any value greater than 1000 will
            be coerced to 1000.
        page_token (str):
            A page token, received from a previous
            [FeatureOnlineStoreAdminService.ListFeatureViewSyncs][google.cloud.aiplatform.v1beta1.FeatureOnlineStoreAdminService.ListFeatureViewSyncs]
            call. Provide this to retrieve the subsequent page.

            When paginating, all other parameters provided to
            [FeatureOnlineStoreAdminService.ListFeatureViewSyncs][google.cloud.aiplatform.v1beta1.FeatureOnlineStoreAdminService.ListFeatureViewSyncs]
            must match the call that provided the page token.
        order_by (str):
            A comma-separated list of fields to order by, sorted in
            ascending order. Use "desc" after a field name for
            descending.

            Supported fields:

            -  ``create_time``
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
        number=5,
    )


class ListFeatureViewSyncsResponse(proto.Message):
    r"""Response message for
    [FeatureOnlineStoreAdminService.ListFeatureViewSyncs][google.cloud.aiplatform.v1beta1.FeatureOnlineStoreAdminService.ListFeatureViewSyncs].

    Attributes:
        feature_view_syncs (MutableSequence[google.cloud.aiplatform_v1beta1.types.FeatureViewSync]):
            The FeatureViewSyncs matching the request.
        next_page_token (str):
            A token, which can be sent as
            [ListFeatureViewSyncsRequest.page_token][google.cloud.aiplatform.v1beta1.ListFeatureViewSyncsRequest.page_token]
            to retrieve the next page. If this field is omitted, there
            are no subsequent pages.
    """

    @property
    def raw_page(self):
        return self

    feature_view_syncs: MutableSequence[
        gca_feature_view_sync.FeatureViewSync
    ] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message=gca_feature_view_sync.FeatureViewSync,
    )
    next_page_token: str = proto.Field(
        proto.STRING,
        number=2,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
