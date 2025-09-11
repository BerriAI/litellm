# -*- coding: utf-8 -*-

# Copyright 2021 Google LLC
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

from typing import Dict, List, Optional, Sequence, Tuple

from google.auth import credentials as auth_credentials
from google.protobuf import field_mask_pb2

from google.cloud.aiplatform import base
from google.cloud.aiplatform.compat.types import feature as gca_feature
from google.cloud.aiplatform import featurestore
from google.cloud.aiplatform import initializer
from google.cloud.aiplatform import utils
from google.cloud.aiplatform.utils import featurestore_utils

_LOGGER = base.Logger(__name__)


class Feature(base.VertexAiResourceNounWithFutureManager):
    """Managed feature resource for Vertex AI."""

    client_class = utils.FeaturestoreClientWithOverride

    _resource_noun = "features"
    _getter_method = "get_feature"
    _list_method = "list_features"
    _delete_method = "delete_feature"
    _parse_resource_name_method = "parse_feature_path"
    _format_resource_name_method = "feature_path"

    @staticmethod
    def _resource_id_validator(resource_id: str):
        """Validates resource ID.

        Args:
            resource_id(str):
                The resource id to validate.
        """
        featurestore_utils.validate_feature_id(resource_id)

    def __init__(
        self,
        feature_name: str,
        featurestore_id: Optional[str] = None,
        entity_type_id: Optional[str] = None,
        project: Optional[str] = None,
        location: Optional[str] = None,
        credentials: Optional[auth_credentials.Credentials] = None,
    ):
        """Retrieves an existing managed feature given a feature resource name or a feature ID.

        Example Usage:

            my_feature = aiplatform.Feature(
                feature_name='projects/123/locations/us-central1/featurestores/my_featurestore_id/\
                entityTypes/my_entity_type_id/features/my_feature_id'
            )
            or
            my_feature = aiplatform.Feature(
                feature_name='my_feature_id',
                featurestore_id='my_featurestore_id',
                entity_type_id='my_entity_type_id',
            )

        Args:
            feature_name (str):
                Required. A fully-qualified feature resource name or a feature ID.
                Example: "projects/123/locations/us-central1/featurestores/my_featurestore_id/entityTypes/my_entity_type_id/features/my_feature_id"
                or "my_feature_id" when project and location are initialized or passed, with featurestore_id and entity_type_id passed.
            featurestore_id (str):
                Optional. Featurestore ID of an existing featurestore to retrieve feature from,
                when feature_name is passed as Feature ID.
            entity_type_id (str):
                Optional. EntityType ID of an existing entityType to retrieve feature from,
                when feature_name is passed as Feature ID.
                The EntityType must exist in the Featurestore if provided by the featurestore_id.
            project (str):
                Optional. Project to retrieve feature from. If not set, project
                set in aiplatform.init will be used.
            location (str):
                Optional. Location to retrieve feature from. If not set, location
                set in aiplatform.init will be used.
            credentials (auth_credentials.Credentials):
                Optional. Custom credentials to use to retrieve this Feature. Overrides
                credentials set in aiplatform.init.
        Raises:
            ValueError: If only one of featurestore_id or entity_type_id is provided.
        """

        if bool(featurestore_id) != bool(entity_type_id):
            raise ValueError(
                "featurestore_id and entity_type_id must both be provided or ommitted."
            )

        super().__init__(
            project=project,
            location=location,
            credentials=credentials,
            resource_name=feature_name,
        )
        self._gca_resource = self._get_gca_resource(
            resource_name=feature_name,
            parent_resource_name_fields={
                featurestore.Featurestore._resource_noun: featurestore_id,
                featurestore.EntityType._resource_noun: entity_type_id,
            }
            if featurestore_id
            else featurestore_id,
        )

    def _get_featurestore_name(self) -> str:
        """Gets full qualified resource name of the managed featurestore in which this Feature is."""
        feature_path_components = self._parse_resource_name(self.resource_name)
        return featurestore.Featurestore._format_resource_name(
            project=feature_path_components["project"],
            location=feature_path_components["location"],
            featurestore=feature_path_components["featurestore"],
        )

    @property
    def featurestore_name(self) -> str:
        """Full qualified resource name of the managed featurestore in which this Feature is."""
        self.wait()
        return self._get_featurestore_name()

    def get_featurestore(self) -> "featurestore.Featurestore":
        """Retrieves the managed featurestore in which this Feature is.

        Returns:
            featurestore.Featurestore - The managed featurestore in which this Feature is.
        """
        return featurestore.Featurestore(featurestore_name=self.featurestore_name)

    def _get_entity_type_name(self) -> str:
        """Gets full qualified resource name of the managed entityType in which this Feature is."""
        feature_path_components = self._parse_resource_name(self.resource_name)
        return featurestore.EntityType._format_resource_name(
            project=feature_path_components["project"],
            location=feature_path_components["location"],
            featurestore=feature_path_components["featurestore"],
            entity_type=feature_path_components["entity_type"],
        )

    @property
    def entity_type_name(self) -> str:
        """Full qualified resource name of the managed entityType in which this Feature is."""
        self.wait()
        return self._get_entity_type_name()

    def get_entity_type(self) -> "featurestore.EntityType":
        """Retrieves the managed entityType in which this Feature is.

        Returns:
            featurestore.EntityType - The managed entityType in which this Feature is.
        """
        return featurestore.EntityType(entity_type_name=self.entity_type_name)

    def update(
        self,
        description: Optional[str] = None,
        labels: Optional[Dict[str, str]] = None,
        request_metadata: Optional[Sequence[Tuple[str, str]]] = (),
        update_request_timeout: Optional[float] = None,
    ) -> "Feature":
        """Updates an existing managed feature resource.

        Example Usage:

            my_feature = aiplatform.Feature(
                feature_name='my_feature_id',
                featurestore_id='my_featurestore_id',
                entity_type_id='my_entity_type_id',
            )
            my_feature.update(
                description='update my description',
            )

        Args:
            description (str):
                Optional. Description of the Feature.
            labels (Dict[str, str]):
                Optional. The labels with user-defined
                metadata to organize your Features.
                Label keys and values can be no longer than 64
                characters (Unicode codepoints), can only
                contain lowercase letters, numeric characters,
                underscores and dashes. International characters
                are allowed.
                See https://goo.gl/xmQnxf for more information
                on and examples of labels. No more than 64 user
                labels can be associated with one Feature
                (System labels are excluded)."
                System reserved label keys are prefixed with
                "aiplatform.googleapis.com/" and are immutable.
            request_metadata (Sequence[Tuple[str, str]]):
                Optional. Strings which should be sent along with the request as metadata.
            update_request_timeout (float):
                Optional. The timeout for the update request in seconds.

        Returns:
            Feature - The updated feature resource object.
        """
        self.wait()
        update_mask = list()

        if description:
            update_mask.append("description")

        if labels:
            utils.validate_labels(labels)
            update_mask.append("labels")

        update_mask = field_mask_pb2.FieldMask(paths=update_mask)

        gapic_feature = gca_feature.Feature(
            name=self.resource_name,
            description=description,
            labels=labels,
        )

        _LOGGER.log_action_start_against_resource(
            "Updating",
            "feature",
            self,
        )

        self._gca_resource = self.api_client.update_feature(
            feature=gapic_feature,
            update_mask=update_mask,
            metadata=request_metadata,
            timeout=update_request_timeout,
        )
        return self

    @classmethod
    def list(
        cls,
        entity_type_name: str,
        featurestore_id: Optional[str] = None,
        filter: Optional[str] = None,
        order_by: Optional[str] = None,
        project: Optional[str] = None,
        location: Optional[str] = None,
        credentials: Optional[auth_credentials.Credentials] = None,
    ) -> List["Feature"]:
        """Lists existing managed feature resources in an entityType, given an entityType resource name or an entity_type ID.

        Example Usage:

            my_features = aiplatform.Feature.list(
                entity_type_name='projects/123/locations/us-central1/featurestores/my_featurestore_id/\
                entityTypes/my_entity_type_id'
            )
            or
            my_features = aiplatform.Feature.list(
                entity_type_name='my_entity_type_id',
                featurestore_id='my_featurestore_id',
            )

        Args:
            entity_type_name (str):
                Required. A fully-qualified entityType resource name or an entity_type ID of an existing entityType
                to list features in. The EntityType must exist in the Featurestore if provided by the featurestore_id.
                Example: "projects/123/locations/us-central1/featurestores/my_featurestore_id/entityTypes/my_entity_type_id"
                or "my_entity_type_id" when project and location are initialized or passed, with featurestore_id passed.
            featurestore_id (str):
                Optional. Featurestore ID of an existing featurestore to list features in,
                when entity_type_name is passed as entity_type ID.
            filter (str):
                Optional. Lists the Features that match the filter expression. The
                following filters are supported:

                -  ``value_type``: Supports = and != comparisons.
                -  ``create_time``: Supports =, !=, <, >, >=, and <=
                   comparisons. Values must be in RFC 3339 format.
                -  ``update_time``: Supports =, !=, <, >, >=, and <=
                   comparisons. Values must be in RFC 3339 format.
                -  ``labels``: Supports key-value equality as well as key
                   presence.

                Examples:

                -  ``value_type = DOUBLE`` --> Features whose type is
                   DOUBLE.
                -  ``create_time > \"2020-01-31T15:30:00.000000Z\" OR update_time > \"2020-01-31T15:30:00.000000Z\"``
                   --> EntityTypes created or updated after
                   2020-01-31T15:30:00.000000Z.
                -  ``labels.active = yes AND labels.env = prod`` -->
                   Features having both (active: yes) and (env: prod)
                   labels.
                -  ``labels.env: *`` --> Any Feature which has a label with
                   'env' as the key.
            order_by (str):
                Optional. A comma-separated list of fields to order by, sorted in
                ascending order. Use "desc" after a field name for
                descending. Supported fields:

                -  ``feature_id``
                -  ``value_type``
                -  ``create_time``
                -  ``update_time``
            project (str):
                Optional. Project to list features in. If not set, project
                set in aiplatform.init will be used.
            location (str):
                Optional. Location to list features in. If not set, location
                set in aiplatform.init will be used.
            credentials (auth_credentials.Credentials):
                Optional. Custom credentials to use to list features. Overrides
                credentials set in aiplatform.init.

        Returns:
            List[Feature] - A list of managed feature resource objects
        """

        return cls._list(
            filter=filter,
            order_by=order_by,
            project=project,
            location=location,
            credentials=credentials,
            parent=utils.full_resource_name(
                resource_name=entity_type_name,
                resource_noun=featurestore.EntityType._resource_noun,
                parse_resource_name_method=featurestore.EntityType._parse_resource_name,
                format_resource_name_method=featurestore.EntityType._format_resource_name,
                parent_resource_name_fields={
                    featurestore.Featurestore._resource_noun: featurestore_id
                }
                if featurestore_id
                else featurestore_id,
                project=project,
                location=location,
                resource_id_validator=featurestore.EntityType._resource_id_validator,
            ),
        )

    @classmethod
    def search(
        cls,
        query: Optional[str] = None,
        page_size: Optional[int] = None,
        project: Optional[str] = None,
        location: Optional[str] = None,
        credentials: Optional[auth_credentials.Credentials] = None,
    ) -> List["Feature"]:
        """Searches existing managed Feature resources.

        Example Usage:

            my_features = aiplatform.Feature.search()

        Args:
            query (str):
                Optional. Query string that is a conjunction of field-restricted
                queries and/or field-restricted filters.
                Field-restricted queries and filters can be combined
                using ``AND`` to form a conjunction.

                A field query is in the form FIELD:QUERY. This
                implicitly checks if QUERY exists as a substring within
                Feature's FIELD. The QUERY and the FIELD are converted
                to a sequence of words (i.e. tokens) for comparison.
                This is done by:

                -  Removing leading/trailing whitespace and tokenizing
                   the search value. Characters that are not one of
                   alphanumeric ``[a-zA-Z0-9]``, underscore ``_``, or
                   asterisk ``*`` are treated as delimiters for tokens.
                   ``*`` is treated as a wildcard that matches
                   characters within a token.
                -  Ignoring case.
                -  Prepending an asterisk to the first and appending an
                   asterisk to the last token in QUERY.

                A QUERY must be either a singular token or a phrase. A
                phrase is one or multiple words enclosed in double
                quotation marks ("). With phrases, the order of the
                words is important. Words in the phrase must be matching
                in order and consecutively.

                Supported FIELDs for field-restricted queries:

                -  ``feature_id``
                -  ``description``
                -  ``entity_type_id``

                Examples:

                -  ``feature_id: foo`` --> Matches a Feature with ID
                   containing the substring ``foo`` (eg. ``foo``,
                   ``foofeature``, ``barfoo``).
                -  ``feature_id: foo*feature`` --> Matches a Feature
                   with ID containing the substring ``foo*feature`` (eg.
                   ``foobarfeature``).
                -  ``feature_id: foo AND description: bar`` --> Matches
                   a Feature with ID containing the substring ``foo``
                   and description containing the substring ``bar``.

                Besides field queries, the following exact-match filters
                are supported. The exact-match filters do not support
                wildcards. Unlike field-restricted queries, exact-match
                filters are case-sensitive.

                -  ``feature_id``: Supports = comparisons.
                -  ``description``: Supports = comparisons. Multi-token
                   filters should be enclosed in quotes.
                -  ``entity_type_id``: Supports = comparisons.
                -  ``value_type``: Supports = and != comparisons.
                -  ``labels``: Supports key-value equality as well as
                   key presence.
                -  ``featurestore_id``: Supports = comparisons.

                Examples:

                -  ``description = "foo bar"`` --> Any Feature with
                   description exactly equal to ``foo bar``
                -  ``value_type = DOUBLE`` --> Features whose type is
                   DOUBLE.
                -  ``labels.active = yes AND labels.env = prod`` -->
                   Features having both (active: yes) and (env: prod)
                   labels.
                -  ``labels.env: *`` --> Any Feature which has a label
                   with ``env`` as the key.

                This corresponds to the ``query`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            page_size (int):
                Optional. The maximum number of Features to return. The
                service may return fewer than this value. If
                unspecified, at most 100 Features will be
                returned. The maximum value is 100; any value
                greater than 100 will be coerced to 100.
            project (str):
                Optional. Project to list features in. If not set, project
                set in aiplatform.init will be used.
            location (str):
                Optional. Location to list features in. If not set, location
                set in aiplatform.init will be used.
            credentials (auth_credentials.Credentials):
                Optional. Custom credentials to use to list features. Overrides
                credentials set in aiplatform.init.

        Returns:
            List[Feature] - A list of managed feature resource objects
        """
        resource = cls._empty_constructor(
            project=project, location=location, credentials=credentials
        )

        # Fetch credentials once and re-use for all `_empty_constructor()` calls
        creds = resource.credentials

        search_features_request = {
            "location": initializer.global_config.common_location_path(
                project=project, location=location
            ),
            "query": query,
        }

        if page_size:
            search_features_request["page_size"] = page_size

        resource_list = (
            resource.api_client.search_features(request=search_features_request) or []
        )

        return [
            cls._construct_sdk_resource_from_gapic(
                gapic_resource, project=project, location=location, credentials=creds
            )
            for gapic_resource in resource_list
        ]

    @classmethod
    @base.optional_sync()
    def create(
        cls,
        feature_id: str,
        value_type: str,
        entity_type_name: str,
        featurestore_id: Optional[str] = None,
        description: Optional[str] = None,
        labels: Optional[Dict[str, str]] = None,
        project: Optional[str] = None,
        location: Optional[str] = None,
        credentials: Optional[auth_credentials.Credentials] = None,
        request_metadata: Optional[Sequence[Tuple[str, str]]] = (),
        sync: bool = True,
        create_request_timeout: Optional[float] = None,
    ) -> "Feature":
        """Creates a Feature resource in an EntityType.

        Example Usage:

            my_feature = aiplatform.Feature.create(
                feature_id='my_feature_id',
                value_type='INT64',
                entity_type_name='projects/123/locations/us-central1/featurestores/my_featurestore_id/\
                entityTypes/my_entity_type_id'
            )
            or
            my_feature = aiplatform.Feature.create(
                feature_id='my_feature_id',
                value_type='INT64',
                entity_type_name='my_entity_type_id',
                featurestore_id='my_featurestore_id',
            )

        Args:
            feature_id (str):
                Required. The ID to use for the Feature, which will become
                the final component of the Feature's resource name, which is immutable.

                This value may be up to 60 characters, and valid characters
                are ``[a-z0-9_]``. The first character cannot be a number.

                The value must be unique within an EntityType.
            value_type (str):
                Required. Immutable. Type of Feature value.
                One of BOOL, BOOL_ARRAY, DOUBLE, DOUBLE_ARRAY, INT64, INT64_ARRAY, STRING, STRING_ARRAY, BYTES.
            entity_type_name (str):
                Required. A fully-qualified entityType resource name or an entity_type ID of an existing entityType
                to create Feature in. The EntityType must exist in the Featurestore if provided by the featurestore_id.
                Example: "projects/123/locations/us-central1/featurestores/my_featurestore_id/entityTypes/my_entity_type_id"
                or "my_entity_type_id" when project and location are initialized or passed, with featurestore_id passed.
            featurestore_id (str):
                Optional. Featurestore ID of an existing featurestore to create Feature in
                if `entity_type_name` is passed an entity_type ID.
            description (str):
                Optional. Description of the Feature.
            labels (Dict[str, str]):
                Optional. The labels with user-defined
                metadata to organize your Features.
                Label keys and values can be no longer than 64
                characters (Unicode codepoints), can only
                contain lowercase letters, numeric characters,
                underscores and dashes. International characters
                are allowed.
                See https://goo.gl/xmQnxf for more information
                on and examples of labels. No more than 64 user
                labels can be associated with one Feature
                (System labels are excluded)."
                System reserved label keys are prefixed with
                "aiplatform.googleapis.com/" and are immutable.
            project (str):
                Optional. Project to create Feature in if `entity_type_name` is passed an entity_type ID.
                If not set, project set in aiplatform.init will be used.
            location (str):
                Optional. Location to create Feature in if `entity_type_name` is passed an entity_type ID.
                If not set, location set in aiplatform.init will be used.
            credentials (auth_credentials.Credentials):
                Optional. Custom credentials to use to create Features. Overrides
                credentials set in aiplatform.init.
            request_metadata (Sequence[Tuple[str, str]]):
                Optional. Strings which should be sent along with the request as metadata.
            sync (bool):
                Optional. Whether to execute this creation synchronously. If False, this method
                will be executed in concurrent Future and any downstream object will
                be immediately returned and synced when the Future has completed.
            create_request_timeout (float):
                Optional. The timeout for the create request in seconds.

        Returns:
            Feature - feature resource object

        """
        entity_type_name = utils.full_resource_name(
            resource_name=entity_type_name,
            resource_noun=featurestore.EntityType._resource_noun,
            parse_resource_name_method=featurestore.EntityType._parse_resource_name,
            format_resource_name_method=featurestore.EntityType._format_resource_name,
            parent_resource_name_fields={
                featurestore.Featurestore._resource_noun: featurestore_id
            }
            if featurestore_id
            else featurestore_id,
            project=project,
            location=location,
            resource_id_validator=featurestore.EntityType._resource_id_validator,
        )
        entity_type_name_components = featurestore.EntityType._parse_resource_name(
            entity_type_name
        )

        feature_config = featurestore_utils._FeatureConfig(
            feature_id=feature_id,
            value_type=value_type,
            description=description,
            labels=labels,
        )

        create_feature_request = feature_config.get_create_feature_request()
        create_feature_request.parent = entity_type_name

        api_client = cls._instantiate_client(
            location=entity_type_name_components["location"],
            credentials=credentials,
        )

        created_feature_lro = api_client.create_feature(
            request=create_feature_request,
            metadata=request_metadata,
            timeout=create_request_timeout,
        )

        _LOGGER.log_create_with_lro(cls, created_feature_lro)

        created_feature = created_feature_lro.result()

        _LOGGER.log_create_complete(cls, created_feature, "feature")

        feature_obj = cls(
            feature_name=created_feature.name,
            project=project,
            location=location,
            credentials=credentials,
        )

        return feature_obj
