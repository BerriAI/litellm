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

from google.cloud.aiplatform_v1.types import machine_resources
from google.cloud.aiplatform_v1.types import model


__protobuf__ = proto.module(
    package="google.cloud.aiplatform.v1",
    manifest={
        "PublisherModel",
    },
)


class PublisherModel(proto.Message):
    r"""A Model Garden Publisher Model.

    Attributes:
        name (str):
            Output only. The resource name of the
            PublisherModel.
        version_id (str):
            Output only. Immutable. The version ID of the
            PublisherModel. A new version is committed when
            a new model version is uploaded under an
            existing model id. It is an auto-incrementing
            decimal number in string representation.
        open_source_category (google.cloud.aiplatform_v1.types.PublisherModel.OpenSourceCategory):
            Required. Indicates the open source category
            of the publisher model.
        supported_actions (google.cloud.aiplatform_v1.types.PublisherModel.CallToAction):
            Optional. Supported call-to-action options.
        frameworks (MutableSequence[str]):
            Optional. Additional information about the
            model's Frameworks.
        launch_stage (google.cloud.aiplatform_v1.types.PublisherModel.LaunchStage):
            Optional. Indicates the launch stage of the
            model.
        version_state (google.cloud.aiplatform_v1.types.PublisherModel.VersionState):
            Optional. Indicates the state of the model
            version.
        publisher_model_template (str):
            Optional. Output only. Immutable. Used to
            indicate this model has a publisher model and
            provide the template of the publisher model
            resource name.
        predict_schemata (google.cloud.aiplatform_v1.types.PredictSchemata):
            Optional. The schemata that describes formats of the
            PublisherModel's predictions and explanations as given and
            returned via
            [PredictionService.Predict][google.cloud.aiplatform.v1.PredictionService.Predict].
    """

    class OpenSourceCategory(proto.Enum):
        r"""An enum representing the open source category of a
        PublisherModel.

        Values:
            OPEN_SOURCE_CATEGORY_UNSPECIFIED (0):
                The open source category is unspecified,
                which should not be used.
            PROPRIETARY (1):
                Used to indicate the PublisherModel is not
                open sourced.
            GOOGLE_OWNED_OSS_WITH_GOOGLE_CHECKPOINT (2):
                Used to indicate the PublisherModel is a
                Google-owned open source model w/ Google
                checkpoint.
            THIRD_PARTY_OWNED_OSS_WITH_GOOGLE_CHECKPOINT (3):
                Used to indicate the PublisherModel is a
                3p-owned open source model w/ Google checkpoint.
            GOOGLE_OWNED_OSS (4):
                Used to indicate the PublisherModel is a
                Google-owned pure open source model.
            THIRD_PARTY_OWNED_OSS (5):
                Used to indicate the PublisherModel is a
                3p-owned pure open source model.
        """
        OPEN_SOURCE_CATEGORY_UNSPECIFIED = 0
        PROPRIETARY = 1
        GOOGLE_OWNED_OSS_WITH_GOOGLE_CHECKPOINT = 2
        THIRD_PARTY_OWNED_OSS_WITH_GOOGLE_CHECKPOINT = 3
        GOOGLE_OWNED_OSS = 4
        THIRD_PARTY_OWNED_OSS = 5

    class LaunchStage(proto.Enum):
        r"""An enum representing the launch stage of a PublisherModel.

        Values:
            LAUNCH_STAGE_UNSPECIFIED (0):
                The model launch stage is unspecified.
            EXPERIMENTAL (1):
                Used to indicate the PublisherModel is at
                Experimental launch stage, available to a small
                set of customers.
            PRIVATE_PREVIEW (2):
                Used to indicate the PublisherModel is at
                Private Preview launch stage, only available to
                a small set of customers, although a larger set
                of customers than an Experimental launch.
                Previews are the first launch stage used to get
                feedback from customers.
            PUBLIC_PREVIEW (3):
                Used to indicate the PublisherModel is at
                Public Preview launch stage, available to all
                customers, although not supported for production
                workloads.
            GA (4):
                Used to indicate the PublisherModel is at GA
                launch stage, available to all customers and
                ready for production workload.
        """
        LAUNCH_STAGE_UNSPECIFIED = 0
        EXPERIMENTAL = 1
        PRIVATE_PREVIEW = 2
        PUBLIC_PREVIEW = 3
        GA = 4

    class VersionState(proto.Enum):
        r"""An enum representing the state of the PublicModelVersion.

        Values:
            VERSION_STATE_UNSPECIFIED (0):
                The version state is unspecified.
            VERSION_STATE_STABLE (1):
                Used to indicate the version is stable.
            VERSION_STATE_UNSTABLE (2):
                Used to indicate the version is unstable.
        """
        VERSION_STATE_UNSPECIFIED = 0
        VERSION_STATE_STABLE = 1
        VERSION_STATE_UNSTABLE = 2

    class ResourceReference(proto.Message):
        r"""Reference to a resource.

        This message has `oneof`_ fields (mutually exclusive fields).
        For each oneof, at most one member field can be set at the same time.
        Setting any member of the oneof automatically clears all other
        members.

        .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

        Attributes:
            uri (str):
                The URI of the resource.

                This field is a member of `oneof`_ ``reference``.
            resource_name (str):
                The resource name of the Google Cloud
                resource.

                This field is a member of `oneof`_ ``reference``.
            use_case (str):
                Use case (CUJ) of the resource.

                This field is a member of `oneof`_ ``reference``.
            description (str):
                Description of the resource.

                This field is a member of `oneof`_ ``reference``.
        """

        uri: str = proto.Field(
            proto.STRING,
            number=1,
            oneof="reference",
        )
        resource_name: str = proto.Field(
            proto.STRING,
            number=2,
            oneof="reference",
        )
        use_case: str = proto.Field(
            proto.STRING,
            number=3,
            oneof="reference",
        )
        description: str = proto.Field(
            proto.STRING,
            number=4,
            oneof="reference",
        )

    class Documentation(proto.Message):
        r"""A named piece of documentation.

        Attributes:
            title (str):
                Required. E.g., OVERVIEW, USE CASES,
                DOCUMENTATION, SDK & SAMPLES, JAVA, NODE.JS,
                etc..
            content (str):
                Required. Content of this piece of document
                (in Markdown format).
        """

        title: str = proto.Field(
            proto.STRING,
            number=1,
        )
        content: str = proto.Field(
            proto.STRING,
            number=2,
        )

    class CallToAction(proto.Message):
        r"""Actions could take on this Publisher Model.

        .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

        Attributes:
            view_rest_api (google.cloud.aiplatform_v1.types.PublisherModel.CallToAction.ViewRestApi):
                Optional. To view Rest API docs.
            open_notebook (google.cloud.aiplatform_v1.types.PublisherModel.CallToAction.RegionalResourceReferences):
                Optional. Open notebook of the
                PublisherModel.
            open_notebooks (google.cloud.aiplatform_v1.types.PublisherModel.CallToAction.OpenNotebooks):
                Optional. Open notebooks of the
                PublisherModel.

                This field is a member of `oneof`_ ``_open_notebooks``.
            create_application (google.cloud.aiplatform_v1.types.PublisherModel.CallToAction.RegionalResourceReferences):
                Optional. Create application using the
                PublisherModel.
            open_fine_tuning_pipeline (google.cloud.aiplatform_v1.types.PublisherModel.CallToAction.RegionalResourceReferences):
                Optional. Open fine-tuning pipeline of the
                PublisherModel.
            open_fine_tuning_pipelines (google.cloud.aiplatform_v1.types.PublisherModel.CallToAction.OpenFineTuningPipelines):
                Optional. Open fine-tuning pipelines of the
                PublisherModel.

                This field is a member of `oneof`_ ``_open_fine_tuning_pipelines``.
            open_prompt_tuning_pipeline (google.cloud.aiplatform_v1.types.PublisherModel.CallToAction.RegionalResourceReferences):
                Optional. Open prompt-tuning pipeline of the
                PublisherModel.
            open_genie (google.cloud.aiplatform_v1.types.PublisherModel.CallToAction.RegionalResourceReferences):
                Optional. Open Genie / Playground.
            deploy (google.cloud.aiplatform_v1.types.PublisherModel.CallToAction.Deploy):
                Optional. Deploy the PublisherModel to Vertex
                Endpoint.
            deploy_gke (google.cloud.aiplatform_v1.types.PublisherModel.CallToAction.DeployGke):
                Optional. Deploy PublisherModel to Google
                Kubernetes Engine.
            open_generation_ai_studio (google.cloud.aiplatform_v1.types.PublisherModel.CallToAction.RegionalResourceReferences):
                Optional. Open in Generation AI Studio.
            request_access (google.cloud.aiplatform_v1.types.PublisherModel.CallToAction.RegionalResourceReferences):
                Optional. Request for access.
            open_evaluation_pipeline (google.cloud.aiplatform_v1.types.PublisherModel.CallToAction.RegionalResourceReferences):
                Optional. Open evaluation pipeline of the
                PublisherModel.
        """

        class RegionalResourceReferences(proto.Message):
            r"""The regional resource name or the URI. Key is region, e.g.,
            us-central1, europe-west2, global, etc..


            .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

            Attributes:
                references (MutableMapping[str, google.cloud.aiplatform_v1.types.PublisherModel.ResourceReference]):
                    Required.
                title (str):
                    Required.
                resource_title (str):
                    Optional. Title of the resource.

                    This field is a member of `oneof`_ ``_resource_title``.
                resource_use_case (str):
                    Optional. Use case (CUJ) of the resource.

                    This field is a member of `oneof`_ ``_resource_use_case``.
                resource_description (str):
                    Optional. Description of the resource.

                    This field is a member of `oneof`_ ``_resource_description``.
            """

            references: MutableMapping[
                str, "PublisherModel.ResourceReference"
            ] = proto.MapField(
                proto.STRING,
                proto.MESSAGE,
                number=1,
                message="PublisherModel.ResourceReference",
            )
            title: str = proto.Field(
                proto.STRING,
                number=2,
            )
            resource_title: str = proto.Field(
                proto.STRING,
                number=3,
                optional=True,
            )
            resource_use_case: str = proto.Field(
                proto.STRING,
                number=4,
                optional=True,
            )
            resource_description: str = proto.Field(
                proto.STRING,
                number=5,
                optional=True,
            )

        class ViewRestApi(proto.Message):
            r"""Rest API docs.

            Attributes:
                documentations (MutableSequence[google.cloud.aiplatform_v1.types.PublisherModel.Documentation]):
                    Required.
                title (str):
                    Required. The title of the view rest API.
            """

            documentations: MutableSequence[
                "PublisherModel.Documentation"
            ] = proto.RepeatedField(
                proto.MESSAGE,
                number=1,
                message="PublisherModel.Documentation",
            )
            title: str = proto.Field(
                proto.STRING,
                number=2,
            )

        class OpenNotebooks(proto.Message):
            r"""Open notebooks.

            Attributes:
                notebooks (MutableSequence[google.cloud.aiplatform_v1.types.PublisherModel.CallToAction.RegionalResourceReferences]):
                    Required. Regional resource references to
                    notebooks.
            """

            notebooks: MutableSequence[
                "PublisherModel.CallToAction.RegionalResourceReferences"
            ] = proto.RepeatedField(
                proto.MESSAGE,
                number=1,
                message="PublisherModel.CallToAction.RegionalResourceReferences",
            )

        class OpenFineTuningPipelines(proto.Message):
            r"""Open fine tuning pipelines.

            Attributes:
                fine_tuning_pipelines (MutableSequence[google.cloud.aiplatform_v1.types.PublisherModel.CallToAction.RegionalResourceReferences]):
                    Required. Regional resource references to
                    fine tuning pipelines.
            """

            fine_tuning_pipelines: MutableSequence[
                "PublisherModel.CallToAction.RegionalResourceReferences"
            ] = proto.RepeatedField(
                proto.MESSAGE,
                number=1,
                message="PublisherModel.CallToAction.RegionalResourceReferences",
            )

        class Deploy(proto.Message):
            r"""Model metadata that is needed for UploadModel or
            DeployModel/CreateEndpoint requests.

            This message has `oneof`_ fields (mutually exclusive fields).
            For each oneof, at most one member field can be set at the same time.
            Setting any member of the oneof automatically clears all other
            members.

            .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

            Attributes:
                dedicated_resources (google.cloud.aiplatform_v1.types.DedicatedResources):
                    A description of resources that are dedicated
                    to the DeployedModel, and that need a higher
                    degree of manual configuration.

                    This field is a member of `oneof`_ ``prediction_resources``.
                automatic_resources (google.cloud.aiplatform_v1.types.AutomaticResources):
                    A description of resources that to large
                    degree are decided by Vertex AI, and require
                    only a modest additional configuration.

                    This field is a member of `oneof`_ ``prediction_resources``.
                shared_resources (str):
                    The resource name of the shared DeploymentResourcePool to
                    deploy on. Format:
                    ``projects/{project}/locations/{location}/deploymentResourcePools/{deployment_resource_pool}``

                    This field is a member of `oneof`_ ``prediction_resources``.
                model_display_name (str):
                    Optional. Default model display name.
                large_model_reference (google.cloud.aiplatform_v1.types.LargeModelReference):
                    Optional. Large model reference. When this is set,
                    model_artifact_spec is not needed.
                container_spec (google.cloud.aiplatform_v1.types.ModelContainerSpec):
                    Optional. The specification of the container
                    that is to be used when deploying this Model in
                    Vertex AI. Not present for Large Models.
                artifact_uri (str):
                    Optional. The path to the directory
                    containing the Model artifact and any of its
                    supporting files.
                title (str):
                    Required. The title of the regional resource
                    reference.
                public_artifact_uri (str):
                    Optional. The signed URI for ephemeral Cloud
                    Storage access to model artifact.
            """

            dedicated_resources: machine_resources.DedicatedResources = proto.Field(
                proto.MESSAGE,
                number=5,
                oneof="prediction_resources",
                message=machine_resources.DedicatedResources,
            )
            automatic_resources: machine_resources.AutomaticResources = proto.Field(
                proto.MESSAGE,
                number=6,
                oneof="prediction_resources",
                message=machine_resources.AutomaticResources,
            )
            shared_resources: str = proto.Field(
                proto.STRING,
                number=7,
                oneof="prediction_resources",
            )
            model_display_name: str = proto.Field(
                proto.STRING,
                number=1,
            )
            large_model_reference: model.LargeModelReference = proto.Field(
                proto.MESSAGE,
                number=2,
                message=model.LargeModelReference,
            )
            container_spec: model.ModelContainerSpec = proto.Field(
                proto.MESSAGE,
                number=3,
                message=model.ModelContainerSpec,
            )
            artifact_uri: str = proto.Field(
                proto.STRING,
                number=4,
            )
            title: str = proto.Field(
                proto.STRING,
                number=8,
            )
            public_artifact_uri: str = proto.Field(
                proto.STRING,
                number=9,
            )

        class DeployGke(proto.Message):
            r"""Configurations for PublisherModel GKE deployment

            Attributes:
                gke_yaml_configs (MutableSequence[str]):
                    Optional. GKE deployment configuration in
                    yaml format.
            """

            gke_yaml_configs: MutableSequence[str] = proto.RepeatedField(
                proto.STRING,
                number=1,
            )

        view_rest_api: "PublisherModel.CallToAction.ViewRestApi" = proto.Field(
            proto.MESSAGE,
            number=1,
            message="PublisherModel.CallToAction.ViewRestApi",
        )
        open_notebook: "PublisherModel.CallToAction.RegionalResourceReferences" = (
            proto.Field(
                proto.MESSAGE,
                number=2,
                message="PublisherModel.CallToAction.RegionalResourceReferences",
            )
        )
        open_notebooks: "PublisherModel.CallToAction.OpenNotebooks" = proto.Field(
            proto.MESSAGE,
            number=12,
            optional=True,
            message="PublisherModel.CallToAction.OpenNotebooks",
        )
        create_application: "PublisherModel.CallToAction.RegionalResourceReferences" = (
            proto.Field(
                proto.MESSAGE,
                number=3,
                message="PublisherModel.CallToAction.RegionalResourceReferences",
            )
        )
        open_fine_tuning_pipeline: "PublisherModel.CallToAction.RegionalResourceReferences" = proto.Field(
            proto.MESSAGE,
            number=4,
            message="PublisherModel.CallToAction.RegionalResourceReferences",
        )
        open_fine_tuning_pipelines: "PublisherModel.CallToAction.OpenFineTuningPipelines" = proto.Field(
            proto.MESSAGE,
            number=13,
            optional=True,
            message="PublisherModel.CallToAction.OpenFineTuningPipelines",
        )
        open_prompt_tuning_pipeline: "PublisherModel.CallToAction.RegionalResourceReferences" = proto.Field(
            proto.MESSAGE,
            number=5,
            message="PublisherModel.CallToAction.RegionalResourceReferences",
        )
        open_genie: "PublisherModel.CallToAction.RegionalResourceReferences" = (
            proto.Field(
                proto.MESSAGE,
                number=6,
                message="PublisherModel.CallToAction.RegionalResourceReferences",
            )
        )
        deploy: "PublisherModel.CallToAction.Deploy" = proto.Field(
            proto.MESSAGE,
            number=7,
            message="PublisherModel.CallToAction.Deploy",
        )
        deploy_gke: "PublisherModel.CallToAction.DeployGke" = proto.Field(
            proto.MESSAGE,
            number=14,
            message="PublisherModel.CallToAction.DeployGke",
        )
        open_generation_ai_studio: "PublisherModel.CallToAction.RegionalResourceReferences" = proto.Field(
            proto.MESSAGE,
            number=8,
            message="PublisherModel.CallToAction.RegionalResourceReferences",
        )
        request_access: "PublisherModel.CallToAction.RegionalResourceReferences" = (
            proto.Field(
                proto.MESSAGE,
                number=9,
                message="PublisherModel.CallToAction.RegionalResourceReferences",
            )
        )
        open_evaluation_pipeline: "PublisherModel.CallToAction.RegionalResourceReferences" = proto.Field(
            proto.MESSAGE,
            number=11,
            message="PublisherModel.CallToAction.RegionalResourceReferences",
        )

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    version_id: str = proto.Field(
        proto.STRING,
        number=2,
    )
    open_source_category: OpenSourceCategory = proto.Field(
        proto.ENUM,
        number=7,
        enum=OpenSourceCategory,
    )
    supported_actions: CallToAction = proto.Field(
        proto.MESSAGE,
        number=19,
        message=CallToAction,
    )
    frameworks: MutableSequence[str] = proto.RepeatedField(
        proto.STRING,
        number=23,
    )
    launch_stage: LaunchStage = proto.Field(
        proto.ENUM,
        number=29,
        enum=LaunchStage,
    )
    version_state: VersionState = proto.Field(
        proto.ENUM,
        number=37,
        enum=VersionState,
    )
    publisher_model_template: str = proto.Field(
        proto.STRING,
        number=30,
    )
    predict_schemata: model.PredictSchemata = proto.Field(
        proto.MESSAGE,
        number=31,
        message=model.PredictSchemata,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
