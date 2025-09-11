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

from google.cloud.aiplatform_v1.types import operation
from google.cloud.aiplatform_v1.types import study as gca_study
from google.protobuf import timestamp_pb2  # type: ignore


__protobuf__ = proto.module(
    package="google.cloud.aiplatform.v1",
    manifest={
        "GetStudyRequest",
        "CreateStudyRequest",
        "ListStudiesRequest",
        "ListStudiesResponse",
        "DeleteStudyRequest",
        "LookupStudyRequest",
        "SuggestTrialsRequest",
        "SuggestTrialsResponse",
        "SuggestTrialsMetadata",
        "CreateTrialRequest",
        "GetTrialRequest",
        "ListTrialsRequest",
        "ListTrialsResponse",
        "AddTrialMeasurementRequest",
        "CompleteTrialRequest",
        "DeleteTrialRequest",
        "CheckTrialEarlyStoppingStateRequest",
        "CheckTrialEarlyStoppingStateResponse",
        "CheckTrialEarlyStoppingStateMetatdata",
        "StopTrialRequest",
        "ListOptimalTrialsRequest",
        "ListOptimalTrialsResponse",
    },
)


class GetStudyRequest(proto.Message):
    r"""Request message for
    [VizierService.GetStudy][google.cloud.aiplatform.v1.VizierService.GetStudy].

    Attributes:
        name (str):
            Required. The name of the Study resource. Format:
            ``projects/{project}/locations/{location}/studies/{study}``
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


class CreateStudyRequest(proto.Message):
    r"""Request message for
    [VizierService.CreateStudy][google.cloud.aiplatform.v1.VizierService.CreateStudy].

    Attributes:
        parent (str):
            Required. The resource name of the Location to create the
            CustomJob in. Format:
            ``projects/{project}/locations/{location}``
        study (google.cloud.aiplatform_v1.types.Study):
            Required. The Study configuration used to
            create the Study.
    """

    parent: str = proto.Field(
        proto.STRING,
        number=1,
    )
    study: gca_study.Study = proto.Field(
        proto.MESSAGE,
        number=2,
        message=gca_study.Study,
    )


class ListStudiesRequest(proto.Message):
    r"""Request message for
    [VizierService.ListStudies][google.cloud.aiplatform.v1.VizierService.ListStudies].

    Attributes:
        parent (str):
            Required. The resource name of the Location to list the
            Study from. Format:
            ``projects/{project}/locations/{location}``
        page_token (str):
            Optional. A page token to request the next
            page of results. If unspecified, there are no
            subsequent pages.
        page_size (int):
            Optional. The maximum number of studies to
            return per "page" of results. If unspecified,
            service will pick an appropriate default.
    """

    parent: str = proto.Field(
        proto.STRING,
        number=1,
    )
    page_token: str = proto.Field(
        proto.STRING,
        number=2,
    )
    page_size: int = proto.Field(
        proto.INT32,
        number=3,
    )


class ListStudiesResponse(proto.Message):
    r"""Response message for
    [VizierService.ListStudies][google.cloud.aiplatform.v1.VizierService.ListStudies].

    Attributes:
        studies (MutableSequence[google.cloud.aiplatform_v1.types.Study]):
            The studies associated with the project.
        next_page_token (str):
            Passes this token as the ``page_token`` field of the request
            for a subsequent call. If this field is omitted, there are
            no subsequent pages.
    """

    @property
    def raw_page(self):
        return self

    studies: MutableSequence[gca_study.Study] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message=gca_study.Study,
    )
    next_page_token: str = proto.Field(
        proto.STRING,
        number=2,
    )


class DeleteStudyRequest(proto.Message):
    r"""Request message for
    [VizierService.DeleteStudy][google.cloud.aiplatform.v1.VizierService.DeleteStudy].

    Attributes:
        name (str):
            Required. The name of the Study resource to be deleted.
            Format:
            ``projects/{project}/locations/{location}/studies/{study}``
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


class LookupStudyRequest(proto.Message):
    r"""Request message for
    [VizierService.LookupStudy][google.cloud.aiplatform.v1.VizierService.LookupStudy].

    Attributes:
        parent (str):
            Required. The resource name of the Location to get the Study
            from. Format: ``projects/{project}/locations/{location}``
        display_name (str):
            Required. The user-defined display name of
            the Study
    """

    parent: str = proto.Field(
        proto.STRING,
        number=1,
    )
    display_name: str = proto.Field(
        proto.STRING,
        number=2,
    )


class SuggestTrialsRequest(proto.Message):
    r"""Request message for
    [VizierService.SuggestTrials][google.cloud.aiplatform.v1.VizierService.SuggestTrials].

    Attributes:
        parent (str):
            Required. The project and location that the Study belongs
            to. Format:
            ``projects/{project}/locations/{location}/studies/{study}``
        suggestion_count (int):
            Required. The number of suggestions
            requested. It must be positive.
        client_id (str):
            Required. The identifier of the client that is requesting
            the suggestion.

            If multiple SuggestTrialsRequests have the same
            ``client_id``, the service will return the identical
            suggested Trial if the Trial is pending, and provide a new
            Trial if the last suggested Trial was completed.
        contexts (MutableSequence[google.cloud.aiplatform_v1.types.TrialContext]):
            Optional. This allows you to specify the "context" for a
            Trial; a context is a slice (a subspace) of the search
            space.

            Typical uses for contexts:

            1) You are using Vizier to tune a server for best
               performance, but there's a strong weekly cycle. The
               context specifies the day-of-week. This allows Tuesday to
               generalize from Wednesday without assuming that
               everything is identical.
            2) Imagine you're optimizing some medical treatment for
               people. As they walk in the door, you know certain facts
               about them (e.g. sex, weight, height, blood-pressure).
               Put that information in the context, and Vizier will
               adapt its suggestions to the patient.
            3) You want to do a fair A/B test efficiently. Specify the
               "A" and "B" conditions as contexts, and Vizier will
               generalize between "A" and "B" conditions. If they are
               similar, this will allow Vizier to converge to the
               optimum faster than if "A" and "B" were separate Studies.
               NOTE: You can also enter contexts as REQUESTED Trials,
               e.g. via the CreateTrial() RPC; that's the asynchronous
               option where you don't need a close association between
               contexts and suggestions.

            NOTE: All the Parameters you set in a context MUST be
            defined in the Study. NOTE: You must supply 0 or
            $suggestion_count contexts. If you don't supply any
            contexts, Vizier will make suggestions from the full search
            space specified in the StudySpec; if you supply a full set
            of context, each suggestion will match the corresponding
            context. NOTE: A Context with no features set matches
            anything, and allows suggestions from the full search space.
            NOTE: Contexts MUST lie within the search space specified in
            the StudySpec. It's an error if they don't. NOTE: Contexts
            preferentially match ACTIVE then REQUESTED trials before new
            suggestions are generated. NOTE: Generation of suggestions
            involves a match between a Context and (optionally) a
            REQUESTED trial; if that match is not fully specified, a
            suggestion will be geneated in the merged subspace.
    """

    parent: str = proto.Field(
        proto.STRING,
        number=1,
    )
    suggestion_count: int = proto.Field(
        proto.INT32,
        number=2,
    )
    client_id: str = proto.Field(
        proto.STRING,
        number=3,
    )
    contexts: MutableSequence[gca_study.TrialContext] = proto.RepeatedField(
        proto.MESSAGE,
        number=4,
        message=gca_study.TrialContext,
    )


class SuggestTrialsResponse(proto.Message):
    r"""Response message for
    [VizierService.SuggestTrials][google.cloud.aiplatform.v1.VizierService.SuggestTrials].

    Attributes:
        trials (MutableSequence[google.cloud.aiplatform_v1.types.Trial]):
            A list of Trials.
        study_state (google.cloud.aiplatform_v1.types.Study.State):
            The state of the Study.
        start_time (google.protobuf.timestamp_pb2.Timestamp):
            The time at which the operation was started.
        end_time (google.protobuf.timestamp_pb2.Timestamp):
            The time at which operation processing
            completed.
    """

    trials: MutableSequence[gca_study.Trial] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message=gca_study.Trial,
    )
    study_state: gca_study.Study.State = proto.Field(
        proto.ENUM,
        number=2,
        enum=gca_study.Study.State,
    )
    start_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=3,
        message=timestamp_pb2.Timestamp,
    )
    end_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=4,
        message=timestamp_pb2.Timestamp,
    )


class SuggestTrialsMetadata(proto.Message):
    r"""Details of operations that perform Trials suggestion.

    Attributes:
        generic_metadata (google.cloud.aiplatform_v1.types.GenericOperationMetadata):
            Operation metadata for suggesting Trials.
        client_id (str):
            The identifier of the client that is requesting the
            suggestion.

            If multiple SuggestTrialsRequests have the same
            ``client_id``, the service will return the identical
            suggested Trial if the Trial is pending, and provide a new
            Trial if the last suggested Trial was completed.
    """

    generic_metadata: operation.GenericOperationMetadata = proto.Field(
        proto.MESSAGE,
        number=1,
        message=operation.GenericOperationMetadata,
    )
    client_id: str = proto.Field(
        proto.STRING,
        number=2,
    )


class CreateTrialRequest(proto.Message):
    r"""Request message for
    [VizierService.CreateTrial][google.cloud.aiplatform.v1.VizierService.CreateTrial].

    Attributes:
        parent (str):
            Required. The resource name of the Study to create the Trial
            in. Format:
            ``projects/{project}/locations/{location}/studies/{study}``
        trial (google.cloud.aiplatform_v1.types.Trial):
            Required. The Trial to create.
    """

    parent: str = proto.Field(
        proto.STRING,
        number=1,
    )
    trial: gca_study.Trial = proto.Field(
        proto.MESSAGE,
        number=2,
        message=gca_study.Trial,
    )


class GetTrialRequest(proto.Message):
    r"""Request message for
    [VizierService.GetTrial][google.cloud.aiplatform.v1.VizierService.GetTrial].

    Attributes:
        name (str):
            Required. The name of the Trial resource. Format:
            ``projects/{project}/locations/{location}/studies/{study}/trials/{trial}``
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


class ListTrialsRequest(proto.Message):
    r"""Request message for
    [VizierService.ListTrials][google.cloud.aiplatform.v1.VizierService.ListTrials].

    Attributes:
        parent (str):
            Required. The resource name of the Study to list the Trial
            from. Format:
            ``projects/{project}/locations/{location}/studies/{study}``
        page_token (str):
            Optional. A page token to request the next
            page of results. If unspecified, there are no
            subsequent pages.
        page_size (int):
            Optional. The number of Trials to retrieve
            per "page" of results. If unspecified, the
            service will pick an appropriate default.
    """

    parent: str = proto.Field(
        proto.STRING,
        number=1,
    )
    page_token: str = proto.Field(
        proto.STRING,
        number=2,
    )
    page_size: int = proto.Field(
        proto.INT32,
        number=3,
    )


class ListTrialsResponse(proto.Message):
    r"""Response message for
    [VizierService.ListTrials][google.cloud.aiplatform.v1.VizierService.ListTrials].

    Attributes:
        trials (MutableSequence[google.cloud.aiplatform_v1.types.Trial]):
            The Trials associated with the Study.
        next_page_token (str):
            Pass this token as the ``page_token`` field of the request
            for a subsequent call. If this field is omitted, there are
            no subsequent pages.
    """

    @property
    def raw_page(self):
        return self

    trials: MutableSequence[gca_study.Trial] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message=gca_study.Trial,
    )
    next_page_token: str = proto.Field(
        proto.STRING,
        number=2,
    )


class AddTrialMeasurementRequest(proto.Message):
    r"""Request message for
    [VizierService.AddTrialMeasurement][google.cloud.aiplatform.v1.VizierService.AddTrialMeasurement].

    Attributes:
        trial_name (str):
            Required. The name of the trial to add measurement. Format:
            ``projects/{project}/locations/{location}/studies/{study}/trials/{trial}``
        measurement (google.cloud.aiplatform_v1.types.Measurement):
            Required. The measurement to be added to a
            Trial.
    """

    trial_name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    measurement: gca_study.Measurement = proto.Field(
        proto.MESSAGE,
        number=3,
        message=gca_study.Measurement,
    )


class CompleteTrialRequest(proto.Message):
    r"""Request message for
    [VizierService.CompleteTrial][google.cloud.aiplatform.v1.VizierService.CompleteTrial].

    Attributes:
        name (str):
            Required. The Trial's name. Format:
            ``projects/{project}/locations/{location}/studies/{study}/trials/{trial}``
        final_measurement (google.cloud.aiplatform_v1.types.Measurement):
            Optional. If provided, it will be used as the completed
            Trial's final_measurement; Otherwise, the service will
            auto-select a previously reported measurement as the
            final-measurement
        trial_infeasible (bool):
            Optional. True if the Trial cannot be run with the given
            Parameter, and final_measurement will be ignored.
        infeasible_reason (str):
            Optional. A human readable reason why the trial was
            infeasible. This should only be provided if
            ``trial_infeasible`` is true.
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    final_measurement: gca_study.Measurement = proto.Field(
        proto.MESSAGE,
        number=2,
        message=gca_study.Measurement,
    )
    trial_infeasible: bool = proto.Field(
        proto.BOOL,
        number=3,
    )
    infeasible_reason: str = proto.Field(
        proto.STRING,
        number=4,
    )


class DeleteTrialRequest(proto.Message):
    r"""Request message for
    [VizierService.DeleteTrial][google.cloud.aiplatform.v1.VizierService.DeleteTrial].

    Attributes:
        name (str):
            Required. The Trial's name. Format:
            ``projects/{project}/locations/{location}/studies/{study}/trials/{trial}``
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


class CheckTrialEarlyStoppingStateRequest(proto.Message):
    r"""Request message for
    [VizierService.CheckTrialEarlyStoppingState][google.cloud.aiplatform.v1.VizierService.CheckTrialEarlyStoppingState].

    Attributes:
        trial_name (str):
            Required. The Trial's name. Format:
            ``projects/{project}/locations/{location}/studies/{study}/trials/{trial}``
    """

    trial_name: str = proto.Field(
        proto.STRING,
        number=1,
    )


class CheckTrialEarlyStoppingStateResponse(proto.Message):
    r"""Response message for
    [VizierService.CheckTrialEarlyStoppingState][google.cloud.aiplatform.v1.VizierService.CheckTrialEarlyStoppingState].

    Attributes:
        should_stop (bool):
            True if the Trial should stop.
    """

    should_stop: bool = proto.Field(
        proto.BOOL,
        number=1,
    )


class CheckTrialEarlyStoppingStateMetatdata(proto.Message):
    r"""This message will be placed in the metadata field of a
    google.longrunning.Operation associated with a
    CheckTrialEarlyStoppingState request.

    Attributes:
        generic_metadata (google.cloud.aiplatform_v1.types.GenericOperationMetadata):
            Operation metadata for suggesting Trials.
        study (str):
            The name of the Study that the Trial belongs
            to.
        trial (str):
            The Trial name.
    """

    generic_metadata: operation.GenericOperationMetadata = proto.Field(
        proto.MESSAGE,
        number=1,
        message=operation.GenericOperationMetadata,
    )
    study: str = proto.Field(
        proto.STRING,
        number=2,
    )
    trial: str = proto.Field(
        proto.STRING,
        number=3,
    )


class StopTrialRequest(proto.Message):
    r"""Request message for
    [VizierService.StopTrial][google.cloud.aiplatform.v1.VizierService.StopTrial].

    Attributes:
        name (str):
            Required. The Trial's name. Format:
            ``projects/{project}/locations/{location}/studies/{study}/trials/{trial}``
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


class ListOptimalTrialsRequest(proto.Message):
    r"""Request message for
    [VizierService.ListOptimalTrials][google.cloud.aiplatform.v1.VizierService.ListOptimalTrials].

    Attributes:
        parent (str):
            Required. The name of the Study that the
            optimal Trial belongs to.
    """

    parent: str = proto.Field(
        proto.STRING,
        number=1,
    )


class ListOptimalTrialsResponse(proto.Message):
    r"""Response message for
    [VizierService.ListOptimalTrials][google.cloud.aiplatform.v1.VizierService.ListOptimalTrials].

    Attributes:
        optimal_trials (MutableSequence[google.cloud.aiplatform_v1.types.Trial]):
            The pareto-optimal Trials for multiple objective Study or
            the optimal trial for single objective Study. The definition
            of pareto-optimal can be checked in wiki page.
            https://en.wikipedia.org/wiki/Pareto_efficiency
    """

    optimal_trials: MutableSequence[gca_study.Trial] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message=gca_study.Trial,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
