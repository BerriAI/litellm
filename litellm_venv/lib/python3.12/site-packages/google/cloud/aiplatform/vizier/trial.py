# -*- coding: utf-8 -*-

# Copyright 2022 Google LLC
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
import copy

from typing import Optional, TypeVar, Mapping, Any
from google.cloud.aiplatform.vizier.client_abc import TrialInterface

from google.auth import credentials as auth_credentials
from google.cloud.aiplatform import base
from google.cloud.aiplatform import utils
from google.cloud.aiplatform.vizier import study
from google.cloud.aiplatform.vizier import pyvizier as vz

_T = TypeVar("_T")
_LOGGER = base.Logger(__name__)


class Trial(base.VertexAiResourceNounWithFutureManager, TrialInterface):
    """Manage Trial resource for Vertex Vizier."""

    client_class = utils.VizierClientWithOverride

    _resource_noun = "trial"
    _getter_method = "get_trial"
    _list_method = "list_trials"
    _delete_method = "delete_trial"
    _parse_resource_name_method = "parse_trial_path"
    _format_resource_name_method = "trial_path"

    def __init__(
        self,
        trial_name: str,
        study_id: Optional[str] = None,
        project: Optional[str] = None,
        location: Optional[str] = None,
        credentials: Optional[auth_credentials.Credentials] = None,
    ):
        """Retrieves an existing managed trial given a trial resource name or a study id.

        Example Usage:
            trial = aiplatform.Trial(trial_name = 'projects/123/locations/us-central1/studies/12345678/trials/1')
            or
            trial = aiplatform.Trial(trial_name = '1', study_id = '12345678')

        Args:
            trial_name (str):
                Required. A fully-qualified trial resource name or a trial ID.
                Example: "projects/123/locations/us-central1/studies/12345678/trials/1" or "12345678" when
                project and location are initialized or passed.
            study_id (str):
                Optional. A fully-qualified study resource name or a study ID.
                Example: "projects/123/locations/us-central1/studies/12345678" or "12345678" when
                project and location are initialized or passed.
            project (str):
                Optional. Project to retrieve trial from. If not set, project
                set in aiplatform.init will be used.
            location (str):
                Optional. Location to retrieve trial from. If not set, location
                set in aiplatform.init will be used.
            credentials (auth_credentials.Credentials):
                Optional. Custom credentials to use to retrieve this Feature. Overrides
                credentials set in aiplatform.init.
        """

        base.VertexAiResourceNounWithFutureManager.__init__(
            self,
            project=project,
            location=location,
            credentials=credentials,
            resource_name=trial_name,
        )
        self._gca_resource = self._get_gca_resource(
            resource_name=trial_name,
            parent_resource_name_fields={
                study.Study._resource_noun: study_id,
            }
            if study_id
            else study_id,
        )

    @property
    def uid(self) -> int:
        """Unique identifier of the trial."""
        trial_path_components = self._parse_resource_name(self.resource_name)
        return int(trial_path_components["trial"])

    @property
    def parameters(self) -> Mapping[str, Any]:
        """Parameters of the trial."""
        trial = self.api_client.get_trial(name=self.resource_name)
        return vz.TrialConverter.from_proto(trial).parameters

    @property
    def status(self) -> vz.TrialStatus:
        """Status of the Trial."""
        trial = self.api_client.get_trial(name=self.resource_name)
        return vz.TrialConverter.from_proto(trial).status

    def delete(self) -> None:
        """Deletes the Trial in Vizier service."""
        self.api_client.delete_trial(name=self.resource_name)

    def complete(
        self,
        measurement: Optional[vz.Measurement] = None,
        *,
        infeasible_reason: Optional[str] = None
    ) -> Optional[vz.Measurement]:
        """Completes the trial and #materializes the measurement.

        * If `measurement` is provided, then Vizier writes it as the trial's final
        measurement and returns it.
        * If `infeasible_reason` is provided, `measurement` is not needed.
        * If neither is provided, then Vizier selects an existing (intermediate)
        measurement to be the final measurement and returns it.

        Args:
          measurement: Final measurement.
          infeasible_reason: Indefeasibly reason for missing final measurement.
        """
        complete_trial_request = {"name": self.resource_name}
        if infeasible_reason is not None:
            complete_trial_request["infeasible_reason"] = infeasible_reason
            complete_trial_request["trial_infeasible"] = True
        if measurement is not None:
            complete_trial_request[
                "final_measurement"
            ] = vz.MeasurementConverter.to_proto(measurement)
        trial = self.api_client.complete_trial(request=complete_trial_request)
        return (
            vz.MeasurementConverter.from_proto(trial.final_measurement)
            if trial.final_measurement
            else None
        )

    def should_stop(self) -> bool:
        """Returns true if the Trial should stop."""
        check_trial_early_stopping_state_request = {"trial_name": self.resource_name}
        should_stop_lro = self.api_client.check_trial_early_stopping_state(
            request=check_trial_early_stopping_state_request
        )
        _LOGGER.log_action_started_against_resource_with_lro(
            "ShouldStop", "trial", self.__class__, should_stop_lro
        )
        should_stop_lro.result()
        _LOGGER.log_action_completed_against_resource("trial", "should_stop", self)
        return should_stop_lro.result().should_stop

    def add_measurement(self, measurement: vz.Measurement) -> None:
        """Adds an intermediate measurement."""
        add_trial_measurement_request = {
            "trial_name": self.resource_name,
        }
        add_trial_measurement_request["measurement"] = vz.MeasurementConverter.to_proto(
            measurement
        )
        self.api_client.add_trial_measurement(request=add_trial_measurement_request)

    def materialize(self, *, include_all_measurements: bool = True) -> vz.Trial:
        """#Materializes the Trial.

        Args:
          include_all_measurements: If True, returned Trial includes all
            intermediate measurements. The final measurement is always provided.
        """
        trial = self.api_client.get_trial(name=self.resource_name)
        return copy.deepcopy(vz.TrialConverter.from_proto(trial))
