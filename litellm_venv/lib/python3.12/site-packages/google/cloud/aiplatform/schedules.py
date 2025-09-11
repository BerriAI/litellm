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

import time
from typing import Any, Optional

from google.auth import credentials as auth_credentials
from google.cloud.aiplatform import base
from google.cloud.aiplatform import utils
from google.cloud.aiplatform.compat.types import (
    schedule as gca_schedule,
)
from google.cloud.aiplatform.constants import (
    schedule as schedule_constants,
)

_LOGGER = base.Logger(__name__)

_SCHEDULE_COMPLETE_STATES = schedule_constants._SCHEDULE_COMPLETE_STATES

_SCHEDULE_ERROR_STATES = schedule_constants._SCHEDULE_ERROR_STATES


class _Schedule(
    base.VertexAiStatefulResource,
):
    """Schedule resource for Vertex AI."""

    client_class = utils.ScheduleClientWithOverride
    _resource_noun = "schedules"
    _delete_method = "delete_schedule"
    _getter_method = "get_schedule"
    _list_method = "list_schedules"
    _pause_method = "pause_schedule"
    _resume_method = "resume_schedule"
    _parse_resource_name_method = "parse_schedule_path"
    _format_resource_name_method = "schedule_path"

    # Required by the done() method
    _valid_done_states = schedule_constants._SCHEDULE_COMPLETE_STATES

    def __init__(
        self,
        credentials: auth_credentials.Credentials,
        project: str,
        location: str,
    ):
        """Retrieves a Schedule resource and instantiates its representation.
        Args:
            credentials (auth_credentials.Credentials):
                Optional. Custom credentials to use to create this Schedule.
                Overrides credentials set in aiplatform.init.
            project (str):
                Optional. The project that you want to run this Schedule in.
                If not set, the project set in aiplatform.init will be used.
            location (str):
                Optional. Location to create Schedule. If not set,
                location set in aiplatform.init will be used.
        """
        super().__init__(project=project, location=location, credentials=credentials)

    @classmethod
    def get(
        cls,
        schedule_id: str,
        project: Optional[str] = None,
        location: Optional[str] = None,
        credentials: Optional[auth_credentials.Credentials] = None,
    ) -> Any:
        """Get a Vertex AI Schedule for the given resource_name.

        Args:
            schedule_id (str):
                Required. Schedule ID used to identify or locate the schedule.
            project (str):
                Optional. Project to retrieve dataset from. If not set, project
                set in aiplatform.init will be used.
            location (str):
                Optional. Location to retrieve dataset from. If not set,
                location set in aiplatform.init will be used.
            credentials (auth_credentials.Credentials):
                Optional. Custom credentials to use to upload this model.
                Overrides credentials set in aiplatform.init.

        Returns:
            A Vertex AI Schedule.
        """
        self = cls._empty_constructor(
            project=project,
            location=location,
            credentials=credentials,
            resource_name=schedule_id,
        )

        self._gca_resource = self._get_gca_resource(resource_name=schedule_id)

        return self

    def pause(self) -> None:
        """Starts asynchronous pause on the Schedule.

        Changes Schedule state from State.ACTIVE to State.PAUSED.
        """
        self.api_client.pause_schedule(name=self.resource_name)

    def resume(
        self,
        catch_up: bool = True,
    ) -> None:
        """Starts asynchronous resume on the Schedule.

        Changes Schedule state from State.PAUSED to State.ACTIVE.

        Args:
            catch_up (bool):
                Optional. Whether to backfill missed runs when the Schedule is
                resumed from State.PAUSED.
        """
        self.api_client.resume_schedule(name=self.resource_name)

    def done(self) -> bool:
        """Helper method that return True is Schedule is done. False otherwise."""
        if not self._gca_resource:
            return False

        return self.state in _SCHEDULE_COMPLETE_STATES

    def wait(self) -> None:
        """Wait for all runs scheduled by this Schedule to complete."""
        if self._latest_future is None:
            self._block_until_complete()
        else:
            super().wait()

    @property
    def state(self) -> Optional[gca_schedule.Schedule.State]:
        """Current Schedule state.

        Returns:
            Schedule state.
        """
        self._sync_gca_resource()
        return self._gca_resource.state

    @property
    def max_run_count(self) -> int:
        """Current Schedule max_run_count.

        Returns:
            Schedule max_run_count.
        """
        self._sync_gca_resource()
        return self._gca_resource.max_run_count

    @property
    def cron(self) -> str:
        """Current Schedule cron.

        Returns:
            Schedule cron.
        """
        self._sync_gca_resource()
        return self._gca_resource.cron

    @property
    def max_concurrent_run_count(self) -> int:
        """Current Schedule max_concurrent_run_count.

        Returns:
            Schedule max_concurrent_run_count.
        """
        self._sync_gca_resource()
        return self._gca_resource.max_concurrent_run_count

    @property
    def allow_queueing(self) -> bool:
        """Whether current Schedule allows queueing.

        Returns:
            Schedule allow_queueing.
        """
        self._sync_gca_resource()
        return self._gca_resource.allow_queueing

    def _block_until_complete(self) -> None:
        """Helper method to block and check on Schedule until complete."""
        # Used these numbers so failures surface fast
        wait = 5  # start at five seconds
        log_wait = 5
        max_wait = 60 * 5  # 5 minute wait
        multiplier = 2  # scale wait by 2 every iteration

        previous_time = time.time()
        while self.state not in _SCHEDULE_COMPLETE_STATES:
            current_time = time.time()
            if current_time - previous_time >= log_wait:
                _LOGGER.info(
                    "%s %s current state:\n%s"
                    % (
                        self.__class__.__name__,
                        self._gca_resource.name,
                        self._gca_resource.state,
                    )
                )
                log_wait = min(log_wait * multiplier, max_wait)
                previous_time = current_time
            time.sleep(wait)

        # Error is only populated when the schedule state is STATE_UNSPECIFIED.
        if self._gca_resource.state in _SCHEDULE_ERROR_STATES:
            raise RuntimeError("Schedule failed with:\n%s" % self._gca_resource.error)
        else:
            _LOGGER.log_action_completed_against_resource("run", "completed", self)

    def _dashboard_uri(self) -> str:
        """Helper method to compose the dashboard uri where Schedule can be
        viewed.

        Returns:
            Dashboard uri where Schedule can be viewed.
        """
        fields = self._parse_resource_name(self.resource_name)
        url = f"https://console.cloud.google.com/vertex-ai/locations/{fields['location']}/pipelines/schedules/{fields['schedule']}?project={fields['project']}"
        return url
