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

from google.auth import credentials as auth_credentials

from google.cloud.aiplatform.schedules import _Schedule as _ScheduleGa


class _Schedule(
    _ScheduleGa,
):
    """Preview Schedule resource for Vertex AI."""

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

    @property
    def cron_expression(self) -> str:
        """Current Schedule cron expression.

        Returns:
            Schedule cron expression.
        """
        return super().cron
