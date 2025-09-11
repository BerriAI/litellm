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
#

from google.protobuf import duration_pb2  # type: ignore
from google.cloud.aiplatform_v1.types import (
    model_deployment_monitoring_job as gca_model_deployment_monitoring_job,
)


class ScheduleConfig:
    """A class that configures model monitoring schedule."""

    def __init__(self, monitor_interval: int):
        """Initializer for ScheduleConfig.

        Args:
        monitor_interval (int):
            Sets the model monitoring job scheduling interval in hours.
            This defines how often the monitoring jobs are triggered.
        """
        super().__init__()
        self.monitor_interval = monitor_interval

    def as_proto(self):
        """Returns ScheduleConfig as a proto message."""
        return (
            gca_model_deployment_monitoring_job.ModelDeploymentMonitoringScheduleConfig(
                monitor_interval=duration_pb2.Duration(
                    seconds=self.monitor_interval * 3600
                )
            )
        )
