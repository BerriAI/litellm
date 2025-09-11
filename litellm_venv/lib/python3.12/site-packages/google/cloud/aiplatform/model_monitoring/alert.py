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

from typing import List, Optional
from google.cloud.aiplatform_v1.types import (
    model_monitoring as gca_model_monitoring_v1,
)

# TODO(b/242108750): remove temporary logic once model monitoring for
# batch prediction is GA.
from google.cloud.aiplatform_v1beta1.types import (
    model_monitoring as gca_model_monitoring_v1beta1,
)

gca_model_monitoring = gca_model_monitoring_v1


class AlertConfig:
    def __init__(
        self,
        user_emails: List[str] = [],
        enable_logging: Optional[bool] = False,
        notification_channels: List[str] = [],
    ):
        """Initializer for AlertConfig.

        Args:
            user_emails (List[str]): The email addresses to send the alert to.
            enable_logging (bool): Optional. Defaults to False. Streams detected
              anomalies to Cloud Logging. The anomalies will be put into json
              payload encoded from proto
              [google.cloud.aiplatform.logging.ModelMonitoringAnomaliesLogEntry][].
              This can be further sync'd to Pub/Sub or any other services supported
              by Cloud Logging.
            notification_channels (List[str]): The Cloud notification channels to
              send the alert to.
        """
        self.user_emails = user_emails
        self.enable_logging = enable_logging
        self.notification_channels = notification_channels
        self._config_for_bp = False

    def as_proto(self) -> gca_model_monitoring.ModelMonitoringAlertConfig:
        """Converts AlertConfig to a proto message.

        Returns:
            The GAPIC representation of the alert config.
        """
        # TODO(b/242108750): remove temporary logic once model monitoring for
        # batch prediction is GA.
        if self._config_for_bp:
            gca_model_monitoring = gca_model_monitoring_v1beta1
        else:
            gca_model_monitoring = gca_model_monitoring_v1

        return gca_model_monitoring.ModelMonitoringAlertConfig(
            email_alert_config=gca_model_monitoring.ModelMonitoringAlertConfig.EmailAlertConfig(
                user_emails=self.user_emails
            ),
            enable_logging=self.enable_logging,
            notification_channels=self.notification_channels,
        )


class EmailAlertConfig(AlertConfig):
    def __init__(
        self, user_emails: List[str] = [], enable_logging: Optional[bool] = False
    ):
        """Initializer for EmailAlertConfig.

        Args:
            user_emails (List[str]): The email addresses to send the alert to.
            enable_logging (bool): Optional. Defaults to False. Streams detected
              anomalies to Cloud Logging. The anomalies will be put into json
              payload encoded from proto
              [google.cloud.aiplatform.logging.ModelMonitoringAnomaliesLogEntry][].
              This can be further sync'd to Pub/Sub or any other services supported
              by Cloud Logging.
        """
        super().__init__(user_emails=user_emails, enable_logging=enable_logging)
