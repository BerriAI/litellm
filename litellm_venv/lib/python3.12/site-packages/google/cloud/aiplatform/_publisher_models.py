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

import re
from typing import Optional

from google.auth import credentials as auth_credentials
from google.cloud.aiplatform import base
from google.cloud.aiplatform import utils


class _PublisherModel(base.VertexAiResourceNoun):
    """Publisher Model Resource for Vertex AI."""

    client_class = utils.ModelGardenClientWithOverride

    _resource_noun = "publisher_model"
    _getter_method = "get_publisher_model"
    _delete_method = None
    _parse_resource_name_method = "parse_publisher_model_path"
    _format_resource_name_method = "publisher_model_path"

    def __init__(
        self,
        resource_name: str,
        project: Optional[str] = None,
        location: Optional[str] = None,
        credentials: Optional[auth_credentials.Credentials] = None,
    ):
        """Retrieves an existing PublisherModel resource given a resource name or model garden id.

        Args:
            resource_name (str):
                Required. A fully-qualified PublisherModel resource name or
                model garden id. Format:
                `publishers/{publisher}/models/{publisher_model}` or
                `{publisher}/{publisher_model}`.
            project (str):
                Optional. Project to retrieve the resource from. If not set,
                project set in aiplatform.init will be used.
            location (str):
                Optional. Location to retrieve the resource from. If not set,
                location set in aiplatform.init will be used.
            credentials (auth_credentials.Credentials):
                Optional. Custom credentials to use to retrieve the resource.
                Overrides credentials set in aiplatform.init.
        """

        super().__init__(project=project, location=location, credentials=credentials)

        if self._parse_resource_name(resource_name):
            full_resource_name = resource_name
        else:
            m = re.match(r"^(?P<publisher>.+?)/(?P<model>.+?)$", resource_name)
            if m:
                full_resource_name = self._format_resource_name(**m.groupdict())
            else:
                raise ValueError(
                    f"`{resource_name}` is not a valid PublisherModel resource "
                    "name or model garden id."
                )

        self._gca_resource = getattr(self.api_client, self._getter_method)(
            name=full_resource_name, retry=base._DEFAULT_RETRY
        )
