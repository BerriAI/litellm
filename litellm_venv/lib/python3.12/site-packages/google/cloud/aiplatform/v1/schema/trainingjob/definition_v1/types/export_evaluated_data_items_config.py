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


__protobuf__ = proto.module(
    package="google.cloud.aiplatform.v1.schema.trainingjob.definition",
    manifest={
        "ExportEvaluatedDataItemsConfig",
    },
)


class ExportEvaluatedDataItemsConfig(proto.Message):
    r"""Configuration for exporting test set predictions to a
    BigQuery table.

    Attributes:
        destination_bigquery_uri (str):
            URI of desired destination BigQuery table. Expected format:
            bq://<project_id>:<dataset_id>:

            If not specified, then results are exported to the following
            auto-created BigQuery table:
            <project_id>:export_evaluated_examples_<model_name>_<yyyy_MM_dd'T'HH_mm_ss_SSS'Z'>.evaluated_examples
        override_existing_table (bool):
            If true and an export destination is
            specified, then the contents of the destination
            are overwritten. Otherwise, if the export
            destination already exists, then the export
            operation fails.
    """

    destination_bigquery_uri: str = proto.Field(
        proto.STRING,
        number=1,
    )
    override_existing_table: bool = proto.Field(
        proto.BOOL,
        number=2,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
