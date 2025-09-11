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

import copy
import json
from typing import Any, Dict, Mapping, Optional, Union
from google.cloud.aiplatform.compat.types import pipeline_failure_policy
import packaging.version


class PipelineRuntimeConfigBuilder(object):
    """Pipeline RuntimeConfig builder.

    Constructs a RuntimeConfig spec with pipeline_root and parameter overrides.
    """

    def __init__(
        self,
        pipeline_root: str,
        schema_version: str,
        parameter_types: Mapping[str, str],
        parameter_values: Optional[Dict[str, Any]] = None,
        input_artifacts: Optional[Dict[str, str]] = None,
        failure_policy: Optional[pipeline_failure_policy.PipelineFailurePolicy] = None,
    ):
        """Creates a PipelineRuntimeConfigBuilder object.

        Args:
          pipeline_root (str):
              Required. The root of the pipeline outputs.
          schema_version (str):
              Required. Schema version of the IR. This field determines the fields supported in current version of IR.
          parameter_types (Mapping[str, str]):
              Required. The mapping from pipeline parameter name to its type.
          parameter_values (Dict[str, Any]):
              Optional. The mapping from runtime parameter name to its value.
          input_artifacts (Dict[str, str]):
              Optional. The mapping from the runtime parameter name for this artifact to its resource id.
          failure_policy (pipeline_failure_policy.PipelineFailurePolicy):
              Optional. Represents the failure policy of a pipeline. Currently, the
              default of a pipeline is that the pipeline will continue to
              run until no more tasks can be executed, also known as
              PIPELINE_FAILURE_POLICY_FAIL_SLOW. However, if a pipeline is
              set to PIPELINE_FAILURE_POLICY_FAIL_FAST, it will stop
              scheduling any new tasks when a task has failed. Any
              scheduled tasks will continue to completion.
        """
        self._pipeline_root = pipeline_root
        self._schema_version = schema_version
        self._parameter_types = parameter_types
        self._parameter_values = copy.deepcopy(parameter_values or {})
        self._input_artifacts = copy.deepcopy(input_artifacts or {})
        self._failure_policy = failure_policy

    @classmethod
    def from_job_spec_json(
        cls,
        job_spec: Mapping[str, Any],
    ) -> "PipelineRuntimeConfigBuilder":
        """Creates a PipelineRuntimeConfigBuilder object from PipelineJob json spec.

        Args:
          job_spec (Mapping[str, Any]):
              Required. The PipelineJob spec.

        Returns:
          A PipelineRuntimeConfigBuilder object.
        """
        runtime_config_spec = job_spec["runtimeConfig"]
        parameter_input_definitions = (
            job_spec["pipelineSpec"]["root"]
            .get("inputDefinitions", {})
            .get("parameters", {})
        )
        schema_version = job_spec["pipelineSpec"]["schemaVersion"]

        # 'type' is deprecated in IR and change to 'parameterType'.
        parameter_types = {
            k: v.get("parameterType") or v.get("type")
            for k, v in parameter_input_definitions.items()
        }

        pipeline_root = runtime_config_spec.get("gcsOutputDirectory")
        parameter_values = _parse_runtime_parameters(runtime_config_spec)
        failure_policy = runtime_config_spec.get("failurePolicy")
        return cls(
            pipeline_root=pipeline_root,
            schema_version=schema_version,
            parameter_types=parameter_types,
            parameter_values=parameter_values,
            failure_policy=failure_policy,
        )

    def update_pipeline_root(self, pipeline_root: Optional[str]) -> None:
        """Updates pipeline_root value.

        Args:
          pipeline_root (str):
              Optional. The root of the pipeline outputs.
        """
        if pipeline_root:
            self._pipeline_root = pipeline_root

    def update_runtime_parameters(
        self, parameter_values: Optional[Mapping[str, Any]] = None
    ) -> None:
        """Merges runtime parameter values.

        Args:
          parameter_values (Mapping[str, Any]):
              Optional. The mapping from runtime parameter names to its values.
        """
        if parameter_values:
            parameters = dict(parameter_values)
            if packaging.version.parse(self._schema_version) <= packaging.version.parse(
                "2.0.0"
            ):
                for k, v in parameter_values.items():
                    if isinstance(v, (dict, list, bool)):
                        parameters[k] = json.dumps(v)
            self._parameter_values.update(parameters)

    def update_input_artifacts(
        self, input_artifacts: Optional[Mapping[str, str]]
    ) -> None:
        """Merges runtime input artifacts.

        Args:
          input_artifacts (Mapping[str, str]):
              Optional. The mapping from the runtime parameter name for this artifact to its resource id.
        """
        if input_artifacts:
            self._input_artifacts.update(input_artifacts)

    def update_failure_policy(self, failure_policy: Optional[str] = None) -> None:
        """Merges runtime failure policy.

        Args:
          failure_policy (str):
              Optional. The failure policy - "slow" or "fast".

        Raises:
          ValueError: if failure_policy is not valid.
        """
        if failure_policy:
            if failure_policy in _FAILURE_POLICY_TO_ENUM_VALUE:
                self._failure_policy = _FAILURE_POLICY_TO_ENUM_VALUE[failure_policy]
            else:
                raise ValueError(
                    f'failure_policy should be either "slow" or "fast", but got: "{failure_policy}".'
                )

    def build(self) -> Dict[str, Any]:
        """Build a RuntimeConfig proto.

        Raises:
          ValueError: if the pipeline root is not specified.
        """
        if not self._pipeline_root:
            raise ValueError(
                "Pipeline root must be specified, either during "
                "compile time, or when calling the service."
            )
        if packaging.version.parse(self._schema_version) > packaging.version.parse(
            "2.0.0"
        ):
            parameter_values_key = "parameterValues"
        else:
            parameter_values_key = "parameters"

        runtime_config = {
            "gcsOutputDirectory": self._pipeline_root,
            parameter_values_key: {
                k: self._get_vertex_value(k, v)
                for k, v in self._parameter_values.items()
                if v is not None
            },
            "inputArtifacts": {
                k: {"artifactId": v} for k, v in self._input_artifacts.items()
            },
        }

        if self._failure_policy:
            runtime_config["failurePolicy"] = self._failure_policy

        return runtime_config

    def _get_vertex_value(
        self, name: str, value: Union[int, float, str, bool, list, dict]
    ) -> Union[int, float, str, bool, list, dict]:
        """Converts primitive values into Vertex pipeline Value proto message.

        Args:
          name (str):
              Required. The name of the pipeline parameter.
          value (Union[int, float, str, bool, list, dict]):
              Required. The value of the pipeline parameter.

        Returns:
          A dictionary represents the Vertex pipeline Value proto message.

        Raises:
          ValueError: if the parameter name is not found in pipeline root
          inputs, or value is none.
        """
        if value is None:
            raise ValueError("None values should be filtered out.")

        if name not in self._parameter_types:
            raise ValueError(
                "The pipeline parameter {} is not found in the "
                "pipeline job input definitions.".format(name)
            )

        if packaging.version.parse(self._schema_version) <= packaging.version.parse(
            "2.0.0"
        ):
            result = {}
            if self._parameter_types[name] == "INT":
                result["intValue"] = value
            elif self._parameter_types[name] == "DOUBLE":
                result["doubleValue"] = value
            elif self._parameter_types[name] == "STRING":
                result["stringValue"] = value
            else:
                raise TypeError("Got unknown type of value: {}".format(value))
            return result
        else:
            return value


def _parse_runtime_parameters(
    runtime_config_spec: Mapping[str, Any]
) -> Optional[Dict[str, Any]]:
    """Extracts runtime parameters from runtime config json spec.

    Raises:
        TypeError: if the parameter type is not one of 'INT', 'DOUBLE', 'STRING'.
    """
    # 'parameters' are deprecated in IR and changed to 'parameterValues'.
    if runtime_config_spec.get("parameterValues") is not None:
        return runtime_config_spec.get("parameterValues")

    if runtime_config_spec.get("parameters") is not None:
        result = {}
        for name, value in runtime_config_spec.get("parameters").items():
            if "intValue" in value:
                result[name] = int(value["intValue"])
            elif "doubleValue" in value:
                result[name] = float(value["doubleValue"])
            elif "stringValue" in value:
                result[name] = value["stringValue"]
            else:
                raise TypeError("Got unknown type of value: {}".format(value))
        return result


_FAILURE_POLICY_TO_ENUM_VALUE = {
    "slow": pipeline_failure_policy.PipelineFailurePolicy.PIPELINE_FAILURE_POLICY_FAIL_SLOW,
    "fast": pipeline_failure_policy.PipelineFailurePolicy.PIPELINE_FAILURE_POLICY_FAIL_FAST,
    None: pipeline_failure_policy.PipelineFailurePolicy.PIPELINE_FAILURE_POLICY_UNSPECIFIED,
}
