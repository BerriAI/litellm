# -*- coding: utf-8 -*-

# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from google.protobuf import json_format
from typing import Any, Dict, List, Optional

from google.cloud.aiplatform.compat.types import explanation_metadata
from google.cloud.aiplatform.explain.metadata import metadata_builder


class SavedModelMetadataBuilder(metadata_builder.MetadataBuilder):
    """Metadata builder class that accepts a TF1 saved model."""

    def __init__(
        self,
        model_path: str,
        tags: Optional[List[str]] = None,
        signature_name: Optional[str] = None,
        outputs_to_explain: Optional[List[str]] = None,
    ) -> None:
        """Initializes a SavedModelMetadataBuilder object.

        Args:
          model_path:
              Required. Local or GCS path to load the saved model from.
          tags:
              Optional. Tags to identify the model graph. If None or empty,
              TensorFlow's default serving tag will be used.
          signature_name:
              Optional. Name of the signature to be explained. Inputs and
              outputs of this signature will be written in the metadata. If not
              provided, the default signature will be used.
          outputs_to_explain:
              Optional. List of output names to explain. Only single output is
              supported for now. Hence, the list should contain one element.
              This parameter is required if the model signature (provided via
              signature_name) specifies multiple outputs.

        Raises:
            ValueError: If outputs_to_explain contains more than 1 element or
            signature contains multiple outputs.
        """
        if outputs_to_explain:
            if len(outputs_to_explain) > 1:
                raise ValueError(
                    "Only one output is supported at the moment. "
                    f"Received: {outputs_to_explain}."
                )
            self._output_to_explain = next(iter(outputs_to_explain))

        try:
            import tensorflow.compat.v1 as tf
        except ImportError:
            raise ImportError(
                "Tensorflow is not installed and is required to load saved model. "
                'Please install the SDK using "pip install "tensorflow>=1.15,<2.0""'
            )

        if not signature_name:
            signature_name = tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY
        self._tags = tags or [tf.saved_model.tag_constants.SERVING]
        self._graph = tf.Graph()

        with self.graph.as_default():
            self._session = tf.Session(graph=self.graph)
            self._metagraph_def = tf.saved_model.loader.load(
                sess=self.session, tags=self._tags, export_dir=model_path
            )
            if signature_name not in self._metagraph_def.signature_def:
                raise ValueError(
                    f"Serving sigdef key {signature_name} not in the signature def."
                )
            serving_sigdef = self._metagraph_def.signature_def[signature_name]
        if not outputs_to_explain:
            if len(serving_sigdef.outputs) > 1:
                raise ValueError(
                    "The signature contains multiple outputs. Specify "
                    'an output via "outputs_to_explain" parameter.'
                )
            self._output_to_explain = next(iter(serving_sigdef.outputs.keys()))

        self._inputs = _create_input_metadata_from_signature(serving_sigdef.inputs)
        self._outputs = _create_output_metadata_from_signature(
            serving_sigdef.outputs, self._output_to_explain
        )

    @property
    def graph(self) -> "tf.Graph":  # noqa: F821
        return self._graph

    @property
    def session(self) -> "tf.Session":  # noqa: F821
        return self._session

    def get_metadata(self) -> Dict[str, Any]:
        """Returns the current metadata as a dictionary.

        Returns:
            Json format of the explanation metadata.
        """
        return json_format.MessageToDict(self.get_metadata_protobuf()._pb)

    def get_metadata_protobuf(self) -> explanation_metadata.ExplanationMetadata:
        """Returns the current metadata as a Protobuf object.

        Returns:
            ExplanationMetadata object format of the explanation metadata.
        """
        return explanation_metadata.ExplanationMetadata(
            inputs=self._inputs,
            outputs=self._outputs,
        )


def _create_input_metadata_from_signature(
    signature_inputs: Dict[str, "tf.Tensor"]  # noqa: F821
) -> Dict[str, explanation_metadata.ExplanationMetadata.InputMetadata]:
    """Creates InputMetadata from signature inputs.

    Args:
      signature_inputs:
          Required. Inputs of the signature to be explained. If not provided,
          the default signature will be used.

    Returns:
          Inferred input metadata from the model.
    """
    input_mds = {}
    for key, tensor in signature_inputs.items():
        input_mds[key] = explanation_metadata.ExplanationMetadata.InputMetadata(
            input_tensor_name=tensor.name
        )
    return input_mds


def _create_output_metadata_from_signature(
    signature_outputs: Dict[str, "tf.Tensor"],  # noqa: F821
    output_to_explain: Optional[str] = None,
) -> Dict[str, explanation_metadata.ExplanationMetadata.OutputMetadata]:
    """Creates OutputMetadata from signature inputs.

    Args:
      signature_outputs:
          Required. Inputs of the signature to be explained. If not provided,
          the default signature will be used.
      output_to_explain:
          Optional. Output name to explain.

    Returns:
          Inferred output metadata from the model.
    """
    output_mds = {}
    for key, tensor in signature_outputs.items():
        if not output_to_explain or output_to_explain == key:
            output_mds[key] = explanation_metadata.ExplanationMetadata.OutputMetadata(
                output_tensor_name=tensor.name
            )
    return output_mds
