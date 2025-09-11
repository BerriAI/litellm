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
from typing import Optional, List, Dict, Any, Tuple

from google.cloud.aiplatform.explain.metadata import metadata_builder
from google.cloud.aiplatform.compat.types import explanation_metadata


class SavedModelMetadataBuilder(metadata_builder.MetadataBuilder):
    """Class for generating metadata for a model built with TF 2.X Keras API."""

    def __init__(
        self,
        model_path: str,
        signature_name: Optional[str] = None,
        outputs_to_explain: Optional[List[str]] = None,
        **kwargs
    ) -> None:
        """Initializes a SavedModelMetadataBuilder object.

        Args:
          model_path:
              Required. Local or GCS path to load the saved model from.
          signature_name:
              Optional. Name of the signature to be explained. Inputs and
              outputs of this signature will be written in the metadata. If not
              provided, the default signature will be used.
          outputs_to_explain:
              Optional. List of output names to explain. Only single output is
              supported for now. Hence, the list should contain one element.
              This parameter is required if the model signature (provided via
              signature_name) specifies multiple outputs.
          **kwargs:
              Any keyword arguments to be passed to tf.saved_model.save() function.

        Raises:
            ValueError: If outputs_to_explain contains more than 1 element.
            ImportError: If tf is not imported.
        """
        if outputs_to_explain and len(outputs_to_explain) > 1:
            raise ValueError(
                '"outputs_to_explain" can only contain 1 element.\n'
                "Got: %s" % len(outputs_to_explain)
            )
        self._explain_output = outputs_to_explain
        self._saved_model_args = kwargs

        try:
            import tensorflow as tf
        except ImportError:
            raise ImportError(
                "Tensorflow is not installed and is required to load saved model. "
                'Please install the SDK using "pip install google-cloud-aiplatform[full]"'
            )

        if not signature_name:
            signature_name = tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY
        self._loaded_model = tf.saved_model.load(model_path)
        self._inputs, self._outputs = self._infer_metadata_entries_from_model(
            signature_name
        )

    def _infer_metadata_entries_from_model(
        self, signature_name: str
    ) -> Tuple[
        Dict[str, explanation_metadata.ExplanationMetadata.InputMetadata],
        Dict[str, explanation_metadata.ExplanationMetadata.OutputMetadata],
    ]:
        """Infers metadata inputs and outputs.

        Args:
          signature_name:
              Required. Name of the signature to be explained. Inputs and outputs of this signature will be written in the metadata. If not provided, the default signature will be used.

        Returns:
              Inferred input metadata and output metadata from the model.

        Raises:
              ValueError: If specified name is not found in signature outputs.
        """

        loaded_sig = self._loaded_model.signatures[signature_name]
        _, input_sig = loaded_sig.structured_input_signature
        output_sig = loaded_sig.structured_outputs
        input_mds = {}
        for name, tensor_spec in input_sig.items():
            input_mds[name] = explanation_metadata.ExplanationMetadata.InputMetadata(
                input_tensor_name=name,
                modality=None if tensor_spec.dtype.is_floating else "categorical",
            )

        output_mds = {}
        for name in output_sig:
            if not self._explain_output or self._explain_output[0] == name:
                output_mds[
                    name
                ] = explanation_metadata.ExplanationMetadata.OutputMetadata(
                    output_tensor_name=name,
                )
                break
        else:
            raise ValueError(
                "Specified output name cannot be found in given signature outputs."
            )
        return input_mds, output_mds

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
