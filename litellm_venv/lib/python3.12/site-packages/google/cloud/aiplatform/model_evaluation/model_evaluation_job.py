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

from typing import Optional, List, Union

from google.auth import credentials as auth_credentials
import grpc

from google.cloud import aiplatform
from google.cloud.aiplatform import base
from google.cloud.aiplatform import initializer
from google.cloud.aiplatform._pipeline_based_service import (
    pipeline_based_service,
)
from google.cloud.aiplatform import model_evaluation
from google.cloud.aiplatform import pipeline_jobs
from google.cloud.aiplatform.utils import _ipython_utils

from google.cloud.aiplatform.compat.types import (
    pipeline_state_v1 as gca_pipeline_state_v1,
    pipeline_job_v1 as gca_pipeline_job_v1,
    execution_v1 as gca_execution_v1,
)

_LOGGER = base.Logger(__name__)

_PIPELINE_TEMPLATE_ARTIFACT_REGISTRY_TAG = "1.0.0"
_BASE_URI = (
    "base_uri",
    "https://us-kfp.pkg.dev/vertex-evaluation/pipeline-templates/evaluation",
)
_TAG = ("tag", _PIPELINE_TEMPLATE_ARTIFACT_REGISTRY_TAG)
_MODEL_EVAL_TEMPLATE_REF = frozenset((_BASE_URI, _TAG))


class _ModelEvaluationJob(pipeline_based_service._VertexAiPipelineBasedService):
    """Creates a Model Evaluation PipelineJob using _VertexAiPipelineBasedService."""

    _template_ref = _MODEL_EVAL_TEMPLATE_REF

    _creation_log_message = "Created PipelineJob for your Model Evaluation."

    _component_identifier = "fpc-model-evaluation"

    _template_name_identifier = None

    @property
    def _metadata_output_artifact(self) -> Optional[str]:
        """The resource uri for the ML Metadata output artifact from the evaluation component of the Model Evaluation pipeline"""
        if self.state != gca_pipeline_state_v1.PipelineState.PIPELINE_STATE_SUCCEEDED:
            return
        for task in self.backing_pipeline_job._gca_resource.job_detail.task_details:
            if task.task_name == self.backing_pipeline_job.name:
                return task.outputs["evaluation_metrics"].artifacts[0].name

    def __init__(
        self,
        evaluation_pipeline_run_name: str,
        project: Optional[str] = None,
        location: Optional[str] = None,
        credentials: Optional[auth_credentials.Credentials] = None,
    ):
        """Retrieves a ModelEvaluationJob and instantiates its representation.
        Example Usage:
            my_evaluation = aiplatform.ModelEvaluationJob(
                pipeline_job_name = "projects/123/locations/us-central1/pipelineJobs/456"
            )
            my_evaluation = aiplatform.ModelEvaluationJob(
                pipeline_job_name = "456"
            )
        Args:
            evaluation_pipeline_run_name (str):
                Required. A fully-qualified pipeline job run ID.
                Example: "projects/123/locations/us-central1/pipelineJobs/456" or
                "456" when project and location are initialized or passed.
            project (str):
                Optional. Project to retrieve pipeline job from. If not set, project
                set in aiplatform.init will be used.
            location (str):
                Optional. Location to retrieve pipeline job from. If not set, location
                set in aiplatform.init will be used.
            credentials (auth_credentials.Credentials):
                Optional. Custom credentials to use to retrieve this pipeline job. Overrides
                credentials set in aiplatform.init.
        """
        super().__init__(
            pipeline_job_name=evaluation_pipeline_run_name,
            project=project,
            location=location,
            credentials=credentials,
        )

    @staticmethod
    def _get_template_url(
        model_type: str,
        feature_attributions: bool,
        prediction_type: str,
    ) -> str:
        """Gets the pipeline template URL for this model evaluation job given the type of data
        used to train the model and whether feature attributions should be generated.

        Args:
            model_type (str):
                Required. Whether the model is an AutoML Tabular model or not. Used to determine which pipeline template should be used.
            feature_attributions (bool):
                Required. Whether this evaluation job should generate feature attributions.
            prediction_type (str):
                Required. The type of prediction performed by the Model. One of "classification" or "regression".

        Returns:
            (str): The pipeline template URL to use for this model evaluation job.
        """

        # Examples of formatted template URIs:
        # model_type="automl_tabular", feature_attrubtions=True, prediction_type="classification"
        # https://us-kfp.pkg.dev/vertex-evaluation/pipeline-templates/evaluation-automl-tabular-feature-attribution-classification-pipeline/1.0.0
        # model_type="other", feature_attributions=False, prediction_type="regression"
        # https://us-kfp.pkg.dev/vertex-evaluation/pipeline-templates/evaluation-regression-pipeline/1.0.0
        model_type_uri_str = "automl-tabular" if model_type == "automl_tabular" else ""
        feature_attributions_uri_str = (
            "feature-attribution" if feature_attributions else ""
        )

        template_ref_dict = dict(_ModelEvaluationJob._template_ref)

        uri_parts = [
            template_ref_dict["base_uri"],
            model_type_uri_str,
            feature_attributions_uri_str,
            prediction_type,
            "pipeline/" + template_ref_dict["tag"],
        ]
        template_url = "-".join(filter(None, uri_parts))

        return template_url

    @classmethod
    def submit(
        cls,
        model_name: Union[str, "aiplatform.Model"],
        prediction_type: str,
        target_field_name: str,
        pipeline_root: str,
        model_type: str,
        gcs_source_uris: Optional[List[str]] = None,
        bigquery_source_uri: Optional[str] = None,
        batch_predict_bigquery_destination_output_uri: Optional[str] = None,
        class_labels: Optional[List[str]] = None,
        prediction_label_column: Optional[str] = None,
        prediction_score_column: Optional[str] = None,
        generate_feature_attributions: Optional[bool] = False,
        instances_format: Optional[str] = "jsonl",
        evaluation_pipeline_display_name: Optional[str] = None,
        evaluation_metrics_display_name: Optional[str] = None,
        job_id: Optional[str] = None,
        service_account: Optional[str] = None,
        network: Optional[str] = None,
        encryption_spec_key_name: Optional[str] = None,
        project: Optional[str] = None,
        location: Optional[str] = None,
        credentials: Optional[auth_credentials.Credentials] = None,
        experiment: Optional[Union[str, "aiplatform.Experiment"]] = None,
    ) -> "_ModelEvaluationJob":
        """Submits a Model Evaluation Job using aiplatform.PipelineJob and returns
        the ModelEvaluationJob resource.

        Example usage:

        ```
        my_evaluation = _ModelEvaluationJob.submit(
            model="projects/123/locations/us-central1/models/456",
            prediction_type="classification",
            pipeline_root="gs://my-pipeline-bucket/runpath",
            gcs_source_uris=["gs://test-prediction-data"],
            target_field_name=["prediction_class"],
            instances_format="jsonl",
        )

        my_evaluation = _ModelEvaluationJob.submit(
            model="projects/123/locations/us-central1/models/456",
            prediction_type="regression",
            pipeline_root="gs://my-pipeline-bucket/runpath",
            gcs_source_uris=["gs://test-prediction-data"],
            target_field_name=["price"],
            instances_format="jsonl",
        )
        ```

        Args:
            model_name (Union[str, "aiplatform.Model"]):
                Required. An instance of aiplatform.Model or a fully-qualified model resource name or model ID to run the evaluation
                job on. Example: "projects/123/locations/us-central1/models/456" or
                "456" when project and location are initialized or passed.
            prediction_type (str):
                Required. The type of prediction performed by the Model. One of "classification" or "regression".
            target_field_name (str):
                Required. The name of your prediction column.
            pipeline_root (str):
                Required. The GCS directory to store output from the model evaluation PipelineJob.
            model_type (str):
                Required. One of "automl_tabular" or "other". This determines the Model Evaluation template used by this PipelineJob.
            gcs_source_uris (List[str]):
                Optional. A list of Cloud Storage data files containing the ground truth data to use for this
                evaluation job, for example: ["gs://path/to/your/data.csv"]. These files should contain your
                model's prediction column. The provided data files must be either CSV or JSONL. One of `gcs_source_uris`
                or `bigquery_source_uri` is required.
            bigquery_source_uri (str):
                Optional. A bigquery table URI containing the ground truth data to use for this evaluation job. This uri should
                be in the format 'bq://my-project-id.dataset.table'. One of `gcs_source_uris` or `bigquery_source_uri` is
                required.
            bigquery_destination_output_uri (str):
                Optional. A bigquery table URI where the Batch Prediction job associated with your Model Evaluation will write
                prediction output. This can be a BigQuery URI to a project ('bq://my-project'), a dataset
                ('bq://my-project.my-dataset'), or a table ('bq://my-project.my-dataset.my-table'). Required if `bigquery_source_uri`
                is provided.
            class_labels (List[str]):
                Optional. For custom (non-AutoML) classification models, a list of possible class names, in the
                same order that predictions are generated. This argument is required when prediction_type is 'classification'.
                For example, in a classification model with 3 possible classes that are outputted in the format: [0.97, 0.02, 0.01]
                with the class names "cat", "dog", and "fish", the value of `class_labels` should be `["cat", "dog", "fish"]` where
                the class "cat" corresponds with 0.97 in the example above.
            prediction_label_column (str):
                Optional. The column name of the field containing classes the model is scoring. Formatted to be able to find nested
                columns, delimeted by `.`. If not set, defaulted to `prediction.classes` for classification.
            prediction_score_column (str):
                Optional. The column name of the field containing batch prediction scores. Formatted to be able to find nested columns,
                delimeted by `.`. If not set, defaulted to `prediction.scores` for a `classification` problem_type, `prediction.value`
                for a `regression` problem_type.
            generate_feature_attributions (boolean):
                Optional. Whether the model evaluation job should generate feature attributions. Defaults to False if not specified.
            instances_format (str):
                The format in which instances are given, must be one of the Model's supportedInputStorageFormats. If not set, defaults to "jsonl".
            evaluation_pipeline_display_name (str)
                Optional. The user-defined name of the PipelineJob created by this Pipeline Based Service.
            evaluation_metrics_display_name (str)
                Optional. The user-defined name of the evaluation metrics resource uploaded to Vertex in the evaluation pipeline job.
            job_id (str):
                Optional. The unique ID of the job run. If not specified, pipeline name + timestamp will be used.
            service_account (str):
                Specifies the service account for workload run-as account for this Model Evaluation PipelineJob.
                Users submitting jobs must have act-as permission on this run-as account. The service account running
                this Model Evaluation job needs the following permissions: Dataflow Worker, Storage Admin, Vertex AI User.
            network (str):
                The full name of the Compute Engine network to which the job
                should be peered. For example, projects/12345/global/networks/myVPC.
                Private services access must already be configured for the network.
                If left unspecified, the job is not peered with any network.
            encryption_spec_key_name (str):
                Optional. The Cloud KMS resource identifier of the customer managed encryption key used to protect the job. Has the
                form: ``projects/my-project/locations/my-region/keyRings/my-kr/cryptoKeys/my-key``. The key needs to be in the same
                region as where the compute resource is created. If this is set, then all
                resources created by the PipelineJob for this Model Evaluation will be encrypted with the provided encryption key.
                If not specified, encryption_spec of original PipelineJob will be used.
            project (str):
                Optional. The project to run this PipelineJob in. If not set,
                the project set in aiplatform.init will be used.
            location (str):
                Optional. Location to create PipelineJob. If not set,
                location set in aiplatform.init will be used.
            credentials (auth_credentials.Credentials):
                Optional. Custom credentials to use to create the PipelineJob.
                Overrides credentials set in aiplatform.init.
            experiment (Union[str, experiments_resource.Experiment]):
                Optional. The Vertex AI experiment name or instance to associate to the PipelineJob executing
                this model evaluation job.
        Returns:
            (ModelEvaluationJob): Instantiated represnetation of the model evaluation job.
        """
        service_account = service_account or initializer.global_config.service_account

        if isinstance(model_name, aiplatform.Model):
            model_resource_name = model_name.versioned_resource_name
        else:
            model_resource_name = aiplatform.Model(
                model_name=model_name,
                project=project,
                location=location,
                credentials=credentials,
            ).versioned_resource_name

        if not evaluation_pipeline_display_name:
            evaluation_pipeline_display_name = cls._generate_display_name()

        template_params = {
            "batch_predict_instances_format": instances_format,
            "model_name": model_resource_name,
            "evaluation_display_name": evaluation_metrics_display_name,
            "project": project or initializer.global_config.project,
            "location": location or initializer.global_config.location,
            "batch_predict_gcs_destination_output_uri": pipeline_root,
            "target_field_name": target_field_name,
            "encryption_spec_key_name": encryption_spec_key_name,
        }

        if bigquery_source_uri:
            template_params["batch_predict_predictions_format"] = "bigquery"
            template_params["batch_predict_bigquery_source_uri"] = bigquery_source_uri
            template_params[
                "batch_predict_bigquery_destination_output_uri"
            ] = batch_predict_bigquery_destination_output_uri
        elif gcs_source_uris:
            template_params["batch_predict_gcs_source_uris"] = gcs_source_uris

        if prediction_type == "classification" and model_type == "other":
            template_params["evaluation_class_labels"] = class_labels

        if prediction_label_column:
            template_params[
                "evaluation_prediction_label_column"
            ] = prediction_label_column

        if prediction_score_column:
            template_params[
                "evaluation_prediction_score_column"
            ] = prediction_score_column

        # If the user provides a SA, use it for the Dataflow job as well
        if service_account is not None:
            template_params["dataflow_service_account"] = service_account

        template_url = cls._get_template_url(
            model_type,
            generate_feature_attributions,
            prediction_type,
        )

        eval_pipeline_run = cls._create_and_submit_pipeline_job(
            template_params=template_params,
            template_path=template_url,
            pipeline_root=pipeline_root,
            display_name=evaluation_pipeline_display_name,
            job_id=job_id,
            service_account=service_account,
            network=network,
            encryption_spec_key_name=encryption_spec_key_name,
            project=project,
            location=location,
            credentials=credentials,
            experiment=experiment,
        )

        _LOGGER.info(
            f"{_ModelEvaluationJob._creation_log_message} View it in the console: {eval_pipeline_run.pipeline_console_uri}"
        )

        return eval_pipeline_run

    def get_model_evaluation(
        self,
    ) -> Optional["model_evaluation.ModelEvaluation"]:
        """Gets the ModelEvaluation created by this ModelEvlauationJob.

        Returns:
            aiplatform.ModelEvaluation: Instantiated representation of the ModelEvaluation resource.
        Raises:
            RuntimeError: If the ModelEvaluationJob pipeline failed.
        """
        eval_job_state = self.backing_pipeline_job.state

        if eval_job_state in pipeline_jobs._PIPELINE_ERROR_STATES:
            raise RuntimeError(
                f"Evaluation job failed. For more details see the logs: {self.pipeline_console_uri}"
            )
        if eval_job_state not in pipeline_jobs._PIPELINE_COMPLETE_STATES:
            _LOGGER.info(
                f"Your evaluation job is still in progress. For more details see the logs {self.pipeline_console_uri}"
            )
            return

        for component in self.backing_pipeline_job.task_details:
            # This assumes that task_details has a task with a task_name == backing_pipeline_job.name
            if not component.task_name == self.backing_pipeline_job.name:
                continue

            # If component execution didn't succeed or the execution wasn't cached, don't return an evaluation
            if (
                component.state
                not in (
                    gca_pipeline_job_v1.PipelineTaskDetail.State.SUCCEEDED,
                    gca_pipeline_job_v1.PipelineTaskDetail.State.SKIPPED,
                )
                and component.execution.state != gca_execution_v1.Execution.State.CACHED
            ):
                continue

            if "output:evaluation_resource_name" not in component.execution.metadata:
                continue

            eval_resource_name = component.execution.metadata[
                "output:evaluation_resource_name"
            ]

            eval_resource = model_evaluation.ModelEvaluation(
                evaluation_name=eval_resource_name,
                credentials=self.credentials,
            )
            _ipython_utils.display_model_evaluation_button(eval_resource)
            return eval_resource

    def wait(self) -> None:
        """Wait for the PipelineJob to complete, then get the model evaluation resource."""
        super().wait()

        try:
            self.get_model_evaluation()
        except grpc.RpcError as e:
            _LOGGER.error("Get model evaluation call failed with error %s", e)
