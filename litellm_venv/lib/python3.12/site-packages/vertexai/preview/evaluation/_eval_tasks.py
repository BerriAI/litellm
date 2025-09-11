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
from typing import Any, Callable, Dict, List, Literal, Optional, TYPE_CHECKING, Union
import uuid

from google.api_core import exceptions
import vertexai
from google.cloud.aiplatform import base
from google.cloud.aiplatform.metadata import metadata
from vertexai import generative_models
from vertexai.preview.evaluation import _base as eval_base
from vertexai.preview.evaluation import _evaluation
from vertexai.preview.evaluation import utils
from vertexai.preview.evaluation.metrics import (
    _base as metrics_base,
)

if TYPE_CHECKING:
    import pandas as pd

# pylint: disable=g-import-not-at-top
try:
    from IPython import display as IPython_display
except ImportError:
    IPython_display = None

_LOGGER = base.Logger(__name__)

EvalResult = eval_base.EvalResult
GenerativeModel = generative_models.GenerativeModel


class EvalTask:
    """A class representing an EvalTask.

    An Evaluation Tasks is defined to measure the model's ability to perform a
    certain task in response to specific prompts or inputs. Evaluation tasks must
    contain an evaluation dataset, and a list of metrics to evaluate. Evaluation
    tasks help developers compare propmpt templates, track experiments, compare
    models and their settings, and assess the quality of the model's generated
    text.

    Dataset details:
        Default dataset column names:
            * content_column_name: "content"
            * reference_column_name: "reference"
            * response_column_name: "response"
        Requirement for different use cases:
          * Bring your own prediction: A `response` column is required. Response
              column name can be customized by providing `response_column_name`
              parameter.
          * Without prompt template: A column representing the input prompt to the
              model is required. If `content_column_name` is not specified, the
              eval dataset requires `content` column by default. The response
              column is not used if present and new responses from the model are
              generated with the content column and used for evaluation.
          * With prompt template: Dataset must contain column names corresponding to
              the placeholder names in the prompt template. For example, if prompt
              template is "Instruction: {instruction}, context: {context}", the
              dataset must contain `instruction` and `context` column.

    Metrics Details:
        The supported metrics, metric bundle descriptions, grading rubrics, and
        the required input fields can be found on the Vertex AI public
        documentation.

    Usage:
        1. To perform bring your own prediction evaluation, provide the model
        responses in the response column in the dataset. The response column name
        is "response" by default, or specify `response_column_name` parameter to
        customize.

          ```
          eval_dataset = pd.DataFrame({
                  "reference": [...],
                  "response" : [...],
          })
          eval_task = EvalTask(
            dataset=eval_dataset,
            metrics=["bleu", "rouge_l_sum", "coherence", "fluency"],
            experiment="my-experiment",
          )
          eval_result = eval_task.evaluate(
                experiment_run_name="eval-experiment-run"
          )
          ```

        2. To perform evaluation with built-in Gemini model inference, specify the
        `model` parameter with a GenerativeModel instance.  The default query
        column name to the model is `content`.

          ```
          eval_dataset = pd.DataFrame({
                "reference": [...],
                "content"  : [...],
          })
          result = EvalTask(
              dataset=eval_dataset,
              metrics=["exact_match", "bleu", "rouge_1", "rouge_2",
              "rouge_l_sum"],
              experiment="my-experiment",
          ).evaluate(
              model=GenerativeModel("gemini-pro"),
              experiment_run_name="gemini-pro-eval-run"
          )
          ```

        3. If a `prompt_template` is specified, the `content` column is not required.
        Prompts can be assembled from the evaluation dataset, and all placeholder
        names must be present in the dataset columns.
          ```
          eval_dataset = pd.DataFrame({
              "context"    : [...],
              "instruction": [...],
              "reference"  : [...],
          })
          result = EvalTask(
              dataset=eval_dataset,
              metrics=["summarization_quality"],
          ).evaluate(
              model=model,
              prompt_template="{instruction}. Article: {context}. Summary:",
          )
          ```

        4. To perform evaluation with custom model inference, specify the `model`
        parameter with a custom prediction function. The `content` column in the
        dataset is used to generate predictions with the custom model function for
        evaluation.

          ```
          def custom_model_fn(input: str) -> str:
            response = client.chat.completions.create(
              model="gpt-3.5-turbo",
              messages=[
                {"role": "user", "content": input}
              ]
            )
            return response.choices[0].message.content

          eval_dataset = pd.DataFrame({
                "content"  : [...],
                "reference": [...],
          })
          result = EvalTask(
              dataset=eval_dataset,
              metrics=["text_generation_similarity","text_generation_quality"],
              experiment="my-experiment",
          ).evaluate(
              model=custom_model_fn,
              experiment_run_name="gpt-eval-run"
          )
          ```
    """

    _resource_noun = "evalTasks"

    def __init__(
        self,
        *,
        dataset: Union["pd.DataFrame", str, Dict[str, Any]],
        metrics: List[
            Union[
                Literal[
                    "exact_match",
                    "bleu",
                    "rouge_1",
                    "rouge_2",
                    "rouge_l",
                    "rouge_l_sum",
                    "coherence",
                    "fluency",
                    "safety",
                    "groundedness",
                    "fulfillment",
                    "summarization_quality",
                    "summarization_helpfulness",
                    "summarization_verbosity",
                    "question_answering_quality",
                    "question_answering_relevance",
                    "question_answering_helpfulness",
                    "question_answering_correctness",
                    "text_generation_similarity",
                    "text_generation_quality",
                    "text_generation_instruction_following",
                    "text_generation_safety",
                    "text_generation_factuality",
                    "summarization_pointwise_reference_free",
                    "qa_pointwise_reference_free",
                    "qa_pointwise_reference_based",
                    "tool_call_quality",
                ],
                metrics_base.CustomMetric,
            ]
        ],
        experiment: Optional[str] = None,
        content_column_name: str = "content",
        reference_column_name: str = "reference",
        response_column_name: str = "response",
    ):
        """Initializes an EvalTask.

        Args:
            dataset: The dataset to be evaluated.
                Supports the following dataset formats:
                * pandas.DataFrame: Used directly for evaluation.
                * Dict: Converted to a pandas DataFrame before evaluation.
                * str: Interpreted as a file path or URI. Supported formats include:
                    * Local JSONL or CSV files:  Loaded from the local filesystem.
                    * GCS JSONL or CSV files: Loaded from Google Cloud Storage
                        (e.g., 'gs://bucket/data.csv').
                    * BigQuery table URI: Loaded from Google Cloud BigQuery
                        (e.g., 'bq://project-id.dataset.table_name').
            metrics: The list of metrics names to be evaluated, or a metrics
                bundle for an evaluation task, or custom metric instances.
            experiment: The name of the experiment to log the evaluations to.
            content_column_name: The column name of content in the dataset to send to
                the model. If not set, default to `content`.
            reference_column_name: The column name of ground truth in the dataset. If
                not set, default to `reference`.
            response_column_name: The column name of model response in the dataset. If
                not set, default to `response`.
        """
        self.dataset = utils.load_dataset(dataset)
        self.metrics = metrics
        self.experiment = experiment
        self.content_column_name = content_column_name
        self.reference_column_name = reference_column_name
        self.response_column_name = response_column_name

    def _evaluate_with_experiment(
        self,
        model: Optional[Union[GenerativeModel, Callable[[str], str]]] = None,
        prompt_template: Optional[str] = None,
        experiment_run_name: Optional[str] = None,
        response_column_name: str = "response",
    ) -> EvalResult:
        """Runs an evaluation for the EvalTask with an experiment.

        Args:
          model: A GenerativeModel instance or a custom model function to generate
            responses to evaluate. If not provided, the evaluation is computed with
            the `response` column in the `dataset`.
          prompt_template: The prompt template to use for the evaluation. If not
            set, the prompt template that was used to create the EvalTask will be
            used.
          experiment_run_name: The name of the experiment run to log the evaluation
            to if an experiment is set for this EvalTask. If not provided, a random
            unique experiment run name is used.
          response_column_name: The column name of model response in the dataset. If
            not set, default to `response`.

        Returns:
          The evaluation result.
        """
        self._validate_experiment_run()
        with vertexai.preview.start_run(experiment_run_name):
            self._log_eval_experiment_param(model, prompt_template)
            eval_result = _evaluation.evaluate(
                dataset=self.dataset,
                metrics=self.metrics,
                model=model,
                prompt_template=prompt_template,
                content_column_name=self.content_column_name,
                reference_column_name=self.reference_column_name,
                response_column_name=response_column_name or self.response_column_name,
            )
            try:
                vertexai.preview.log_metrics(eval_result.summary_metrics)
            except (ValueError, TypeError, exceptions.InvalidArgument) as e:
                _LOGGER.warning(f"Experiment metrics logging failed: {str(e)}")
        return eval_result

    def evaluate(
        self,
        *,
        model: Optional[Union[GenerativeModel, Callable[[str], str]]] = None,
        prompt_template: Optional[str] = None,
        experiment_run_name: Optional[str] = None,
        response_column_name: str = "response",
    ) -> EvalResult:
        """Runs an evaluation for the EvalTask.

        Args:
          model: A GenerativeModel instance or a custom model function to generate
            responses to evaluate. If not provided, the evaluation is computed with
            the `response` column in the `dataset`.
          prompt_template: The prompt template to use for the evaluation. If not
            set, the prompt template that was used to create the EvalTask will be
            used.
          experiment_run_name: The name of the experiment run to log the evaluation
            to if an experiment is set for this EvalTask. If not provided, a random
            unique experiment run name is used.
          response_column_name: The column name of model response in the dataset. If
            not set, default to `response`.

        Returns:
          The evaluation result.
        """
        global_experiment_name = metadata._experiment_tracker.experiment_name
        if experiment_run_name and not self.experiment and not global_experiment_name:
            raise ValueError(
                "Experiment is not set. Please initialize EvalTask with an"
                " experiment, or initialize a global experiment with "
                "`vertexai.init(experiment='experiment_name')`for logging this"
                " evaluation run."
            )

        experiment_run_name = experiment_run_name or f"{uuid.uuid4()}"

        if self.experiment and global_experiment_name:
            metadata._experiment_tracker.set_experiment(
                experiment=self.experiment, backing_tensorboard=False
            )
            eval_result = self._evaluate_with_experiment(
                model, prompt_template, experiment_run_name, response_column_name
            )
            metadata._experiment_tracker.set_experiment(
                experiment=global_experiment_name, backing_tensorboard=False
            )
        elif self.experiment and not global_experiment_name:
            metadata._experiment_tracker.set_experiment(
                experiment=self.experiment, backing_tensorboard=False
            )
            eval_result = self._evaluate_with_experiment(
                model, prompt_template, experiment_run_name, response_column_name
            )
            metadata._experiment_tracker.reset()
        elif not self.experiment and global_experiment_name:
            eval_result = self._evaluate_with_experiment(
                model, prompt_template, experiment_run_name, response_column_name
            )
        else:
            eval_result = _evaluation.evaluate(
                dataset=self.dataset,
                metrics=self.metrics,
                model=model,
                prompt_template=prompt_template,
                content_column_name=self.content_column_name,
                reference_column_name=self.reference_column_name,
                response_column_name=response_column_name or self.response_column_name,
            )
        return eval_result

    def _validate_experiment_run(self) -> None:
        """Checks if an experiment run already exists."""
        if metadata._experiment_tracker.experiment_run:
            raise ValueError(
                "Experiment run already exists. Please specify the name of the"
                " experiment run to assign current session with in this evaluate"
                " method."
            )

    def _log_eval_experiment_param(
        self,
        model: Optional[Union[GenerativeModel, Callable[[str], str]]] = None,
        prompt_template: Optional[str] = None,
    ) -> None:
        """Logs variable input parameters of an evaluation to an experiment run."""
        model_metadata = {}

        if prompt_template is not None:
            model_metadata.update({"prompt_template": prompt_template})

        if isinstance(model, GenerativeModel):
            model_metadata.update(
                {
                    "model_name": model._model_name,
                }
            )

            if model._generation_config and isinstance(model._generation_config, dict):
                # TODO(b/311221071): support logging GenerationConfig type.
                model_metadata.update(**model._generation_config)

            if model._safety_settings and isinstance(model._safety_settings, dict):
                # TODO(b/311221071): support logging List[SafetySetting] type.
                safety_settings = model._safety_settings
                safety_settings_as_str = {
                    category.name: threshold.name
                    for category, threshold in safety_settings.items()
                }
                model_metadata.update(safety_settings_as_str)

        if model_metadata:
            _LOGGER.info(f"Logging Rapid Eval experiment metadata: {model_metadata}")
            try:
                vertexai.preview.log_params(model_metadata)
            except (ValueError, TypeError) as e:
                _LOGGER.warning(f"Experiment metadata logging failed: {str(e)}")

    def display_runs(self):
        """Displays experiment runs associated with this EvalTask."""
        if not self.experiment:
            raise ValueError("Experiment is not set.")
        elif IPython_display:
            IPython_display.display(vertexai.preview.get_experiment_df(self.experiment))
