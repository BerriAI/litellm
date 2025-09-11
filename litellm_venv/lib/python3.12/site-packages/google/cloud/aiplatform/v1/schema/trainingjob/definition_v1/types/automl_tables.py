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

from google.cloud.aiplatform.v1.schema.trainingjob.definition_v1.types import (
    export_evaluated_data_items_config as gcastd_export_evaluated_data_items_config,
)


__protobuf__ = proto.module(
    package="google.cloud.aiplatform.v1.schema.trainingjob.definition",
    manifest={
        "AutoMlTables",
        "AutoMlTablesInputs",
        "AutoMlTablesMetadata",
    },
)


class AutoMlTables(proto.Message):
    r"""A TrainingJob that trains and uploads an AutoML Tables Model.

    Attributes:
        inputs (google.cloud.aiplatform.v1.schema.trainingjob.definition_v1.types.AutoMlTablesInputs):
            The input parameters of this TrainingJob.
        metadata (google.cloud.aiplatform.v1.schema.trainingjob.definition_v1.types.AutoMlTablesMetadata):
            The metadata information.
    """

    inputs: "AutoMlTablesInputs" = proto.Field(
        proto.MESSAGE,
        number=1,
        message="AutoMlTablesInputs",
    )
    metadata: "AutoMlTablesMetadata" = proto.Field(
        proto.MESSAGE,
        number=2,
        message="AutoMlTablesMetadata",
    )


class AutoMlTablesInputs(proto.Message):
    r"""

    This message has `oneof`_ fields (mutually exclusive fields).
    For each oneof, at most one member field can be set at the same time.
    Setting any member of the oneof automatically clears all other
    members.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        optimization_objective_recall_value (float):
            Required when optimization_objective is
            "maximize-precision-at-recall". Must be between 0 and 1,
            inclusive.

            This field is a member of `oneof`_ ``additional_optimization_objective_config``.
        optimization_objective_precision_value (float):
            Required when optimization_objective is
            "maximize-recall-at-precision". Must be between 0 and 1,
            inclusive.

            This field is a member of `oneof`_ ``additional_optimization_objective_config``.
        prediction_type (str):
            The type of prediction the Model is to
            produce.   "classification" - Predict one out of
            multiple target values is
            picked for each row.
              "regression" - Predict a value based on its
            relation to other values.                  This
            type is available only to columns that contain
            semantically numeric values, i.e. integers or
            floating                  point number, even if
            stored as e.g. strings.
        target_column (str):
            The column name of the target column that the
            model is to predict.
        transformations (MutableSequence[google.cloud.aiplatform.v1.schema.trainingjob.definition_v1.types.AutoMlTablesInputs.Transformation]):
            Each transformation will apply transform
            function to given input column. And the result
            will be used for training. When creating
            transformation for BigQuery Struct column, the
            column should be flattened using "." as the
            delimiter.
        optimization_objective (str):
            Objective function the model is optimizing
            towards. The training process creates a model
            that maximizes/minimizes the value of the
            objective function over the validation set.

            The supported optimization objectives depend on
            the prediction type. If the field is not set, a
            default objective function is used.

            classification (binary):

              "maximize-au-roc" (default) - Maximize the
            area under the receiver
            operating characteristic (ROC) curve.
            "minimize-log-loss" - Minimize log loss.
              "maximize-au-prc" - Maximize the area under
            the precision-recall curve.
            "maximize-precision-at-recall" - Maximize
            precision for a specified
            recall value.   "maximize-recall-at-precision" -
            Maximize recall for a specified
            precision value.

            classification (multi-class):

              "minimize-log-loss" (default) - Minimize log
            loss.

            regression:

              "minimize-rmse" (default) - Minimize
            root-mean-squared error (RMSE).   "minimize-mae"
            - Minimize mean-absolute error (MAE).
            "minimize-rmsle" - Minimize root-mean-squared
            log error (RMSLE).
        train_budget_milli_node_hours (int):
            Required. The train budget of creating this
            model, expressed in milli node hours i.e. 1,000
            value in this field means 1 node hour.

            The training cost of the model will not exceed
            this budget. The final cost will be attempted to
            be close to the budget, though may end up being
            (even) noticeably smaller - at the backend's
            discretion. This especially may happen when
            further model training ceases to provide any
            improvements.

            If the budget is set to a value known to be
            insufficient to train a model for the given
            dataset, the training won't be attempted and
            will error.

            The train budget must be between 1,000 and
            72,000 milli node hours, inclusive.
        disable_early_stopping (bool):
            Use the entire training budget. This disables
            the early stopping feature. By default, the
            early stopping feature is enabled, which means
            that AutoML Tables might stop training before
            the entire training budget has been used.
        weight_column_name (str):
            Column name that should be used as the weight
            column. Higher values in this column give more
            importance to the row during model training. The
            column must have numeric values between 0 and
            10000 inclusively; 0 means the row is ignored
            for training. If weight column field is not set,
            then all rows are assumed to have equal weight
            of 1.
        export_evaluated_data_items_config (google.cloud.aiplatform.v1.schema.trainingjob.definition_v1.types.ExportEvaluatedDataItemsConfig):
            Configuration for exporting test set
            predictions to a BigQuery table. If this
            configuration is absent, then the export is not
            performed.
        additional_experiments (MutableSequence[str]):
            Additional experiment flags for the Tables
            training pipeline.
    """

    class Transformation(proto.Message):
        r"""

        This message has `oneof`_ fields (mutually exclusive fields).
        For each oneof, at most one member field can be set at the same time.
        Setting any member of the oneof automatically clears all other
        members.

        .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

        Attributes:
            auto (google.cloud.aiplatform.v1.schema.trainingjob.definition_v1.types.AutoMlTablesInputs.Transformation.AutoTransformation):

                This field is a member of `oneof`_ ``transformation_detail``.
            numeric (google.cloud.aiplatform.v1.schema.trainingjob.definition_v1.types.AutoMlTablesInputs.Transformation.NumericTransformation):

                This field is a member of `oneof`_ ``transformation_detail``.
            categorical (google.cloud.aiplatform.v1.schema.trainingjob.definition_v1.types.AutoMlTablesInputs.Transformation.CategoricalTransformation):

                This field is a member of `oneof`_ ``transformation_detail``.
            timestamp (google.cloud.aiplatform.v1.schema.trainingjob.definition_v1.types.AutoMlTablesInputs.Transformation.TimestampTransformation):

                This field is a member of `oneof`_ ``transformation_detail``.
            text (google.cloud.aiplatform.v1.schema.trainingjob.definition_v1.types.AutoMlTablesInputs.Transformation.TextTransformation):

                This field is a member of `oneof`_ ``transformation_detail``.
            repeated_numeric (google.cloud.aiplatform.v1.schema.trainingjob.definition_v1.types.AutoMlTablesInputs.Transformation.NumericArrayTransformation):

                This field is a member of `oneof`_ ``transformation_detail``.
            repeated_categorical (google.cloud.aiplatform.v1.schema.trainingjob.definition_v1.types.AutoMlTablesInputs.Transformation.CategoricalArrayTransformation):

                This field is a member of `oneof`_ ``transformation_detail``.
            repeated_text (google.cloud.aiplatform.v1.schema.trainingjob.definition_v1.types.AutoMlTablesInputs.Transformation.TextArrayTransformation):

                This field is a member of `oneof`_ ``transformation_detail``.
        """

        class AutoTransformation(proto.Message):
            r"""Training pipeline will infer the proper transformation based
            on the statistic of dataset.

            Attributes:
                column_name (str):

            """

            column_name: str = proto.Field(
                proto.STRING,
                number=1,
            )

        class NumericTransformation(proto.Message):
            r"""Training pipeline will perform following transformation functions.

            -  The value converted to float32.
            -  The z_score of the value.
            -  log(value+1) when the value is greater than or equal to 0.
               Otherwise, this transformation is not applied and the value is
               considered a missing value.
            -  z_score of log(value+1) when the value is greater than or equal
               to 0. Otherwise, this transformation is not applied and the value
               is considered a missing value.
            -  A boolean value that indicates whether the value is valid.

            Attributes:
                column_name (str):

                invalid_values_allowed (bool):
                    If invalid values is allowed, the training
                    pipeline will create a boolean feature that
                    indicated whether the value is valid. Otherwise,
                    the training pipeline will discard the input row
                    from trainining data.
            """

            column_name: str = proto.Field(
                proto.STRING,
                number=1,
            )
            invalid_values_allowed: bool = proto.Field(
                proto.BOOL,
                number=2,
            )

        class CategoricalTransformation(proto.Message):
            r"""Training pipeline will perform following transformation functions.

            -  The categorical string as is--no change to case, punctuation,
               spelling, tense, and so on.
            -  Convert the category name to a dictionary lookup index and
               generate an embedding for each index.
            -  Categories that appear less than 5 times in the training dataset
               are treated as the "unknown" category. The "unknown" category
               gets its own special lookup index and resulting embedding.

            Attributes:
                column_name (str):

            """

            column_name: str = proto.Field(
                proto.STRING,
                number=1,
            )

        class TimestampTransformation(proto.Message):
            r"""Training pipeline will perform following transformation functions.

            -  Apply the transformation functions for Numerical columns.
            -  Determine the year, month, day,and weekday. Treat each value from
               the
            -  timestamp as a Categorical column.
            -  Invalid numerical values (for example, values that fall outside
               of a typical timestamp range, or are extreme values) receive no
               special treatment and are not removed.

            Attributes:
                column_name (str):

                time_format (str):
                    The format in which that time field is expressed. The
                    time_format must either be one of:

                    -  ``unix-seconds``
                    -  ``unix-milliseconds``
                    -  ``unix-microseconds``
                    -  ``unix-nanoseconds`` (for respectively number of seconds,
                       milliseconds, microseconds and nanoseconds since start of
                       the Unix epoch); or be written in ``strftime`` syntax. If
                       time_format is not set, then the default format is RFC
                       3339 ``date-time`` format, where ``time-offset`` =
                       ``"Z"`` (e.g. 1985-04-12T23:20:50.52Z)
                invalid_values_allowed (bool):
                    If invalid values is allowed, the training
                    pipeline will create a boolean feature that
                    indicated whether the value is valid. Otherwise,
                    the training pipeline will discard the input row
                    from trainining data.
            """

            column_name: str = proto.Field(
                proto.STRING,
                number=1,
            )
            time_format: str = proto.Field(
                proto.STRING,
                number=2,
            )
            invalid_values_allowed: bool = proto.Field(
                proto.BOOL,
                number=3,
            )

        class TextTransformation(proto.Message):
            r"""Training pipeline will perform following transformation functions.

            -  The text as is--no change to case, punctuation, spelling, tense,
               and so on.
            -  Tokenize text to words. Convert each words to a dictionary lookup
               index and generate an embedding for each index. Combine the
               embedding of all elements into a single embedding using the mean.
            -  Tokenization is based on unicode script boundaries.
            -  Missing values get their own lookup index and resulting
               embedding.
            -  Stop-words receive no special treatment and are not removed.

            Attributes:
                column_name (str):

            """

            column_name: str = proto.Field(
                proto.STRING,
                number=1,
            )

        class NumericArrayTransformation(proto.Message):
            r"""Treats the column as numerical array and performs following
            transformation functions.

            -  All transformations for Numerical types applied to the average of
               the all elements.
            -  The average of empty arrays is treated as zero.

            Attributes:
                column_name (str):

                invalid_values_allowed (bool):
                    If invalid values is allowed, the training
                    pipeline will create a boolean feature that
                    indicated whether the value is valid. Otherwise,
                    the training pipeline will discard the input row
                    from trainining data.
            """

            column_name: str = proto.Field(
                proto.STRING,
                number=1,
            )
            invalid_values_allowed: bool = proto.Field(
                proto.BOOL,
                number=2,
            )

        class CategoricalArrayTransformation(proto.Message):
            r"""Treats the column as categorical array and performs following
            transformation functions.

            -  For each element in the array, convert the category name to a
               dictionary lookup index and generate an embedding for each index.
               Combine the embedding of all elements into a single embedding
               using the mean.
            -  Empty arrays treated as an embedding of zeroes.

            Attributes:
                column_name (str):

            """

            column_name: str = proto.Field(
                proto.STRING,
                number=1,
            )

        class TextArrayTransformation(proto.Message):
            r"""Treats the column as text array and performs following
            transformation functions.

            -  Concatenate all text values in the array into a single text value
               using a space (" ") as a delimiter, and then treat the result as
               a single text value. Apply the transformations for Text columns.
            -  Empty arrays treated as an empty text.

            Attributes:
                column_name (str):

            """

            column_name: str = proto.Field(
                proto.STRING,
                number=1,
            )

        auto: "AutoMlTablesInputs.Transformation.AutoTransformation" = proto.Field(
            proto.MESSAGE,
            number=1,
            oneof="transformation_detail",
            message="AutoMlTablesInputs.Transformation.AutoTransformation",
        )
        numeric: "AutoMlTablesInputs.Transformation.NumericTransformation" = (
            proto.Field(
                proto.MESSAGE,
                number=2,
                oneof="transformation_detail",
                message="AutoMlTablesInputs.Transformation.NumericTransformation",
            )
        )
        categorical: "AutoMlTablesInputs.Transformation.CategoricalTransformation" = (
            proto.Field(
                proto.MESSAGE,
                number=3,
                oneof="transformation_detail",
                message="AutoMlTablesInputs.Transformation.CategoricalTransformation",
            )
        )
        timestamp: "AutoMlTablesInputs.Transformation.TimestampTransformation" = (
            proto.Field(
                proto.MESSAGE,
                number=4,
                oneof="transformation_detail",
                message="AutoMlTablesInputs.Transformation.TimestampTransformation",
            )
        )
        text: "AutoMlTablesInputs.Transformation.TextTransformation" = proto.Field(
            proto.MESSAGE,
            number=5,
            oneof="transformation_detail",
            message="AutoMlTablesInputs.Transformation.TextTransformation",
        )
        repeated_numeric: "AutoMlTablesInputs.Transformation.NumericArrayTransformation" = proto.Field(
            proto.MESSAGE,
            number=6,
            oneof="transformation_detail",
            message="AutoMlTablesInputs.Transformation.NumericArrayTransformation",
        )
        repeated_categorical: "AutoMlTablesInputs.Transformation.CategoricalArrayTransformation" = proto.Field(
            proto.MESSAGE,
            number=7,
            oneof="transformation_detail",
            message="AutoMlTablesInputs.Transformation.CategoricalArrayTransformation",
        )
        repeated_text: "AutoMlTablesInputs.Transformation.TextArrayTransformation" = (
            proto.Field(
                proto.MESSAGE,
                number=8,
                oneof="transformation_detail",
                message="AutoMlTablesInputs.Transformation.TextArrayTransformation",
            )
        )

    optimization_objective_recall_value: float = proto.Field(
        proto.FLOAT,
        number=5,
        oneof="additional_optimization_objective_config",
    )
    optimization_objective_precision_value: float = proto.Field(
        proto.FLOAT,
        number=6,
        oneof="additional_optimization_objective_config",
    )
    prediction_type: str = proto.Field(
        proto.STRING,
        number=1,
    )
    target_column: str = proto.Field(
        proto.STRING,
        number=2,
    )
    transformations: MutableSequence[Transformation] = proto.RepeatedField(
        proto.MESSAGE,
        number=3,
        message=Transformation,
    )
    optimization_objective: str = proto.Field(
        proto.STRING,
        number=4,
    )
    train_budget_milli_node_hours: int = proto.Field(
        proto.INT64,
        number=7,
    )
    disable_early_stopping: bool = proto.Field(
        proto.BOOL,
        number=8,
    )
    weight_column_name: str = proto.Field(
        proto.STRING,
        number=9,
    )
    export_evaluated_data_items_config: gcastd_export_evaluated_data_items_config.ExportEvaluatedDataItemsConfig = proto.Field(
        proto.MESSAGE,
        number=10,
        message=gcastd_export_evaluated_data_items_config.ExportEvaluatedDataItemsConfig,
    )
    additional_experiments: MutableSequence[str] = proto.RepeatedField(
        proto.STRING,
        number=11,
    )


class AutoMlTablesMetadata(proto.Message):
    r"""Model metadata specific to AutoML Tables.

    Attributes:
        train_cost_milli_node_hours (int):
            Output only. The actual training cost of the
            model, expressed in milli node hours, i.e. 1,000
            value in this field means 1 node hour.
            Guaranteed to not exceed the train budget.
    """

    train_cost_milli_node_hours: int = proto.Field(
        proto.INT64,
        number=1,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
