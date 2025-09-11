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

from google.cloud.aiplatform.v1beta1.schema.trainingjob.definition_v1beta1.types import (
    export_evaluated_data_items_config as gcastd_export_evaluated_data_items_config,
)


__protobuf__ = proto.module(
    package="google.cloud.aiplatform.v1beta1.schema.trainingjob.definition",
    manifest={
        "AutoMlForecasting",
        "AutoMlForecastingInputs",
        "AutoMlForecastingMetadata",
    },
)


class AutoMlForecasting(proto.Message):
    r"""A TrainingJob that trains and uploads an AutoML Forecasting
    Model.

    Attributes:
        inputs (google.cloud.aiplatform.v1beta1.schema.trainingjob.definition_v1beta1.types.AutoMlForecastingInputs):
            The input parameters of this TrainingJob.
        metadata (google.cloud.aiplatform.v1beta1.schema.trainingjob.definition_v1beta1.types.AutoMlForecastingMetadata):
            The metadata information.
    """

    inputs: "AutoMlForecastingInputs" = proto.Field(
        proto.MESSAGE,
        number=1,
        message="AutoMlForecastingInputs",
    )
    metadata: "AutoMlForecastingMetadata" = proto.Field(
        proto.MESSAGE,
        number=2,
        message="AutoMlForecastingMetadata",
    )


class AutoMlForecastingInputs(proto.Message):
    r"""

    Attributes:
        target_column (str):
            The name of the column that the model is to
            predict.
        time_series_identifier_column (str):
            The name of the column that identifies the
            time series.
        time_column (str):
            The name of the column that identifies time
            order in the time series.
        transformations (MutableSequence[google.cloud.aiplatform.v1beta1.schema.trainingjob.definition_v1beta1.types.AutoMlForecastingInputs.Transformation]):
            Each transformation will apply transform
            function to given input column. And the result
            will be used for training. When creating
            transformation for BigQuery Struct column, the
            column should be flattened using "." as the
            delimiter.
        optimization_objective (str):
            Objective function the model is optimizing towards. The
            training process creates a model that optimizes the value of
            the objective function over the validation set.

            The supported optimization objectives:

            -  "minimize-rmse" (default) - Minimize root-mean-squared
               error (RMSE).

            -  "minimize-mae" - Minimize mean-absolute error (MAE).

            -  "minimize-rmsle" - Minimize root-mean-squared log error
               (RMSLE).

            -  "minimize-rmspe" - Minimize root-mean-squared percentage
               error (RMSPE).

            -  "minimize-wape-mae" - Minimize the combination of
               weighted absolute percentage error (WAPE) and
               mean-absolute-error (MAE).

            -  "minimize-quantile-loss" - Minimize the quantile loss at
               the quantiles defined in ``quantiles``.
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
        weight_column (str):
            Column name that should be used as the weight
            column. Higher values in this column give more
            importance to the row during model training. The
            column must have numeric values between 0 and
            10000 inclusively; 0 means the row is ignored
            for training. If weight column field is not set,
            then all rows are assumed to have equal weight
            of 1.
        time_series_attribute_columns (MutableSequence[str]):
            Column names that should be used as attribute
            columns. The value of these columns does not
            vary as a function of time. For example, store
            ID or item color.
        unavailable_at_forecast_columns (MutableSequence[str]):
            Names of columns that are unavailable when a forecast is
            requested. This column contains information for the given
            entity (identified by the time_series_identifier_column)
            that is unknown before the forecast For example, actual
            weather on a given day.
        available_at_forecast_columns (MutableSequence[str]):
            Names of columns that are available and provided when a
            forecast is requested. These columns contain information for
            the given entity (identified by the
            time_series_identifier_column column) that is known at
            forecast. For example, predicted weather for a specific day.
        data_granularity (google.cloud.aiplatform.v1beta1.schema.trainingjob.definition_v1beta1.types.AutoMlForecastingInputs.Granularity):
            Expected difference in time granularity
            between rows in the data.
        forecast_horizon (int):
            The amount of time into the future for which forecasted
            values for the target are returned. Expressed in number of
            units defined by the ``data_granularity`` field.
        context_window (int):
            The amount of time into the past training and prediction
            data is used for model training and prediction respectively.
            Expressed in number of units defined by the
            ``data_granularity`` field.
        export_evaluated_data_items_config (google.cloud.aiplatform.v1beta1.schema.trainingjob.definition_v1beta1.types.ExportEvaluatedDataItemsConfig):
            Configuration for exporting test set
            predictions to a BigQuery table. If this
            configuration is absent, then the export is not
            performed.
        quantiles (MutableSequence[float]):
            Quantiles to use for minimize-quantile-loss
            ``optimization_objective``. Up to 5 quantiles are allowed of
            values between 0 and 1, exclusive. Required if the value of
            optimization_objective is minimize-quantile-loss. Represents
            the percent quantiles to use for that objective. Quantiles
            must be unique.
        validation_options (str):
            Validation options for the data validation component. The
            available options are:

            -  "fail-pipeline" - default, will validate against the
               validation and fail the pipeline if it fails.

            -  "ignore-validation" - ignore the results of the
               validation and continue
        additional_experiments (MutableSequence[str]):
            Additional experiment flags for the time
            series forcasting training.
    """

    class Transformation(proto.Message):
        r"""

        This message has `oneof`_ fields (mutually exclusive fields).
        For each oneof, at most one member field can be set at the same time.
        Setting any member of the oneof automatically clears all other
        members.

        .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

        Attributes:
            auto (google.cloud.aiplatform.v1beta1.schema.trainingjob.definition_v1beta1.types.AutoMlForecastingInputs.Transformation.AutoTransformation):

                This field is a member of `oneof`_ ``transformation_detail``.
            numeric (google.cloud.aiplatform.v1beta1.schema.trainingjob.definition_v1beta1.types.AutoMlForecastingInputs.Transformation.NumericTransformation):

                This field is a member of `oneof`_ ``transformation_detail``.
            categorical (google.cloud.aiplatform.v1beta1.schema.trainingjob.definition_v1beta1.types.AutoMlForecastingInputs.Transformation.CategoricalTransformation):

                This field is a member of `oneof`_ ``transformation_detail``.
            timestamp (google.cloud.aiplatform.v1beta1.schema.trainingjob.definition_v1beta1.types.AutoMlForecastingInputs.Transformation.TimestampTransformation):

                This field is a member of `oneof`_ ``transformation_detail``.
            text (google.cloud.aiplatform.v1beta1.schema.trainingjob.definition_v1beta1.types.AutoMlForecastingInputs.Transformation.TextTransformation):

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

            """

            column_name: str = proto.Field(
                proto.STRING,
                number=1,
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
               the timestamp as a Categorical column.

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

                    -  ``unix-nanoseconds``

                    (for respectively number of seconds, milliseconds,
                    microseconds and nanoseconds since start of the Unix epoch);

                    or be written in ``strftime`` syntax.

                    If time_format is not set, then the default format is RFC
                    3339 ``date-time`` format, where ``time-offset`` = ``"Z"``
                    (e.g. 1985-04-12T23:20:50.52Z)
            """

            column_name: str = proto.Field(
                proto.STRING,
                number=1,
            )
            time_format: str = proto.Field(
                proto.STRING,
                number=2,
            )

        class TextTransformation(proto.Message):
            r"""Training pipeline will perform following transformation functions.

            -  The text as is--no change to case, punctuation, spelling, tense,
               and so on.

            -  Convert the category name to a dictionary lookup index and
               generate an embedding for each index.

            Attributes:
                column_name (str):

            """

            column_name: str = proto.Field(
                proto.STRING,
                number=1,
            )

        auto: "AutoMlForecastingInputs.Transformation.AutoTransformation" = proto.Field(
            proto.MESSAGE,
            number=1,
            oneof="transformation_detail",
            message="AutoMlForecastingInputs.Transformation.AutoTransformation",
        )
        numeric: "AutoMlForecastingInputs.Transformation.NumericTransformation" = (
            proto.Field(
                proto.MESSAGE,
                number=2,
                oneof="transformation_detail",
                message="AutoMlForecastingInputs.Transformation.NumericTransformation",
            )
        )
        categorical: "AutoMlForecastingInputs.Transformation.CategoricalTransformation" = proto.Field(
            proto.MESSAGE,
            number=3,
            oneof="transformation_detail",
            message="AutoMlForecastingInputs.Transformation.CategoricalTransformation",
        )
        timestamp: "AutoMlForecastingInputs.Transformation.TimestampTransformation" = proto.Field(
            proto.MESSAGE,
            number=4,
            oneof="transformation_detail",
            message="AutoMlForecastingInputs.Transformation.TimestampTransformation",
        )
        text: "AutoMlForecastingInputs.Transformation.TextTransformation" = proto.Field(
            proto.MESSAGE,
            number=5,
            oneof="transformation_detail",
            message="AutoMlForecastingInputs.Transformation.TextTransformation",
        )

    class Granularity(proto.Message):
        r"""A duration of time expressed in time granularity units.

        Attributes:
            unit (str):
                The time granularity unit of this time period. The supported
                units are:

                -  "minute"

                -  "hour"

                -  "day"

                -  "week"

                -  "month"

                -  "year".
            quantity (int):
                The number of granularity_units between data points in the
                training data. If ``granularity_unit`` is ``minute``, can be
                1, 5, 10, 15, or 30. For all other values of
                ``granularity_unit``, must be 1.
        """

        unit: str = proto.Field(
            proto.STRING,
            number=1,
        )
        quantity: int = proto.Field(
            proto.INT64,
            number=2,
        )

    target_column: str = proto.Field(
        proto.STRING,
        number=1,
    )
    time_series_identifier_column: str = proto.Field(
        proto.STRING,
        number=2,
    )
    time_column: str = proto.Field(
        proto.STRING,
        number=3,
    )
    transformations: MutableSequence[Transformation] = proto.RepeatedField(
        proto.MESSAGE,
        number=4,
        message=Transformation,
    )
    optimization_objective: str = proto.Field(
        proto.STRING,
        number=5,
    )
    train_budget_milli_node_hours: int = proto.Field(
        proto.INT64,
        number=6,
    )
    weight_column: str = proto.Field(
        proto.STRING,
        number=7,
    )
    time_series_attribute_columns: MutableSequence[str] = proto.RepeatedField(
        proto.STRING,
        number=19,
    )
    unavailable_at_forecast_columns: MutableSequence[str] = proto.RepeatedField(
        proto.STRING,
        number=20,
    )
    available_at_forecast_columns: MutableSequence[str] = proto.RepeatedField(
        proto.STRING,
        number=21,
    )
    data_granularity: Granularity = proto.Field(
        proto.MESSAGE,
        number=22,
        message=Granularity,
    )
    forecast_horizon: int = proto.Field(
        proto.INT64,
        number=23,
    )
    context_window: int = proto.Field(
        proto.INT64,
        number=24,
    )
    export_evaluated_data_items_config: gcastd_export_evaluated_data_items_config.ExportEvaluatedDataItemsConfig = proto.Field(
        proto.MESSAGE,
        number=15,
        message=gcastd_export_evaluated_data_items_config.ExportEvaluatedDataItemsConfig,
    )
    quantiles: MutableSequence[float] = proto.RepeatedField(
        proto.DOUBLE,
        number=16,
    )
    validation_options: str = proto.Field(
        proto.STRING,
        number=17,
    )
    additional_experiments: MutableSequence[str] = proto.RepeatedField(
        proto.STRING,
        number=25,
    )


class AutoMlForecastingMetadata(proto.Message):
    r"""Model metadata specific to AutoML Forecasting.

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
