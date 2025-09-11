# -*- coding: utf-8 -*-

# Copyright 2020 Google LLC
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

    inputs = proto.Field(
        proto.MESSAGE,
        number=1,
        message="AutoMlForecastingInputs",
    )

    metadata = proto.Field(
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
        transformations (Sequence[google.cloud.aiplatform.v1beta1.schema.trainingjob.definition_v1beta1.types.AutoMlForecastingInputs.Transformation]):
            Each transformation will apply transform
            function to given input column. And the result
            will be used for training. When creating
            transformation for BigQuery Struct column, the
            column should be flattened using "." as the
            delimiter.
        optimization_objective (str):
            Objective function the model is optimizing
            towards. The training process creates a model
            that optimizes the value of the objective
            function over the validation set.

            The supported optimization objectives:
              "minimize-rmse" (default) - Minimize root-
            mean-squared error (RMSE).   "minimize-mae" -
            Minimize mean-absolute error (MAE).   "minimize-
            rmsle" - Minimize root-mean-squared log error
            (RMSLE).   "minimize-rmspe" - Minimize root-
            mean-squared percentage error (RMSPE).
            "minimize-wape-mae" - Minimize the combination
            of weighted absolute     percentage error (WAPE)
            and mean-absolute-error (MAE).
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
        static_columns (Sequence[str]):
            Column names that should be used as static
            columns. The value of these columns are static
            per time series.
        time_variant_past_only_columns (Sequence[str]):
            Column names that should be used as time variant past only
            columns. This column contains information for the given
            entity (identified by the time_series_identifier_column)
            that is known for the past but not the future (e.g.
            population of a city in a given year, or weather on a given
            day).
        time_variant_past_and_future_columns (Sequence[str]):
            Column names that should be used as time
            variant past and future columns. This column
            contains information for the given entity
            (identified by the key column) that is known for
            the past and the future
        period (google.cloud.aiplatform.v1beta1.schema.trainingjob.definition_v1beta1.types.AutoMlForecastingInputs.Period):
            Expected difference in time granularity
            between rows in the data. If it is not set, the
            period is inferred from data.
        forecast_window_start (int):
            The number of periods offset into the future as the start of
            the forecast window (the window of future values to predict,
            relative to the present.), where each period is one unit of
            granularity as defined by the ``period`` field above.
            Default to 0. Inclusive.
        forecast_window_end (int):
            The number of periods offset into the future as the end of
            the forecast window (the window of future values to predict,
            relative to the present.), where each period is one unit of
            granularity as defined by the ``period`` field above.
            Inclusive.
        past_horizon (int):
            The number of periods offset into the past to restrict past
            sequence, where each period is one unit of granularity as
            defined by the ``period``. Default value 0 means that it
            lets algorithm to define the value. Inclusive.
        export_evaluated_data_items_config (google.cloud.aiplatform.v1beta1.schema.trainingjob.definition_v1beta1.types.ExportEvaluatedDataItemsConfig):
            Configuration for exporting test set
            predictions to a BigQuery table. If this
            configuration is absent, then the export is not
            performed.
    """

    class Transformation(proto.Message):
        r"""

        Attributes:
            auto (google.cloud.aiplatform.v1beta1.schema.trainingjob.definition_v1beta1.types.AutoMlForecastingInputs.Transformation.AutoTransformation):

            numeric (google.cloud.aiplatform.v1beta1.schema.trainingjob.definition_v1beta1.types.AutoMlForecastingInputs.Transformation.NumericTransformation):

            categorical (google.cloud.aiplatform.v1beta1.schema.trainingjob.definition_v1beta1.types.AutoMlForecastingInputs.Transformation.CategoricalTransformation):

            timestamp (google.cloud.aiplatform.v1beta1.schema.trainingjob.definition_v1beta1.types.AutoMlForecastingInputs.Transformation.TimestampTransformation):

            text (google.cloud.aiplatform.v1beta1.schema.trainingjob.definition_v1beta1.types.AutoMlForecastingInputs.Transformation.TextTransformation):

            repeated_numeric (google.cloud.aiplatform.v1beta1.schema.trainingjob.definition_v1beta1.types.AutoMlForecastingInputs.Transformation.NumericArrayTransformation):

            repeated_categorical (google.cloud.aiplatform.v1beta1.schema.trainingjob.definition_v1beta1.types.AutoMlForecastingInputs.Transformation.CategoricalArrayTransformation):

            repeated_text (google.cloud.aiplatform.v1beta1.schema.trainingjob.definition_v1beta1.types.AutoMlForecastingInputs.Transformation.TextArrayTransformation):

        """

        class AutoTransformation(proto.Message):
            r"""Training pipeline will infer the proper transformation based
            on the statistic of dataset.

            Attributes:
                column_name (str):

            """

            column_name = proto.Field(proto.STRING, number=1)

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

            column_name = proto.Field(proto.STRING, number=1)

            invalid_values_allowed = proto.Field(proto.BOOL, number=2)

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

            column_name = proto.Field(proto.STRING, number=1)

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

            column_name = proto.Field(proto.STRING, number=1)

            time_format = proto.Field(proto.STRING, number=2)

            invalid_values_allowed = proto.Field(proto.BOOL, number=3)

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

            column_name = proto.Field(proto.STRING, number=1)

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

            column_name = proto.Field(proto.STRING, number=1)

            invalid_values_allowed = proto.Field(proto.BOOL, number=2)

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

            column_name = proto.Field(proto.STRING, number=1)

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

            column_name = proto.Field(proto.STRING, number=1)

        auto = proto.Field(
            proto.MESSAGE,
            number=1,
            oneof="transformation_detail",
            message="AutoMlForecastingInputs.Transformation.AutoTransformation",
        )

        numeric = proto.Field(
            proto.MESSAGE,
            number=2,
            oneof="transformation_detail",
            message="AutoMlForecastingInputs.Transformation.NumericTransformation",
        )

        categorical = proto.Field(
            proto.MESSAGE,
            number=3,
            oneof="transformation_detail",
            message="AutoMlForecastingInputs.Transformation.CategoricalTransformation",
        )

        timestamp = proto.Field(
            proto.MESSAGE,
            number=4,
            oneof="transformation_detail",
            message="AutoMlForecastingInputs.Transformation.TimestampTransformation",
        )

        text = proto.Field(
            proto.MESSAGE,
            number=5,
            oneof="transformation_detail",
            message="AutoMlForecastingInputs.Transformation.TextTransformation",
        )

        repeated_numeric = proto.Field(
            proto.MESSAGE,
            number=6,
            oneof="transformation_detail",
            message="AutoMlForecastingInputs.Transformation.NumericArrayTransformation",
        )

        repeated_categorical = proto.Field(
            proto.MESSAGE,
            number=7,
            oneof="transformation_detail",
            message="AutoMlForecastingInputs.Transformation.CategoricalArrayTransformation",
        )

        repeated_text = proto.Field(
            proto.MESSAGE,
            number=8,
            oneof="transformation_detail",
            message="AutoMlForecastingInputs.Transformation.TextArrayTransformation",
        )

    class Period(proto.Message):
        r"""A duration of time expressed in time granularity units.

        Attributes:
            unit (str):
                The time granularity unit of this time
                period. The supported unit are:
                 "hour"
                 "day"
                 "week"
                 "month"
                 "year".
            quantity (int):
                The number of units per period, e.g. 3 weeks
                or 2 months.
        """

        unit = proto.Field(proto.STRING, number=1)

        quantity = proto.Field(proto.INT64, number=2)

    target_column = proto.Field(proto.STRING, number=1)

    time_series_identifier_column = proto.Field(proto.STRING, number=2)

    time_column = proto.Field(proto.STRING, number=3)

    transformations = proto.RepeatedField(
        proto.MESSAGE,
        number=4,
        message=Transformation,
    )

    optimization_objective = proto.Field(proto.STRING, number=5)

    train_budget_milli_node_hours = proto.Field(proto.INT64, number=6)

    weight_column = proto.Field(proto.STRING, number=7)

    static_columns = proto.RepeatedField(proto.STRING, number=8)

    time_variant_past_only_columns = proto.RepeatedField(proto.STRING, number=9)

    time_variant_past_and_future_columns = proto.RepeatedField(proto.STRING, number=10)

    period = proto.Field(
        proto.MESSAGE,
        number=11,
        message=Period,
    )

    forecast_window_start = proto.Field(proto.INT64, number=12)

    forecast_window_end = proto.Field(proto.INT64, number=13)

    past_horizon = proto.Field(proto.INT64, number=14)

    export_evaluated_data_items_config = proto.Field(
        proto.MESSAGE,
        number=15,
        message=gcastd_export_evaluated_data_items_config.ExportEvaluatedDataItemsConfig,
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

    train_cost_milli_node_hours = proto.Field(proto.INT64, number=1)


__all__ = tuple(sorted(__protobuf__.manifest))
