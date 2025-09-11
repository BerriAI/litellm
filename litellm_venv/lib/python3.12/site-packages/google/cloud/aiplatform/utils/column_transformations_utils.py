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

from typing import Dict, List, Optional, Tuple

from google.cloud.aiplatform import datasets


def get_default_column_transformations(
    dataset: datasets._ColumnNamesDataset,
    target_column: str,
) -> Tuple[List[Dict[str, Dict[str, str]]], List[str]]:
    """Get default column transformations from the column names, while omitting the target column.

    Args:
        dataset (_ColumnNamesDataset):
            Required. The dataset
        target_column (str):
            Required. The name of the column values of which the Model is to predict.

    Returns:
        Tuple[List[Dict[str, Dict[str, str]]], List[str]]:
            The default column transformations and the default column names.
    """

    column_names = [
        column_name
        for column_name in dataset.column_names
        if column_name != target_column
    ]
    column_transformations = [
        {"auto": {"column_name": column_name}} for column_name in column_names
    ]

    return (column_transformations, column_names)


def validate_and_get_column_transformations(
    column_specs: Optional[Dict[str, str]] = None,
    column_transformations: Optional[List[Dict[str, Dict[str, str]]]] = None,
) -> Optional[List[Dict[str, Dict[str, str]]]]:
    """Validates column specs and transformations, then returns processed transformations.

    Args:
        column_specs (Dict[str, str]):
            Optional. Alternative to column_transformations where the keys of the dict
            are column names and their respective values are one of
            AutoMLTabularTrainingJob.column_data_types.
            When creating transformation for BigQuery Struct column, the column
            should be flattened using "." as the delimiter. Only columns with no child
            should have a transformation.
            If an input column has no transformations on it, such a column is
            ignored by the training, except for the targetColumn, which should have
            no transformations defined on.
            Only one of column_transformations or column_specs should be passed.
        column_transformations (List[Dict[str, Dict[str, str]]]):
            Optional. Transformations to apply to the input columns (i.e. columns other
            than the targetColumn). Each transformation may produce multiple
            result values from the column's value, and all are used for training.
            When creating transformation for BigQuery Struct column, the column
            should be flattened using "." as the delimiter. Only columns with no child
            should have a transformation.
            If an input column has no transformations on it, such a column is
            ignored by the training, except for the targetColumn, which should have
            no transformations defined on.
            Only one of column_transformations or column_specs should be passed.
            Consider using column_specs as column_transformations will be deprecated eventually.

    Returns:
        List[Dict[str, Dict[str, str]]]:
            The column transformations.

    Raises:
        ValueError: If both column_transformations and column_specs were provided.
    """
    # user populated transformations
    if column_transformations is not None and column_specs is not None:
        raise ValueError(
            "Both column_transformations and column_specs were passed. Only "
            "one is allowed."
        )
    elif column_specs is not None:
        return [
            {transformation: {"column_name": column_name}}
            for column_name, transformation in column_specs.items()
        ]
    else:
        return column_transformations
