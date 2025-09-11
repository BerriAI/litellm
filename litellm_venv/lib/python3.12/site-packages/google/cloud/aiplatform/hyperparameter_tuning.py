# -*- coding: utf-8 -*-

# Copyright 2022 Google LLC
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

import abc
from typing import Dict, List, Optional, Sequence, Tuple, Union

import proto

from google.cloud.aiplatform.compat.types import (
    study_v1beta1 as gca_study_compat_v1beta1,
    study as gca_study_compat,
)

SEARCH_ALGORITHM_TO_PROTO_VALUE = {
    "random": gca_study_compat.StudySpec.Algorithm.RANDOM_SEARCH,
    "grid": gca_study_compat.StudySpec.Algorithm.GRID_SEARCH,
    None: gca_study_compat.StudySpec.Algorithm.ALGORITHM_UNSPECIFIED,
}

MEASUREMENT_SELECTION_TO_PROTO_VALUE = {
    "best": (gca_study_compat.StudySpec.MeasurementSelectionType.BEST_MEASUREMENT),
    "last": (gca_study_compat.StudySpec.MeasurementSelectionType.LAST_MEASUREMENT),
    None: (
        gca_study_compat.StudySpec.MeasurementSelectionType.MEASUREMENT_SELECTION_TYPE_UNSPECIFIED
    ),
}

_SCALE_TYPE_MAP = {
    "linear": gca_study_compat.StudySpec.ParameterSpec.ScaleType.UNIT_LINEAR_SCALE,
    "log": gca_study_compat.StudySpec.ParameterSpec.ScaleType.UNIT_LOG_SCALE,
    "reverse_log": gca_study_compat.StudySpec.ParameterSpec.ScaleType.UNIT_REVERSE_LOG_SCALE,
    "unspecified": gca_study_compat.StudySpec.ParameterSpec.ScaleType.SCALE_TYPE_UNSPECIFIED,
}

_INT_VALUE_SPEC = "integer_value_spec"
_DISCRETE_VALUE_SPEC = "discrete_value_spec"
_CATEGORICAL_VALUE_SPEC = "categorical_value_spec"


class _ParameterSpec(metaclass=abc.ABCMeta):
    """Base class represents a single parameter to optimize."""

    def __init__(
        self,
        conditional_parameter_spec: Optional[Dict[str, "_ParameterSpec"]] = None,
        parent_values: Optional[List[Union[float, int, str]]] = None,
    ):
        self.conditional_parameter_spec = conditional_parameter_spec
        self.parent_values = parent_values

    @property
    @classmethod
    @abc.abstractmethod
    def _proto_parameter_value_class(self) -> proto.Message:
        """The proto representation of this parameter."""
        pass

    @property
    @classmethod
    @abc.abstractmethod
    def _parameter_value_map(self) -> Tuple[Tuple[str, str]]:
        """A Tuple map of parameter key to underlying proto key."""
        pass

    @property
    @classmethod
    @abc.abstractmethod
    def _parameter_spec_value_key(self) -> Tuple[Tuple[str, str]]:
        """The ParameterSpec key this parameter should be assigned."""
        pass

    @property
    def _proto_parameter_value_spec(self) -> proto.Message:
        """Converts this parameter to it's parameter value representation."""
        proto_parameter_value_spec = self._proto_parameter_value_class()
        for self_attr_key, proto_attr_key in self._parameter_value_map:
            setattr(
                proto_parameter_value_spec, proto_attr_key, getattr(self, self_attr_key)
            )
        return proto_parameter_value_spec

    @property
    def _proto_parameter_value_spec_v1beta1(self) -> proto.Message:
        """Converts this parameter to it's parameter value representation."""
        if isinstance(
            self._proto_parameter_value_class(),
            gca_study_compat.StudySpec.ParameterSpec.DoubleValueSpec,
        ):
            proto_parameter_value_spec = (
                gca_study_compat_v1beta1.StudySpec.ParameterSpec.DoubleValueSpec()
            )
        elif isinstance(
            self._proto_parameter_value_class(),
            gca_study_compat.StudySpec.ParameterSpec.IntegerValueSpec,
        ):
            proto_parameter_value_spec = (
                gca_study_compat_v1beta1.StudySpec.ParameterSpec.IntegerValueSpec()
            )
        elif isinstance(
            self._proto_parameter_value_class(),
            gca_study_compat.StudySpec.ParameterSpec.CategoricalValueSpec,
        ):
            proto_parameter_value_spec = (
                gca_study_compat_v1beta1.StudySpec.ParameterSpec.CategoricalValueSpec()
            )
        elif isinstance(
            self._proto_parameter_value_class(),
            gca_study_compat.StudySpec.ParameterSpec.DiscreteValueSpec,
        ):
            proto_parameter_value_spec = (
                gca_study_compat_v1beta1.StudySpec.ParameterSpec.DiscreteValueSpec()
            )
        else:
            proto_parameter_value_spec = self._proto_parameter_value_class()

        for self_attr_key, proto_attr_key in self._parameter_value_map:
            setattr(
                proto_parameter_value_spec, proto_attr_key, getattr(self, self_attr_key)
            )
        return proto_parameter_value_spec

    def _to_parameter_spec(
        self, parameter_id: str
    ) -> gca_study_compat.StudySpec.ParameterSpec:
        """Converts this parameter to ParameterSpec."""
        conditions = []
        if self.conditional_parameter_spec is not None:
            for conditional_param_id, spec in self.conditional_parameter_spec.items():
                condition = (
                    gca_study_compat.StudySpec.ParameterSpec.ConditionalParameterSpec()
                )
                if self._parameter_spec_value_key == _INT_VALUE_SPEC:
                    condition.parent_int_values = gca_study_compat.StudySpec.ParameterSpec.ConditionalParameterSpec.IntValueCondition(
                        values=spec.parent_values
                    )
                elif self._parameter_spec_value_key == _CATEGORICAL_VALUE_SPEC:
                    condition.parent_categorical_values = gca_study_compat.StudySpec.ParameterSpec.ConditionalParameterSpec.CategoricalValueCondition(
                        values=spec.parent_values
                    )
                elif self._parameter_spec_value_key == _DISCRETE_VALUE_SPEC:
                    condition.parent_discrete_values = gca_study_compat.StudySpec.ParameterSpec.ConditionalParameterSpec.DiscreteValueCondition(
                        values=spec.parent_values
                    )
                condition.parameter_spec = spec._to_parameter_spec(conditional_param_id)
                conditions.append(condition)
        parameter_spec = gca_study_compat.StudySpec.ParameterSpec(
            parameter_id=parameter_id,
            scale_type=_SCALE_TYPE_MAP.get(getattr(self, "scale", "unspecified")),
            conditional_parameter_specs=conditions,
        )

        setattr(
            parameter_spec,
            self._parameter_spec_value_key,
            self._proto_parameter_value_spec,
        )

        return parameter_spec

    def _to_parameter_spec_v1beta1(
        self, parameter_id: str
    ) -> gca_study_compat_v1beta1.StudySpec.ParameterSpec:
        """Converts this parameter to ParameterSpec."""
        conditions = []
        if self.conditional_parameter_spec is not None:
            for conditional_param_id, spec in self.conditional_parameter_spec.items():
                condition = (
                    gca_study_compat_v1beta1.StudySpec.ParameterSpec.ConditionalParameterSpec()
                )
                if self._parameter_spec_value_key == _INT_VALUE_SPEC:
                    condition.parent_int_values = gca_study_compat_v1beta1.StudySpec.ParameterSpec.ConditionalParameterSpec.IntValueCondition(
                        values=spec.parent_values
                    )
                elif self._parameter_spec_value_key == _CATEGORICAL_VALUE_SPEC:
                    condition.parent_categorical_values = gca_study_compat_v1beta1.StudySpec.ParameterSpec.ConditionalParameterSpec.CategoricalValueCondition(
                        values=spec.parent_values
                    )
                elif self._parameter_spec_value_key == _DISCRETE_VALUE_SPEC:
                    condition.parent_discrete_values = gca_study_compat_v1beta1.StudySpec.ParameterSpec.ConditionalParameterSpec.DiscreteValueCondition(
                        values=spec.parent_values
                    )
                condition.parameter_spec = spec._to_parameter_spec_v1beta1(
                    conditional_param_id
                )
                conditions.append(condition)
        parameter_spec = gca_study_compat_v1beta1.StudySpec.ParameterSpec(
            parameter_id=parameter_id,
            scale_type=_SCALE_TYPE_MAP.get(getattr(self, "scale", "unspecified")),
            conditional_parameter_specs=conditions,
        )

        setattr(
            parameter_spec,
            self._parameter_spec_value_key,
            self._proto_parameter_value_spec_v1beta1,
        )

        return parameter_spec


class DoubleParameterSpec(_ParameterSpec):
    _proto_parameter_value_class = (
        gca_study_compat.StudySpec.ParameterSpec.DoubleValueSpec
    )
    _parameter_value_map = (("min", "min_value"), ("max", "max_value"))
    _parameter_spec_value_key = "double_value_spec"

    def __init__(
        self,
        min: float,
        max: float,
        scale: str,
        conditional_parameter_spec: Optional[Dict[str, "_ParameterSpec"]] = None,
        parent_values: Optional[Sequence[Union[int, float, str]]] = None,
    ):
        """
        Value specification for a parameter in ``DOUBLE`` type.

        Args:
            min (float):
                Required. Inclusive minimum value of the
                parameter.
            max (float):
                Required. Inclusive maximum value of the
                parameter.
            scale (str):
                Required. The type of scaling that should be applied to this parameter.

                Accepts: 'linear', 'log', 'reverse_log'
            conditional_parameter_spec (Dict[str, _ParameterSpec]):
                Optional. The conditional parameters associated with the object. The dictionary key
                is the ID of the conditional parameter and the dictionary value is one of
                `IntegerParameterSpec`, `CategoricalParameterSpec`, or `DiscreteParameterSpec`
            parent_values (Sequence[Union[int, float, str]]):
                Optional. This argument is only needed when the object is a conditional parameter
                and specifies the parent parameter's values for which the condition applies.
        """

        super().__init__(conditional_parameter_spec, parent_values)

        self.min = min
        self.max = max
        self.scale = scale


class IntegerParameterSpec(_ParameterSpec):
    _proto_parameter_value_class = (
        gca_study_compat.StudySpec.ParameterSpec.IntegerValueSpec
    )
    _parameter_value_map = (("min", "min_value"), ("max", "max_value"))
    _parameter_spec_value_key = "integer_value_spec"

    def __init__(
        self,
        min: int,
        max: int,
        scale: str,
        conditional_parameter_spec: Optional[Dict[str, "_ParameterSpec"]] = None,
        parent_values: Optional[Sequence[Union[int, float, str]]] = None,
    ):
        """
        Value specification for a parameter in ``INTEGER`` type.

        Args:
            min (float):
                Required. Inclusive minimum value of the
                parameter.
            max (float):
                Required. Inclusive maximum value of the
                parameter.
            scale (str):
                Required. The type of scaling that should be applied to this parameter.

                Accepts: 'linear', 'log', 'reverse_log'
            conditional_parameter_spec (Dict[str, _ParameterSpec]):
                Optional. The conditional parameters associated with the object. The dictionary key
                is the ID of the conditional parameter and the dictionary value is one of
                `IntegerParameterSpec`, `CategoricalParameterSpec`, or `DiscreteParameterSpec`
            parent_values (Sequence[int]):
                Optional. This argument is only needed when the object is a conditional parameter
                and specifies the parent parameter's values for which the condition applies.
        """
        super().__init__(
            conditional_parameter_spec=conditional_parameter_spec,
            parent_values=parent_values,
        )

        self.min = min
        self.max = max
        self.scale = scale


class CategoricalParameterSpec(_ParameterSpec):
    _proto_parameter_value_class = (
        gca_study_compat.StudySpec.ParameterSpec.CategoricalValueSpec
    )
    _parameter_value_map = (("values", "values"),)
    _parameter_spec_value_key = "categorical_value_spec"

    def __init__(
        self,
        values: Sequence[str],
        conditional_parameter_spec: Optional[Dict[str, "_ParameterSpec"]] = None,
        parent_values: Optional[Sequence[Union[int, float, str]]] = None,
    ):
        """Value specification for a parameter in ``CATEGORICAL`` type.

        Args:
            values (Sequence[str]):
                Required. The list of possible categories.
            conditional_parameter_spec (Dict[str, _ParameterSpec]):
                Optional. The conditional parameters associated with the object. The dictionary key
                is the ID of the conditional parameter and the dictionary value is one of
                `IntegerParameterSpec`, `CategoricalParameterSpec`, or `DiscreteParameterSpec`
            parent_values (Sequence[str]):
                Optional. This argument is only needed when the object is a conditional parameter
                and specifies the parent parameter's values for which the condition applies.
        """
        super().__init__(
            conditional_parameter_spec=conditional_parameter_spec,
            parent_values=parent_values,
        )

        self.values = values


class DiscreteParameterSpec(_ParameterSpec):
    _proto_parameter_value_class = (
        gca_study_compat.StudySpec.ParameterSpec.DiscreteValueSpec
    )
    _parameter_value_map = (("values", "values"),)
    _parameter_spec_value_key = "discrete_value_spec"

    def __init__(
        self,
        values: Sequence[float],
        scale: str,
        conditional_parameter_spec: Optional[Dict[str, "_ParameterSpec"]] = None,
        parent_values: Optional[Sequence[Union[int, float, str]]] = None,
    ):
        """Value specification for a parameter in ``DISCRETE`` type.

        values (Sequence[float]):
            Required. A list of possible values.
            The list should be in increasing order and at
            least 1e-10 apart. For instance, this parameter
            might have possible settings of 1.5, 2.5, and
            4.0. This list should not contain more than
            1,000 values.
        scale (str):
            Required. The type of scaling that should be applied to this parameter.

            Accepts: 'linear', 'log', 'reverse_log'
        conditional_parameter_spec (Dict[str, _ParameterSpec]):
            Optional. The conditional parameters associated with the object. The dictionary key
            is the ID of the conditional parameter and the dictionary value is one of
            `IntegerParameterSpec`, `CategoricalParameterSpec`, or `DiscreteParameterSpec`
        parent_values (Sequence[float]):
            Optional. This argument is only needed when the object is a conditional parameter
            and specifies the parent parameter's values for which the condition applies.
        """
        super().__init__(
            conditional_parameter_spec=conditional_parameter_spec,
            parent_values=parent_values,
        )

        self.values = values
        self.scale = scale
