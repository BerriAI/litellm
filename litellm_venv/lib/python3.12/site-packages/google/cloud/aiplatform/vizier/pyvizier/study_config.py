"""Convenience classes for configuring Vizier Study Configs and Search Spaces.

This module contains several classes, used to access/build Vizier StudyConfig
protos:
  * `StudyConfig` class is the main class, which:
  1) Allows to easily build Vizier StudyConfig protos via a convenient
     Python API.
  2) Can be initialized from an existing StudyConfig proto, to enable easy
     Pythonic accessors to information contained in StudyConfig protos,
     and easy field editing capabilities.

  * `SearchSpace` and `SearchSpaceSelector` classes deals with Vizier search
    spaces. Both flat spaces and conditional parameters are supported.
"""
import collections
import copy
import enum
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import attr
from google.cloud.aiplatform.vizier.pyvizier.automated_stopping import (
    AutomatedStoppingConfig,
)
from google.cloud.aiplatform.vizier.pyvizier import proto_converters
from google.cloud.aiplatform.vizier.pyvizier import SearchSpace
from google.cloud.aiplatform.vizier.pyvizier import ProblemStatement
from google.cloud.aiplatform.vizier.pyvizier import ObjectiveMetricGoal
from google.cloud.aiplatform.vizier.pyvizier import SearchSpaceSelector
from google.cloud.aiplatform.vizier.pyvizier import MetricsConfig
from google.cloud.aiplatform.vizier.pyvizier import MetricInformation
from google.cloud.aiplatform.vizier.pyvizier import Trial
from google.cloud.aiplatform.vizier.pyvizier import ParameterValueTypes
from google.cloud.aiplatform.vizier.pyvizier import ParameterConfig
from google.cloud.aiplatform.compat.types import study as study_pb2

################### PyTypes ###################
# A sequence of possible internal parameter values.
# Possible types for trial parameter values after cast to external types.
ParameterValueSequence = Union[
    ParameterValueTypes,
    Sequence[int],
    Sequence[float],
    Sequence[str],
    Sequence[bool],
]

################### Enums ###################


class Algorithm(enum.Enum):
    """Valid Values for StudyConfig.Algorithm."""

    ALGORITHM_UNSPECIFIED = study_pb2.StudySpec.Algorithm.ALGORITHM_UNSPECIFIED
    # GAUSSIAN_PROCESS_BANDIT = study_pb2.StudySpec.Algorithm.GAUSSIAN_PROCESS_BANDIT
    GRID_SEARCH = study_pb2.StudySpec.Algorithm.GRID_SEARCH
    RANDOM_SEARCH = study_pb2.StudySpec.Algorithm.RANDOM_SEARCH
    # NSGA2 = study_pb2.StudySpec.Algorithm.NSGA2
    # EMUKIT_GP_EI = study_pb2.StudySpec.Algorithm.EMUKIT_GP_EI


class ObservationNoise(enum.Enum):
    """Valid Values for StudyConfig.ObservationNoise."""

    OBSERVATION_NOISE_UNSPECIFIED = (
        study_pb2.StudySpec.ObservationNoise.OBSERVATION_NOISE_UNSPECIFIED
    )
    LOW = study_pb2.StudySpec.ObservationNoise.LOW
    HIGH = study_pb2.StudySpec.ObservationNoise.HIGH


################### Classes For Various Config Protos ###################
@attr.define(frozen=False, init=True, slots=True, kw_only=True)
class MetricInformationConverter:
    """A wrapper for vizier_pb2.MetricInformation."""

    @classmethod
    def from_proto(cls, proto: study_pb2.StudySpec.MetricSpec) -> MetricInformation:
        """Converts a MetricInformation proto to a MetricInformation object."""
        if proto.goal not in list(ObjectiveMetricGoal):
            raise ValueError("Unknown MetricInformation.goal: {}".format(proto.goal))

        return MetricInformation(
            name=proto.metric_id,
            goal=proto.goal,
            safety_threshold=None,
            safety_std_threshold=None,
            min_value=None,
            max_value=None,
        )

    @classmethod
    def to_proto(cls, obj: MetricInformation) -> study_pb2.StudySpec.MetricSpec:
        """Returns this object as a proto."""
        return study_pb2.StudySpec.MetricSpec(metric_id=obj.name, goal=obj.goal.value)


class MetricsConfig(MetricsConfig):
    """Metrics config."""

    @classmethod
    def from_proto(
        cls, protos: Iterable[study_pb2.StudySpec.MetricSpec]
    ) -> "MetricsConfig":
        return cls(MetricInformationConverter.from_proto(m) for m in protos)

    def to_proto(self) -> List[study_pb2.StudySpec.MetricSpec]:
        return [MetricInformationConverter.to_proto(metric) for metric in self]


SearchSpaceSelector = SearchSpaceSelector


@attr.define(frozen=True, init=True, slots=True, kw_only=True)
class SearchSpace(SearchSpace):
    """A Selector for all, or part of a SearchSpace."""

    @classmethod
    def from_proto(cls, proto: study_pb2.StudySpec) -> "SearchSpace":
        """Extracts a SearchSpace object from a StudyConfig proto."""

        # For google-vizier <= 0.0.15
        if hasattr(cls, "_factory"):
            parameter_configs = []
            for pc in proto.parameters:
                parameter_configs.append(
                    proto_converters.ParameterConfigConverter.from_proto(pc)
                )
            return cls._factory(parameter_configs=parameter_configs)

        result = cls()
        for pc in proto.parameters:
            result.add(proto_converters.ParameterConfigConverter.from_proto(pc))

        return result

    @property
    def parameter_protos(self) -> List[study_pb2.StudySpec.ParameterSpec]:
        """Returns the search space as a List of ParameterConfig protos."""

        # For google-vizier <= 0.0.15
        if isinstance(self._parameter_configs, list):
            return [
                proto_converters.ParameterConfigConverter.to_proto(pc)
                for pc in self._parameter_configs
            ]

        return [
            proto_converters.ParameterConfigConverter.to_proto(pc)
            for _, pc in self._parameter_configs.items()
        ]


################### Main Class ###################
#
# A StudyConfig object can be initialized:
# (1) From a StudyConfig proto using StudyConfig.from_proto():
#     study_config_proto = study_pb2.StudySpec(...)
#     study_config = pyvizier.StudyConfig.from_proto(study_config_proto)
#     # Attributes can be modified.
#     new_proto = study_config.to_proto()
#
# (2) By directly calling __init__ and setting attributes:
#     study_config = pyvizier.StudyConfig(
#       metric_information=[pyvizier.MetricInformation(
#         name='accuracy', goal=pyvizier.ObjectiveMetricGoal.MAXIMIZE)],
#       search_space=SearchSpace.from_proto(proto),
#     )
#     # OR:
#     study_config = pyvizier.StudyConfig()
#     study_config.metric_information.append(
#        pyvizier.MetricInformation(
#          name='accuracy', goal=pyvizier.ObjectiveMetricGoal.MAXIMIZE))
#
#     # Since building a search space is more involved, get a reference to the
#     # search space, and add parameters to it.
#     root = study_config.search_space.select_root()
#     root.add_float_param('learning_rate', 0.001, 1.0,
#       scale_type=pyvizier.ScaleType.LOG)
#
@attr.define(frozen=False, init=True, slots=True, kw_only=True)
class StudyConfig(ProblemStatement):
    """A builder and wrapper for study_pb2.StudySpec proto."""

    search_space: SearchSpace = attr.field(
        init=True,
        factory=SearchSpace,
        validator=attr.validators.instance_of(SearchSpace),
        on_setattr=attr.setters.validate,
    )

    algorithm: Algorithm = attr.field(
        init=True,
        validator=attr.validators.instance_of(Algorithm),
        on_setattr=[attr.setters.convert, attr.setters.validate],
        default=Algorithm.ALGORITHM_UNSPECIFIED,
        kw_only=True,
    )

    metric_information: MetricsConfig = attr.field(
        init=True,
        factory=MetricsConfig,
        converter=MetricsConfig,
        validator=attr.validators.instance_of(MetricsConfig),
        kw_only=True,
    )

    observation_noise: ObservationNoise = attr.field(
        init=True,
        validator=attr.validators.instance_of(ObservationNoise),
        on_setattr=attr.setters.validate,
        default=ObservationNoise.OBSERVATION_NOISE_UNSPECIFIED,
        kw_only=True,
    )

    automated_stopping_config: Optional[AutomatedStoppingConfig] = attr.field(
        init=True,
        default=None,
        validator=attr.validators.optional(
            attr.validators.instance_of(AutomatedStoppingConfig)
        ),
        on_setattr=attr.setters.validate,
        kw_only=True,
    )

    # An internal representation as a StudyConfig proto.
    # If this object was created from a StudyConfig proto, a copy of the original
    # proto is kept, to make sure that unknown proto fields are preserved in
    # round trip serialization.
    # TODO: Fix the broken proto validation.
    _study_config: study_pb2.StudySpec = attr.field(
        init=True, factory=study_pb2.StudySpec, kw_only=True
    )

    # Public attributes, methods and properties.
    @classmethod
    def from_proto(cls, proto: study_pb2.StudySpec) -> "StudyConfig":
        """Converts a StudyConfig proto to a StudyConfig object.

        Args:
          proto: StudyConfig proto.

        Returns:
          A StudyConfig object.
        """
        metric_information = MetricsConfig(
            sorted(
                [MetricInformationConverter.from_proto(m) for m in proto.metrics],
                key=lambda x: x.name,
            )
        )

        oneof_name = proto._pb.WhichOneof("automated_stopping_spec")
        if not oneof_name:
            automated_stopping_config = None
        else:
            automated_stopping_config = AutomatedStoppingConfig.from_proto(
                getattr(proto, oneof_name)
            )

        return cls(
            search_space=SearchSpace.from_proto(proto),
            algorithm=Algorithm(proto.algorithm),
            metric_information=metric_information,
            observation_noise=ObservationNoise(proto.observation_noise),
            automated_stopping_config=automated_stopping_config,
            study_config=copy.deepcopy(proto),
        )

    def to_proto(self) -> study_pb2.StudySpec:
        """Serializes this object to a StudyConfig proto."""
        proto = copy.deepcopy(self._study_config)
        proto.algorithm = self.algorithm.value
        proto.observation_noise = self.observation_noise.value

        del proto.metrics[:]
        proto.metrics.extend(self.metric_information.to_proto())

        del proto.parameters[:]
        proto.parameters.extend(self.search_space.parameter_protos)

        if self.automated_stopping_config is not None:
            auto_stop_proto = self.automated_stopping_config.to_proto()
            if isinstance(
                auto_stop_proto, study_pb2.StudySpec.DecayCurveAutomatedStoppingSpec
            ):
                proto.decay_curve_stopping_spec = copy.deepcopy(auto_stop_proto)
            elif isinstance(
                auto_stop_proto, study_pb2.StudySpec.DecayCurveAutomatedStoppingSpec
            ):
                for method_name in dir(proto.decay_curve_stopping_spec):
                    if callable(
                        getattr(proto.median_automated_stopping_spec, method_name)
                    ):
                        print(method_name)
                proto.median_automated_stopping_spec = copy.deepcopy(auto_stop_proto)

        return proto

    @property
    def is_single_objective(self) -> bool:
        """Returns True if only one objective metric is configured."""
        return len(self.metric_information) == 1

    @property
    def single_objective_metric_name(self) -> Optional[str]:
        """Returns the name of the single-objective metric, if set.

        Returns:
          String: name of the single-objective metric.
          None: if this is not a single-objective study.
        """
        if len(self.metric_information) == 1:
            return list(self.metric_information)[0].name
        return None

    def _trial_to_external_values(
        self, pytrial: Trial
    ) -> Dict[str, Union[float, int, str, bool]]:
        """Returns the trial paremeter values cast to external types."""
        parameter_values: Dict[str, Union[float, int, str]] = {}
        external_values: Dict[str, Union[float, int, str, bool]] = {}
        # parameter_configs is a list of Tuple[parent_name, ParameterConfig].
        parameter_configs: List[Tuple[Optional[str], ParameterConfig]] = [
            (None, p) for p in self.search_space.parameters
        ]
        remaining_parameters = copy.deepcopy(pytrial.parameters)
        # Traverse the conditional tree using a BFS.
        while parameter_configs and remaining_parameters:
            parent_name, pc = parameter_configs.pop(0)
            parameter_configs.extend(
                (pc.name, child) for child in pc.child_parameter_configs
            )
            if pc.name not in remaining_parameters:
                continue
            if parent_name is not None:
                # This is a child parameter. If the parent was not seen,
                # skip this parameter config.
                if parent_name not in parameter_values:
                    continue
                parent_value = parameter_values[parent_name]
                if parent_value not in pc.matching_parent_values:
                    continue
            parameter_values[pc.name] = remaining_parameters[pc.name].value
            if pc.external_type is None:
                external_value = remaining_parameters[pc.name].value
            else:
                external_value = remaining_parameters[pc.name].cast(
                    pc.external_type
                )  # pytype: disable=wrong-arg-types
            external_values[pc.name] = external_value
            remaining_parameters.pop(pc.name)
        return external_values

    def trial_parameters(
        self, proto: study_pb2.Trial
    ) -> Dict[str, ParameterValueSequence]:
        """Returns the trial values, cast to external types, if they exist.

        Args:
          proto:

        Returns:
          Parameter values dict: cast to each parameter's external_type, if exists.
          NOTE that the values in the dict may be a Sequence as opposed to a single
          element.

        Raises:
          ValueError: If the trial parameters do not exist in this search space.
          ValueError: If the trial contains duplicate parameters.
        """
        pytrial = proto_converters.TrialConverter.from_proto(proto)
        return self._pytrial_parameters(pytrial)

    def _pytrial_parameters(self, pytrial: Trial) -> Dict[str, ParameterValueSequence]:
        """Returns the trial values, cast to external types, if they exist.

        Args:
          pytrial:

        Returns:
          Parameter values dict: cast to each parameter's external_type, if exists.
          NOTE that the values in the dict may be a Sequence as opposed to a single
          element.

        Raises:
          ValueError: If the trial parameters do not exist in this search space.
          ValueError: If the trial contains duplicate parameters.
        """
        trial_external_values: Dict[
            str, Union[float, int, str, bool]
        ] = self._trial_to_external_values(pytrial)
        if len(trial_external_values) != len(pytrial.parameters):
            raise ValueError(
                "Invalid trial for this search space: failed to convert "
                "all trial parameters: {}".format(pytrial)
            )

        # Combine multi-dimensional parameter values to a list of values.
        trial_final_values: Dict[str, ParameterValueSequence] = {}
        # multi_dim_params: Dict[str, List[Tuple[int, ParameterValueSequence]]]
        multi_dim_params = collections.defaultdict(list)
        for name in trial_external_values:
            base_index = SearchSpaceSelector.parse_multi_dimensional_parameter_name(
                name
            )
            if base_index is None:
                trial_final_values[name] = trial_external_values[name]
            else:
                base_name, index = base_index
                multi_dim_params[base_name].append((index, trial_external_values[name]))
        for name in multi_dim_params:
            multi_dim_params[name].sort(key=lambda x: x[0])
            trial_final_values[name] = [x[1] for x in multi_dim_params[name]]

        return trial_final_values

    def trial_metrics(
        self, proto: study_pb2.Trial, *, include_all_metrics=False
    ) -> Dict[str, float]:
        """Returns the trial's final measurement metric values.

        If the trial is not completed, or infeasible, no metrics are returned.
        By default, only metrics configured in the StudyConfig are returned
        (e.g. only objective and safety metrics).

        Args:
          proto:
          include_all_metrics: If True, all metrics in the final measurements are
            returned. If False, only metrics configured in the StudyConfig are
            returned.

        Returns:
          Dict[metric name, metric value]
        """
        pytrial = proto_converters.TrialConverter.from_proto(proto)
        return self._pytrial_metrics(pytrial, include_all_metrics=include_all_metrics)

    def _pytrial_metrics(
        self, pytrial: Trial, *, include_all_metrics=False
    ) -> Dict[str, float]:
        """Returns the trial's final measurement metric values.

        If the trial is not completed, or infeasible, no metrics are returned.
        By default, only metrics configured in the StudyConfig are returned
        (e.g. only objective and safety metrics).

        Args:
          pytrial:
          include_all_metrics: If True, all metrics in the final measurements are
            returned. If False, only metrics configured in the StudyConfig are
            returned.

        Returns:
          Dict[metric name, metric value]
        """
        configured_metrics = [m.name for m in self.metric_information]

        metrics: Dict[str, float] = {}
        if pytrial.is_completed and not pytrial.infeasible:
            for name in pytrial.final_measurement.metrics:
                if include_all_metrics or (
                    not include_all_metrics and name in configured_metrics
                ):
                    # Special case: Measurement always adds an empty metric by default.
                    # If there is a named single objective in study_config, drop the empty
                    # metric.
                    if not name and self.single_objective_metric_name != name:
                        continue
                    metrics[name] = pytrial.final_measurement.metrics[name].value
        return metrics
