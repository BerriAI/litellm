"""Converters for OSS Vizier's protos from/to PyVizier's classes."""
import logging
from datetime import timezone
from typing import List, Optional, Sequence, Tuple, Union

from google.protobuf import duration_pb2
from google.protobuf import struct_pb2
from google.protobuf import timestamp_pb2
from google.cloud.aiplatform.compat.types import study as study_pb2
from google.cloud.aiplatform.vizier.pyvizier import ExternalType
from google.cloud.aiplatform.vizier.pyvizier import ScaleType
from google.cloud.aiplatform.vizier.pyvizier import ParameterType
from google.cloud.aiplatform.vizier.pyvizier import ParameterValue
from google.cloud.aiplatform.vizier.pyvizier import MonotypeParameterSequence
from google.cloud.aiplatform.vizier.pyvizier import ParameterConfig
from google.cloud.aiplatform.vizier.pyvizier import Measurement
from google.cloud.aiplatform.vizier.pyvizier import Metric
from google.cloud.aiplatform.vizier.pyvizier import TrialStatus
from google.cloud.aiplatform.vizier.pyvizier import Trial

_ScaleTypePb2 = study_pb2.StudySpec.ParameterSpec.ScaleType


class _ScaleTypeMap:
    """Proto converter for scale type."""

    _pyvizier_to_proto = {
        ScaleType.LINEAR: _ScaleTypePb2.UNIT_LINEAR_SCALE,
        ScaleType.LOG: _ScaleTypePb2.UNIT_LOG_SCALE,
        ScaleType.REVERSE_LOG: _ScaleTypePb2.UNIT_REVERSE_LOG_SCALE,
    }
    _proto_to_pyvizier = {v: k for k, v in _pyvizier_to_proto.items()}

    @classmethod
    def to_proto(cls, pyvizier: ScaleType) -> _ScaleTypePb2:
        return cls._pyvizier_to_proto[pyvizier]

    @classmethod
    def from_proto(cls, proto: _ScaleTypePb2) -> ScaleType:
        return cls._proto_to_pyvizier[proto]


class ParameterConfigConverter:
    """Converter for ParameterConfig."""

    @classmethod
    def _set_bounds(
        cls,
        proto: study_pb2.StudySpec.ParameterSpec,
        lower: float,
        upper: float,
        parameter_type: ParameterType,
    ):
        """Sets the proto's min_value and max_value fields."""
        if parameter_type == ParameterType.INTEGER:
            proto.integer_value_spec.min_value = lower
            proto.integer_value_spec.max_value = upper
        elif parameter_type == ParameterType.DOUBLE:
            proto.double_value_spec.min_value = lower
            proto.double_value_spec.max_value = upper

    @classmethod
    def _set_feasible_points(
        cls, proto: study_pb2.StudySpec.ParameterSpec, feasible_points: Sequence[float]
    ):
        """Sets the proto's feasible_points field."""
        feasible_points = sorted(feasible_points)
        proto.discrete_value_spec.values.clear()
        proto.discrete_value_spec.values.extend(feasible_points)

    @classmethod
    def _set_categories(
        cls, proto: study_pb2.StudySpec.ParameterSpec, categories: Sequence[str]
    ):
        """Sets the protos' categories field."""
        proto.categorical_value_spec.values.clear()
        proto.categorical_value_spec.values.extend(categories)

    @classmethod
    def _set_default_value(
        cls,
        proto: study_pb2.StudySpec.ParameterSpec,
        default_value: Union[float, int, str],
    ):
        """Sets the protos' default_value field."""
        which_pv_spec = proto._pb.WhichOneof("parameter_value_spec")
        getattr(proto, which_pv_spec).default_value = default_value

    @classmethod
    def _matching_parent_values(
        cls, proto: study_pb2.StudySpec.ParameterSpec.ConditionalParameterSpec
    ) -> MonotypeParameterSequence:
        """Returns the matching parent values, if set."""
        oneof_name = proto.WhichOneof("parent_value_condition")
        if not oneof_name:
            return []
        if oneof_name in (
            "parent_discrete_values",
            "parent_int_values",
            "parent_categorical_values",
        ):
            return list(getattr(getattr(proto, oneof_name), "values"))
        raise ValueError("Unknown matching_parent_vals: {}".format(oneof_name))

    @classmethod
    def from_proto(
        cls,
        proto: study_pb2.StudySpec.ParameterSpec,
        *,
        strict_validation: bool = False
    ) -> ParameterConfig:
        """Creates a ParameterConfig.

        Args:
          proto:
          strict_validation: If True, raise ValueError to enforce that
            from_proto(proto).to_proto == proto.

        Returns:
          ParameterConfig object

        Raises:
          ValueError: See the "strict_validtion" arg documentation.
        """
        feasible_values = []
        external_type = ExternalType.INTERNAL
        oneof_name = proto._pb.WhichOneof("parameter_value_spec")
        if oneof_name == "integer_value_spec":
            bounds = (
                int(proto.integer_value_spec.min_value),
                int(proto.integer_value_spec.max_value),
            )
            external_type = ExternalType.INTEGER
        elif oneof_name == "double_value_spec":
            bounds = (
                proto.double_value_spec.min_value,
                proto.double_value_spec.max_value,
            )
        elif oneof_name == "discrete_value_spec":
            bounds = None
            feasible_values = proto.discrete_value_spec.values
        elif oneof_name == "categorical_value_spec":
            bounds = None
            feasible_values = proto.categorical_value_spec.values
            # Boolean values are encoded as categoricals, check for the special
            # hard-coded values.
            boolean_values = ["False", "True"]
            if sorted(list(feasible_values)) == boolean_values:
                external_type = ExternalType.BOOLEAN

        default_value = None
        if getattr(proto, oneof_name).default_value:
            default_value = getattr(proto, oneof_name).default_value
            if external_type == ExternalType.INTEGER:
                default_value = int(default_value)

        if proto.conditional_parameter_specs:
            children = []
            for conditional_ps in proto.conditional_parameter_specs:
                parent_values = cls._matching_parent_values(conditional_ps)
                children.append(
                    (parent_values, cls.from_proto(conditional_ps.parameter_spec))
                )
        else:
            children = None

        scale_type = None
        if proto.scale_type:
            scale_type = _ScaleTypeMap.from_proto(proto.scale_type)

        try:
            config = ParameterConfig.factory(
                name=proto.parameter_id,
                feasible_values=feasible_values,
                bounds=bounds,
                children=children,
                scale_type=scale_type,
                default_value=default_value,
                external_type=external_type,
            )
        except ValueError as e:
            raise ValueError(
                "The provided proto was misconfigured. {}".format(proto)
            ) from e

        if strict_validation and cls.to_proto(config) != proto:
            raise ValueError(
                "The provided proto was misconfigured. Expected: {} Given: {}".format(
                    cls.to_proto(config), proto
                )
            )
        return config

    @classmethod
    def _set_child_parameter_configs(
        cls,
        parent_proto: study_pb2.StudySpec.ParameterSpec,
        pc: ParameterConfig,
    ):
        """Sets the parent_proto's conditional_parameter_specs field.

        Args:
          parent_proto: Modified in place.
          pc: Parent ParameterConfig to copy children from.

        Raises:
          ValueError: If the child configs are invalid
        """
        children: List[Tuple[MonotypeParameterSequence, ParameterConfig]] = []
        for child in pc.child_parameter_configs:
            children.append((child.matching_parent_values, child))
        if not children:
            return
        parent_proto.conditional_parameter_specs.clear()
        for child_pair in children:
            if len(child_pair) != 2:
                raise ValueError(
                    """Each element in children must be a tuple of
            (Sequence of valid parent values,  ParameterConfig)"""
                )

        logging.debug(
            "_set_child_parameter_configs: parent_proto=%s, children=%s",
            parent_proto,
            children,
        )
        for unsorted_parent_values, child in children:
            parent_values = sorted(unsorted_parent_values)
            child_proto = cls.to_proto(child.clone_without_children)
            conditional_parameter_spec = (
                study_pb2.StudySpec.ParameterSpec.ConditionalParameterSpec(
                    parameter_spec=child_proto
                )
            )

            if "discrete_value_spec" in parent_proto:
                conditional_parameter_spec.parent_discrete_values.values[
                    :
                ] = parent_values
            elif "categorical_value_spec" in parent_proto:
                conditional_parameter_spec.parent_categorical_values.values[
                    :
                ] = parent_values
            elif "integer_value_spec" in parent_proto:
                conditional_parameter_spec.parent_int_values.values[:] = parent_values
            else:
                raise ValueError("DOUBLE type cannot have child parameters")
            if child.child_parameter_configs:
                cls._set_child_parameter_configs(child_proto, child)
            parent_proto.conditional_parameter_specs.extend(
                [conditional_parameter_spec]
            )

    @classmethod
    def to_proto(cls, pc: ParameterConfig) -> study_pb2.StudySpec.ParameterSpec:
        """Returns a ParameterConfig Proto."""
        proto = study_pb2.StudySpec.ParameterSpec(parameter_id=pc.name)
        if pc.type == ParameterType.DISCRETE:
            cls._set_feasible_points(proto, [float(v) for v in pc.feasible_values])
        elif pc.type == ParameterType.CATEGORICAL:
            cls._set_categories(proto, pc.feasible_values)
        elif pc.type in (ParameterType.INTEGER, ParameterType.DOUBLE):
            cls._set_bounds(proto, pc.bounds[0], pc.bounds[1], pc.type)
        else:
            raise ValueError("Invalid ParameterConfig: {}".format(pc))
        if pc.scale_type is not None and pc.scale_type != ScaleType.UNIFORM_DISCRETE:
            proto.scale_type = _ScaleTypeMap.to_proto(pc.scale_type)
        if pc.default_value is not None:
            cls._set_default_value(proto, pc.default_value)

        cls._set_child_parameter_configs(proto, pc)
        return proto


class ParameterValueConverter:
    """Converter for ParameterValue."""

    @classmethod
    def from_proto(cls, proto: study_pb2.Trial.Parameter) -> Optional[ParameterValue]:
        """Returns whichever value that is populated, or None."""
        potential_value = proto.value
        if (
            isinstance(potential_value, float)
            or isinstance(potential_value, str)
            or isinstance(potential_value, bool)
        ):
            return ParameterValue(potential_value)
        else:
            return None

    @classmethod
    def to_proto(
        cls, parameter_value: ParameterValue, name: str
    ) -> study_pb2.Trial.Parameter:
        """Returns Parameter Proto."""
        if isinstance(parameter_value.value, int):
            value = struct_pb2.Value(number_value=parameter_value.value)
        elif isinstance(parameter_value.value, bool):
            value = struct_pb2.Value(bool_value=parameter_value.value)
        elif isinstance(parameter_value.value, float):
            value = struct_pb2.Value(number_value=parameter_value.value)
        elif isinstance(parameter_value.value, str):
            value = struct_pb2.Value(string_value=parameter_value.value)

        proto = study_pb2.Trial.Parameter(parameter_id=name, value=value)
        return proto


class MeasurementConverter:
    """Converter for MeasurementConverter."""

    @classmethod
    def from_proto(cls, proto: study_pb2.Measurement) -> Measurement:
        """Creates a valid instance from proto.

        Args:
          proto: Measurement proto.

        Returns:
          A valid instance of Measurement object. Metrics with invalid values
          are automatically filtered out.
        """

        metrics = dict()

        for metric in proto.metrics:
            if (
                metric.metric_id in metrics
                and metrics[metric.metric_id].value != metric.value
            ):
                logging.log_first_n(
                    logging.ERROR,
                    'Duplicate metric of name "%s".'
                    "The newly found value %s will be used and "
                    "the previously found value %s will be discarded."
                    "This always happens if the proto has an empty-named metric.",
                    5,
                    metric.metric_id,
                    metric.value,
                    metrics[metric.metric_id].value,
                )
            try:
                metrics[metric.metric_id] = Metric(value=metric.value)
            except ValueError:
                pass
        return Measurement(
            metrics=metrics,
            elapsed_secs=proto.elapsed_duration.seconds,
            steps=proto.step_count,
        )

    @classmethod
    def to_proto(cls, measurement: Measurement) -> study_pb2.Measurement:
        """Converts to Measurement proto."""
        int_seconds = int(measurement.elapsed_secs)
        proto = study_pb2.Measurement(
            elapsed_duration=duration_pb2.Duration(
                seconds=int_seconds,
                nanos=int(1e9 * (measurement.elapsed_secs - int_seconds)),
            )
        )
        for name, metric in measurement.metrics.items():
            proto.metrics.append(
                study_pb2.Measurement.Metric(metric_id=name, value=metric.value)
            )

        proto.step_count = measurement.steps
        return proto


def _to_pyvizier_trial_status(proto_state: study_pb2.Trial.State) -> TrialStatus:
    """from_proto conversion for Trial statuses."""
    if proto_state == study_pb2.Trial.State.REQUESTED:
        return TrialStatus.REQUESTED
    elif proto_state == study_pb2.Trial.State.ACTIVE:
        return TrialStatus.ACTIVE
    if proto_state == study_pb2.Trial.State.STOPPING:
        return TrialStatus.STOPPING
    if proto_state == study_pb2.Trial.State.SUCCEEDED:
        return TrialStatus.COMPLETED
    elif proto_state == study_pb2.Trial.State.INFEASIBLE:
        return TrialStatus.COMPLETED
    else:
        return TrialStatus.UNKNOWN


def _from_pyvizier_trial_status(
    status: TrialStatus, infeasible: bool
) -> study_pb2.Trial.State:
    """to_proto conversion for Trial states."""
    if status == TrialStatus.REQUESTED:
        return study_pb2.Trial.State.REQUESTED
    elif status == TrialStatus.ACTIVE:
        return study_pb2.Trial.State.ACTIVE
    elif status == TrialStatus.STOPPING:
        return study_pb2.Trial.State.STOPPING
    elif status == TrialStatus.COMPLETED:
        if infeasible:
            return study_pb2.Trial.State.INFEASIBLE
        else:
            return study_pb2.Trial.State.SUCCEEDED
    else:
        return study_pb2.Trial.State.STATE_UNSPECIFIED


class TrialConverter:
    """Converter for TrialConverter."""

    @classmethod
    def from_proto(cls, proto: study_pb2.Trial) -> Trial:
        """Converts from Trial proto to object.

        Args:
          proto: Trial proto.

        Returns:
          A Trial object.
        """
        parameters = {}
        for parameter in proto.parameters:
            value = ParameterValueConverter.from_proto(parameter)
            if value is not None:
                if parameter.parameter_id in parameters:
                    raise ValueError(
                        "Invalid trial proto contains duplicate parameter {}"
                        ": {}".format(parameter.parameter_id, proto)
                    )
                parameters[parameter.parameter_id] = value
            else:
                logging.warning(
                    "A parameter without a value will be dropped: %s", parameter
                )

        final_measurement = None
        if proto.final_measurement:
            final_measurement = MeasurementConverter.from_proto(proto.final_measurement)

        completion_time = None
        infeasibility_reason = None
        if proto.state == study_pb2.Trial.State.SUCCEEDED:
            if proto.end_time:
                completion_time = (
                    proto.end_time.timestamp_pb()
                    .ToDatetime()
                    .replace(tzinfo=timezone.utc)
                )
        elif proto.state == study_pb2.Trial.State.INFEASIBLE:
            infeasibility_reason = proto.infeasible_reason

        measurements = []
        for measure in proto.measurements:
            measurements.append(MeasurementConverter.from_proto(measure))

        creation_time = None
        if proto.start_time:
            creation_time = (
                proto.start_time.timestamp_pb()
                .ToDatetime()
                .replace(tzinfo=timezone.utc)
            )
        return Trial(
            id=int(proto.name.split("/")[-1]),
            description=proto.name,
            assigned_worker=proto.client_id or None,
            is_requested=proto.state == study_pb2.Trial.State.REQUESTED,
            stopping_reason=(
                "stopping reason not supported yet"
                if proto.state == study_pb2.Trial.State.STOPPING
                else None
            ),
            parameters=parameters,
            creation_time=creation_time,
            completion_time=completion_time,
            infeasibility_reason=infeasibility_reason,
            final_measurement=final_measurement,
            measurements=measurements,
        )  # pytype: disable=wrong-arg-types

    @classmethod
    def from_protos(cls, protos: Sequence[study_pb2.Trial]) -> List[Trial]:
        """Convenience wrapper for from_proto."""
        return [TrialConverter.from_proto(proto) for proto in protos]

    @classmethod
    def to_protos(cls, pytrials: Sequence[Trial]) -> List[study_pb2.Trial]:
        return [TrialConverter.to_proto(pytrial) for pytrial in pytrials]

    @classmethod
    def to_proto(cls, pytrial: Trial) -> study_pb2.Trial:
        """Converts a pyvizier Trial to a Trial proto."""
        proto = study_pb2.Trial()
        if pytrial.description is not None:
            proto.name = pytrial.description
        proto.id = str(pytrial.id)
        proto.state = _from_pyvizier_trial_status(pytrial.status, pytrial.infeasible)
        proto.client_id = pytrial.assigned_worker or ""

        for name, value in pytrial.parameters.items():
            proto.parameters.append(ParameterValueConverter.to_proto(value, name))

        # pytrial always adds an empty metric. Ideally, we should remove it if the
        # metric does not exist in the study config.
        # setattr() is required here as `proto.final_measurement.CopyFrom`
        # raises AttributeErrors when setting the field on the pb2 compat types.
        if pytrial.final_measurement is not None:
            setattr(
                proto,
                "final_measurement",
                MeasurementConverter.to_proto(pytrial.final_measurement),
            )

        for measurement in pytrial.measurements:
            proto.measurements.append(MeasurementConverter.to_proto(measurement))

        if pytrial.creation_time is not None:
            start_time = timestamp_pb2.Timestamp()
            start_time.FromDatetime(pytrial.creation_time)
            setattr(proto, "start_time", start_time)
        if pytrial.completion_time is not None:
            end_time = timestamp_pb2.Timestamp()
            end_time.FromDatetime(pytrial.completion_time)
            setattr(proto, "end_time", end_time)
        if pytrial.infeasibility_reason is not None:
            proto.infeasible_reason = pytrial.infeasibility_reason
        return proto
