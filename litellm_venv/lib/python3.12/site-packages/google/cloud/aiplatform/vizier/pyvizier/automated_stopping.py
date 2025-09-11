"""Convenience classes for configuring Vizier Early-Stopping Configs."""
import copy
from typing import Union

import attr

from google.cloud.aiplatform.compat.types import study as study_pb2

AutomatedStoppingConfigProto = Union[
    study_pb2.StudySpec.DecayCurveAutomatedStoppingSpec,
    study_pb2.StudySpec.MedianAutomatedStoppingSpec,
]


@attr.s(frozen=True, init=True, slots=True, kw_only=True)
class AutomatedStoppingConfig:
    """A wrapper for study_pb2.automated_stopping_spec."""

    _proto: AutomatedStoppingConfigProto = attr.ib(init=True, kw_only=True)

    @classmethod
    def decay_curve_stopping_config(cls, use_steps: bool) -> "AutomatedStoppingConfig":
        """Create a DecayCurve automated stopping config.

        Vizier will early stop the Trial if it predicts the Trial objective value
        will not be better than previous Trials.

        Args:
          use_steps: Bool. If set, use Measurement.step_count as the measure of
            training progress.  Otherwise, use Measurement.elapsed_duration.

        Returns:
          AutomatedStoppingConfig object.

        Raises:
          ValueError: If more than one metric is configured.
          Note that Vizier Early Stopping currently only supports single-objective
          studies.
        """
        config = study_pb2.StudySpec.DecayCurveAutomatedStoppingSpec(
            use_elapsed_duration=not use_steps
        )
        return cls(proto=config)

    @classmethod
    def median_automated_stopping_config(
        cls, use_steps: bool
    ) -> "AutomatedStoppingConfig":
        """Create a Median automated stopping config.

        Vizier will early stop the Trial if it predicts the Trial objective value
        will not be better than previous Trials.

        Args:
          use_steps: Bool. If set, use Measurement.step_count as the measure of
            training progress.  Otherwise, use Measurement.elapsed_duration.

        Returns:
          AutomatedStoppingConfig object.

        Raises:
          ValueError: If more than one metric is configured.
          Note that Vizier Early Stopping currently only supports single-objective
          studies.
        """
        config = study_pb2.StudySpec.MedianAutomatedStoppingSpec(
            use_elapsed_duration=not use_steps
        )
        return cls(proto=config)

    @classmethod
    def from_proto(
        cls, proto: AutomatedStoppingConfigProto
    ) -> "AutomatedStoppingConfig":
        return cls(proto=proto)

    def to_proto(self) -> AutomatedStoppingConfigProto:
        """Returns this object as a proto."""
        return copy.deepcopy(self._proto)
