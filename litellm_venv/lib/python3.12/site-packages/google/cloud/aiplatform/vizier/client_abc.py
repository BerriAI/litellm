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
"""Cross-platform Vizier client interfaces.

Aside from "materialize_" methods, code written using these interfaces are
compatible with OSS and Cloud Vertex Vizier. Note importantly that subclasses
may have more methods than what is required by interfaces, and such methods
are not cross compatible. Our recommendation is to explicitly type your objects
to be `StudyInterface` or `TrialInterface` when you want to guarantee that
a code block is cross-platform.

Keywords:

#Materialize: The method returns a deep copy of the underlying pyvizier object.
Modifying the returned object does not update the Vizier service.
"""

from __future__ import annotations

from typing import Optional, Collection, Type, TypeVar, Mapping, Any
import abc

from google.cloud.aiplatform.vizier import pyvizier as vz

_T = TypeVar("_T")


class ResourceNotFoundError(LookupError):
    """Error raised by Vizier clients when resource is not found."""

    pass


class TrialInterface(abc.ABC):
    """Responsible for trial-level operations."""

    @property
    @abc.abstractmethod
    def uid(self) -> int:
        """Unique identifier of the trial."""

    @property
    @abc.abstractmethod
    def parameters(self) -> Mapping[str, Any]:
        """Parameters of the trial."""

    @property
    @abc.abstractmethod
    def status(self) -> vz.TrialStatus:
        """Trial's status."""

    @abc.abstractmethod
    def delete(self) -> None:
        """Delete the Trial in Vizier service.

        There is currently no promise on how this object behaves after `delete()`.
        If you are sharing a Trial object in parallel processes, proceed with
        caution.
        """

    @abc.abstractmethod
    def complete(
        self,
        measurement: Optional[vz.Measurement] = None,
        *,
        infeasible_reason: Optional[str] = None,
    ) -> Optional[vz.Measurement]:
        """Completes the trial and #materializes the measurement.

        * If `measurement` is provided, then Vizier writes it as the trial's final
        measurement and returns it.
        * If `infeasible_reason` is provided, `measurement` is not needed.
        * If neither is provided, then Vizier selects an existing (intermediate)
        measurement to be the final measurement and returns it.

        Args:
          measurement: Final measurement.
          infeasible_reason: Infeasible reason for missing final measurement.

        Returns:
          The final measurement of the trial, or None if the trial is marked
          infeasible.

        Raises:
          ValueError: If neither `measurement` nor `infeasible_reason` is provided
            but the trial does not contain any intermediate measurements.
        """

    @abc.abstractmethod
    def should_stop(self) -> bool:
        """Returns true if the trial should stop."""

    @abc.abstractmethod
    def add_measurement(self, measurement: vz.Measurement) -> None:
        """Adds an intermediate measurement."""

    @abc.abstractmethod
    def materialize(self, *, include_all_measurements: bool = True) -> vz.Trial:
        """#Materializes the Trial.

        Args:
          include_all_measurements: If True, returned Trial includes all
            intermediate measurements. The final measurement is always provided.

        Returns:
          Trial object.
        """


class StudyInterface(abc.ABC):
    """Responsible for study-level operations."""

    @abc.abstractmethod
    def create_or_load(
        self, display_name: str, problem: vz.ProblemStatement
    ) -> StudyInterface:
        """ """

    @abc.abstractmethod
    def suggest(
        self, *, count: Optional[int] = None, worker: str = ""
    ) -> Collection[TrialInterface]:
        """Returns Trials to be evaluated by worker.

        Args:
          count: Number of suggestions.
          worker: When new Trials are generated, their `assigned_worker` field is
            populated with this worker. suggest() first looks for existing Trials
            that are assigned to `worker`, before generating new ones.

        Returns:
          Trials.
        """

    @abc.abstractmethod
    def delete(self) -> None:
        """Deletes the study."""

    @abc.abstractmethod
    def trials(
        self, trial_filter: Optional[vz.TrialFilter] = None
    ) -> Collection[TrialInterface]:
        """Fetches a collection of trials."""

    @abc.abstractmethod
    def get_trial(self, uid: int) -> TrialInterface:
        """Fetches a single trial.

        Args:
          uid: Unique identifier of the trial within study.

        Returns:
          Trial.

        Raises:
          ResourceNotFoundError: If trial does not exist.
        """

    @abc.abstractmethod
    def optimal_trials(self) -> Collection[TrialInterface]:
        """Returns optimal trial(s)."""

    @abc.abstractmethod
    def materialize_study_config(self) -> vz.StudyConfig:
        """#Materializes the study config."""

    @abc.abstractclassmethod
    def from_uid(cls: Type[_T], uid: str) -> _T:
        """Fetches an existing study from the Vizier service.

        Args:
          uid: Unique identifier of the study.

        Returns:
          Study.

        Raises:
          ResourceNotFoundError: If study does not exist.
        """
