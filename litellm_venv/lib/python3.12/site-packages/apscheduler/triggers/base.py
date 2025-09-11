from abc import ABCMeta, abstractmethod
from datetime import timedelta
import random

import six


class BaseTrigger(six.with_metaclass(ABCMeta)):
    """Abstract base class that defines the interface that every trigger must implement."""

    __slots__ = ()

    @abstractmethod
    def get_next_fire_time(self, previous_fire_time, now):
        """
        Returns the next datetime to fire on, If no such datetime can be calculated, returns
        ``None``.

        :param datetime.datetime previous_fire_time: the previous time the trigger was fired
        :param datetime.datetime now: current datetime
        """

    def _apply_jitter(self, next_fire_time, jitter, now):
        """
        Randomize ``next_fire_time`` by adding a random value (the jitter).

        :param datetime.datetime|None next_fire_time: next fire time without jitter applied. If
            ``None``, returns ``None``.
        :param int|None jitter: maximum number of seconds to add to ``next_fire_time``
            (if ``None`` or ``0``, returns ``next_fire_time``)
        :param datetime.datetime now: current datetime
        :return datetime.datetime|None: next fire time with a jitter.
        """
        if next_fire_time is None or not jitter:
            return next_fire_time

        return next_fire_time + timedelta(seconds=random.uniform(0, jitter))
