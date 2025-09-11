# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import functools


def has_upb():
    try:
        from google._upb import _message  # pylint: disable=unused-import

        has_upb = True
    except ImportError:
        has_upb = False
    return has_upb


def cached_property(fx):
    """Make the callable into a cached property.

    Similar to @property, but the function will only be called once per
    object.

    Args:
        fx (Callable[]): The property function.

    Returns:
        Callable[]: The wrapped function.
    """

    @functools.wraps(fx)
    def inner(self):
        # Sanity check: If there is no cache at all, create an empty cache.
        if not hasattr(self, "_cached_values"):
            object.__setattr__(self, "_cached_values", {})

        # If and only if the function's result is not in the cache,
        # run the function.
        if fx.__name__ not in self._cached_values:
            self._cached_values[fx.__name__] = fx(self)

        # Return the value from cache.
        return self._cached_values[fx.__name__]

    return property(inner)


__all__ = ("cached_property",)
