# Copyright 2017 Google LLC
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

"""Helpers for deprecated code and modules."""

import sys
import warnings


if sys.version_info < (3, 8):
    import importlib_metadata as metadata
else:
    import importlib.metadata as metadata


def complain(distribution_name):
    """Issue a warning if `distribution_name` is installed.

    In a future release, this method will be updated to raise ImportError
    rather than just send a warning.

    Args:
        distribution_name (str): The name of the obsolete distribution.
    """
    try:
        metadata.distribution(distribution_name)
        warnings.warn(
            "The {pkg} distribution is now obsolete. "
            "Please `pip uninstall {pkg}`. "
            "In the future, this warning will become an ImportError.".format(
                pkg=distribution_name
            ),
            DeprecationWarning,
        )
    except metadata.PackageNotFoundError:
        pass
