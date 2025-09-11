# Copyright 2017 Google Inc.
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

"""Utilities for Google Media Downloads and Resumable Uploads.

This package has some general purposes modules, e.g.
:mod:`~google.resumable_media.common`, but the majority of the
public interface will be contained in subpackages.

===========
Subpackages
===========

Each subpackage is tailored to a specific transport library:

* the :mod:`~google.resumable_media.requests` subpackage uses the ``requests``
  transport library.

.. _requests: http://docs.python-requests.org/

==========
Installing
==========

To install with `pip`_:

.. code-block:: console

  $ pip install --upgrade google-resumable-media

.. _pip: https://pip.pypa.io/
"""


from google.resumable_media.common import DataCorruption
from google.resumable_media.common import InvalidResponse
from google.resumable_media.common import PERMANENT_REDIRECT
from google.resumable_media.common import RetryStrategy
from google.resumable_media.common import TOO_MANY_REQUESTS
from google.resumable_media.common import UPLOAD_CHUNK_SIZE


__all__ = [
    "DataCorruption",
    "InvalidResponse",
    "PERMANENT_REDIRECT",
    "RetryStrategy",
    "TOO_MANY_REQUESTS",
    "UPLOAD_CHUNK_SIZE",
]
