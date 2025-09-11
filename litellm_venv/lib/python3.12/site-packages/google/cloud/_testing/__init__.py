# Copyright 2014 Google LLC
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

"""Shared testing utilities."""

from __future__ import absolute_import


class _Monkey(object):
    """Context-manager for replacing module names in the scope of a test."""

    def __init__(self, module, **kw):
        self.module = module
        if not kw:  # pragma: NO COVER
            raise ValueError("_Monkey was used with nothing to monkey-patch")
        self.to_restore = {key: getattr(module, key) for key in kw}
        for key, value in kw.items():
            setattr(module, key, value)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for key, value in self.to_restore.items():
            setattr(self.module, key, value)


class _NamedTemporaryFile(object):
    def __init__(self, suffix=""):
        import os
        import tempfile

        filehandle, self.name = tempfile.mkstemp(suffix=suffix)
        os.close(filehandle)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        import os

        os.remove(self.name)


def _tempdir_maker():
    import contextlib
    import shutil
    import tempfile

    @contextlib.contextmanager
    def _tempdir_mgr():
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    return _tempdir_mgr


# pylint: disable=invalid-name
# Retain _tempdir as a constant for backwards compatibility despite
# being an invalid name.
_tempdir = _tempdir_maker()
del _tempdir_maker
# pylint: enable=invalid-name


class _GAXBaseAPI(object):
    _random_gax_error = False

    def __init__(self, **kw):
        self.__dict__.update(kw)

    @staticmethod
    def _make_grpc_error(status_code, trailing=None):
        from grpc._channel import _RPCState
        from google.cloud.exceptions import GrpcRendezvous

        details = "Some error details."
        exc_state = _RPCState((), None, trailing, status_code, details)
        return GrpcRendezvous(exc_state, None, None, None)

    def _make_grpc_not_found(self):
        from grpc import StatusCode

        return self._make_grpc_error(StatusCode.NOT_FOUND)

    def _make_grpc_failed_precondition(self):
        from grpc import StatusCode

        return self._make_grpc_error(StatusCode.FAILED_PRECONDITION)

    def _make_grpc_already_exists(self):
        from grpc import StatusCode

        return self._make_grpc_error(StatusCode.ALREADY_EXISTS)

    def _make_grpc_deadline_exceeded(self):
        from grpc import StatusCode

        return self._make_grpc_error(StatusCode.DEADLINE_EXCEEDED)


class _GAXPageIterator(object):
    def __init__(self, *pages, **kwargs):
        self._pages = iter(pages)
        self.page_token = kwargs.get("page_token")

    def __next__(self):
        """Iterate to the next page."""
        return next(self._pages)
