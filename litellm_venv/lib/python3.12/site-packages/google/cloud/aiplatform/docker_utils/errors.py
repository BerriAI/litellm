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

import textwrap
from typing import List, NoReturn


class Error(Exception):
    """A base exception for all user recoverable errors."""

    def __init__(self, *args, **kwargs):
        """Initialize an Error."""
        self.exit_code = kwargs.get("exit_code", 1)


class DockerError(Error):
    """Exception that passes info on a failed Docker command."""

    def __init__(self, message, cmd, exit_code):
        super(DockerError, self).__init__(message)
        self.message = message
        self.cmd = cmd
        self.exit_code = exit_code


def raise_docker_error_with_command(command: List[str], return_code: int) -> NoReturn:
    """Raises DockerError with the given command and return code.

    Args:
        command (List(str)):
            Required. The docker command that fails.
        return_code (int):
            Required. The return code from the command.

    Raises:
        DockerError which error message populated by the given command and return code.
    """
    error_msg = textwrap.dedent(
        """
        Docker failed with error code {code}.
        Command: {cmd}
        """.format(
            code=return_code, cmd=" ".join(command)
        )
    )
    raise DockerError(error_msg, command, return_code)
