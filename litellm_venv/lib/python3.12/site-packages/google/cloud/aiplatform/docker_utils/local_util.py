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

import io
import logging
import subprocess
from typing import List, Optional

_logger = logging.getLogger(__name__)


def execute_command(
    cmd: List[str],
    input_str: Optional[str] = None,
) -> int:
    """Executes commands in subprocess.

    Executes the supplied command with the supplied standard input string, streams
    the output to stdout, and returns the process's return code.

    Args:
        cmd (List[str]):
            Required. The strings to send in as the command.
        input_str (str):
            Optional. If supplied, it will be passed as stdin to the supplied command.
            If None, stdin will get closed immediately.

    Returns:
        Return code of the process.
    """
    with subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=False,
        bufsize=1,
    ) as p:
        if input_str:
            p.stdin.write(input_str.encode("utf-8"))
        p.stdin.close()

        out = io.TextIOWrapper(p.stdout, newline="", encoding="utf-8", errors="replace")

        for line in out:
            _logger.info(line)

    return p.returncode
