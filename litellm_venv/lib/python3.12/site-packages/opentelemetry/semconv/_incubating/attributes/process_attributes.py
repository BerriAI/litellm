# Copyright The OpenTelemetry Authors
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


from enum import Enum

PROCESS_COMMAND = "process.command"
"""
The command used to launch the process (i.e. the command name). On Linux based systems, can be set to the zeroth string in `proc/[pid]/cmdline`. On Windows, can be set to the first parameter extracted from `GetCommandLineW`.
"""

PROCESS_COMMAND_ARGS = "process.command_args"
"""
All the command arguments (including the command/executable itself) as received by the process. On Linux-based systems (and some other Unixoid systems supporting procfs), can be set according to the list of null-delimited strings extracted from `proc/[pid]/cmdline`. For libc-based executables, this would be the full argv vector passed to `main`.
"""

PROCESS_COMMAND_LINE = "process.command_line"
"""
The full command used to launch the process as a single string representing the full command. On Windows, can be set to the result of `GetCommandLineW`. Do not set this if you have to assemble it just for monitoring; use `process.command_args` instead.
"""

PROCESS_CONTEXT_SWITCH_TYPE = "process.context_switch_type"
"""
Specifies whether the context switches for this data point were voluntary or involuntary.
"""

PROCESS_CPU_STATE = "process.cpu.state"
"""
The CPU state for this data point. A process SHOULD be characterized _either_ by data points with no `state` labels, _or only_ data points with `state` labels.
"""

PROCESS_EXECUTABLE_NAME = "process.executable.name"
"""
The name of the process executable. On Linux based systems, can be set to the `Name` in `proc/[pid]/status`. On Windows, can be set to the base name of `GetProcessImageFileNameW`.
"""

PROCESS_EXECUTABLE_PATH = "process.executable.path"
"""
The full path to the process executable. On Linux based systems, can be set to the target of `proc/[pid]/exe`. On Windows, can be set to the result of `GetProcessImageFileNameW`.
"""

PROCESS_OWNER = "process.owner"
"""
The username of the user that owns the process.
"""

PROCESS_PAGING_FAULT_TYPE = "process.paging.fault_type"
"""
The type of page fault for this data point. Type `major` is for major/hard page faults, and `minor` is for minor/soft page faults.
"""

PROCESS_PARENT_PID = "process.parent_pid"
"""
Parent Process identifier (PPID).
"""

PROCESS_PID = "process.pid"
"""
Process identifier (PID).
"""

PROCESS_RUNTIME_DESCRIPTION = "process.runtime.description"
"""
An additional description about the runtime of the process, for example a specific vendor customization of the runtime environment.
"""

PROCESS_RUNTIME_NAME = "process.runtime.name"
"""
The name of the runtime of this process. For compiled native binaries, this SHOULD be the name of the compiler.
"""

PROCESS_RUNTIME_VERSION = "process.runtime.version"
"""
The version of the runtime of this process, as returned by the runtime without modification.
"""


class ProcessContextSwitchTypeValues(Enum):
    VOLUNTARY = "voluntary"
    """voluntary."""
    INVOLUNTARY = "involuntary"
    """involuntary."""


class ProcessCpuStateValues(Enum):
    SYSTEM = "system"
    """system."""
    USER = "user"
    """user."""
    WAIT = "wait"
    """wait."""


class ProcessPagingFaultTypeValues(Enum):
    MAJOR = "major"
    """major."""
    MINOR = "minor"
    """minor."""
