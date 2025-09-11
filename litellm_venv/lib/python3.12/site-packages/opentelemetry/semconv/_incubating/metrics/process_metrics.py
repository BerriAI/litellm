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


from typing import Callable, Sequence

from opentelemetry.metrics import (
    Counter,
    Meter,
    ObservableGauge,
    UpDownCounter,
)

PROCESS_CONTEXT_SWITCHES = "process.context_switches"
"""
Number of times the process has been context switched
Instrument: counter
Unit: {count}
"""


def create_process_context_switches(meter: Meter) -> Counter:
    """Number of times the process has been context switched"""
    return meter.create_counter(
        name="process.context_switches",
        description="Number of times the process has been context switched.",
        unit="{count}",
    )


PROCESS_CPU_TIME = "process.cpu.time"
"""
Total CPU seconds broken down by different states
Instrument: counter
Unit: s
"""


def create_process_cpu_time(meter: Meter) -> Counter:
    """Total CPU seconds broken down by different states"""
    return meter.create_counter(
        name="process.cpu.time",
        description="Total CPU seconds broken down by different states.",
        unit="s",
    )


PROCESS_CPU_UTILIZATION = "process.cpu.utilization"
"""
Difference in process.cpu.time since the last measurement, divided by the elapsed time and number of CPUs available to the process
Instrument: gauge
Unit: 1
"""


def create_process_cpu_utilization(
    meter: Meter, callback: Sequence[Callable]
) -> ObservableGauge:
    """Difference in process.cpu.time since the last measurement, divided by the elapsed time and number of CPUs available to the process"""
    return meter.create_observable_gauge(
        name="process.cpu.utilization",
        callback=callback,
        description="Difference in process.cpu.time since the last measurement, divided by the elapsed time and number of CPUs available to the process.",
        unit="1",
    )


PROCESS_DISK_IO = "process.disk.io"
"""
Disk bytes transferred
Instrument: counter
Unit: By
"""


def create_process_disk_io(meter: Meter) -> Counter:
    """Disk bytes transferred"""
    return meter.create_counter(
        name="process.disk.io",
        description="Disk bytes transferred.",
        unit="By",
    )


PROCESS_MEMORY_USAGE = "process.memory.usage"
"""
The amount of physical memory in use
Instrument: updowncounter
Unit: By
"""


def create_process_memory_usage(meter: Meter) -> UpDownCounter:
    """The amount of physical memory in use"""
    return meter.create_up_down_counter(
        name="process.memory.usage",
        description="The amount of physical memory in use.",
        unit="By",
    )


PROCESS_MEMORY_VIRTUAL = "process.memory.virtual"
"""
The amount of committed virtual memory
Instrument: updowncounter
Unit: By
"""


def create_process_memory_virtual(meter: Meter) -> UpDownCounter:
    """The amount of committed virtual memory"""
    return meter.create_up_down_counter(
        name="process.memory.virtual",
        description="The amount of committed virtual memory.",
        unit="By",
    )


PROCESS_NETWORK_IO = "process.network.io"
"""
Network bytes transferred
Instrument: counter
Unit: By
"""


def create_process_network_io(meter: Meter) -> Counter:
    """Network bytes transferred"""
    return meter.create_counter(
        name="process.network.io",
        description="Network bytes transferred.",
        unit="By",
    )


PROCESS_OPEN_FILE_DESCRIPTOR_COUNT = "process.open_file_descriptor.count"
"""
Number of file descriptors in use by the process
Instrument: updowncounter
Unit: {count}
"""


def create_process_open_file_descriptor_count(meter: Meter) -> UpDownCounter:
    """Number of file descriptors in use by the process"""
    return meter.create_up_down_counter(
        name="process.open_file_descriptor.count",
        description="Number of file descriptors in use by the process.",
        unit="{count}",
    )


PROCESS_PAGING_FAULTS = "process.paging.faults"
"""
Number of page faults the process has made
Instrument: counter
Unit: {fault}
"""


def create_process_paging_faults(meter: Meter) -> Counter:
    """Number of page faults the process has made"""
    return meter.create_counter(
        name="process.paging.faults",
        description="Number of page faults the process has made.",
        unit="{fault}",
    )


PROCESS_THREAD_COUNT = "process.thread.count"
"""
Process threads count
Instrument: updowncounter
Unit: {thread}
"""


def create_process_thread_count(meter: Meter) -> UpDownCounter:
    """Process threads count"""
    return meter.create_up_down_counter(
        name="process.thread.count",
        description="Process threads count.",
        unit="{thread}",
    )
