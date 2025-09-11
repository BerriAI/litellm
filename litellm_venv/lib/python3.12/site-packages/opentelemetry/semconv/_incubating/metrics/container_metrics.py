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


from opentelemetry.metrics import Counter, Meter

CONTAINER_CPU_TIME = "container.cpu.time"
"""
Total CPU time consumed
Instrument: counter
Unit: s
Note: Total CPU time consumed by the specific container on all available CPU cores.
"""


def create_container_cpu_time(meter: Meter) -> Counter:
    """Total CPU time consumed"""
    return meter.create_counter(
        name="container.cpu.time",
        description="Total CPU time consumed",
        unit="s",
    )


CONTAINER_DISK_IO = "container.disk.io"
"""
Disk bytes for the container
Instrument: counter
Unit: By
Note: The total number of bytes read/written successfully (aggregated from all disks).
"""


def create_container_disk_io(meter: Meter) -> Counter:
    """Disk bytes for the container"""
    return meter.create_counter(
        name="container.disk.io",
        description="Disk bytes for the container.",
        unit="By",
    )


CONTAINER_MEMORY_USAGE = "container.memory.usage"
"""
Memory usage of the container
Instrument: counter
Unit: By
Note: Memory usage of the container.
"""


def create_container_memory_usage(meter: Meter) -> Counter:
    """Memory usage of the container"""
    return meter.create_counter(
        name="container.memory.usage",
        description="Memory usage of the container.",
        unit="By",
    )


CONTAINER_NETWORK_IO = "container.network.io"
"""
Network bytes for the container
Instrument: counter
Unit: By
Note: The number of bytes sent/received on all network interfaces by the container.
"""


def create_container_network_io(meter: Meter) -> Counter:
    """Network bytes for the container"""
    return meter.create_counter(
        name="container.network.io",
        description="Network bytes for the container.",
        unit="By",
    )
