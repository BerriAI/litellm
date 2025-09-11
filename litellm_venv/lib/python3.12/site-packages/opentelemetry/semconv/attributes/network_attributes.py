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

NETWORK_LOCAL_ADDRESS = "network.local.address"
"""
Local address of the network connection - IP address or Unix domain socket name.
"""

NETWORK_LOCAL_PORT = "network.local.port"
"""
Local port number of the network connection.
"""

NETWORK_PEER_ADDRESS = "network.peer.address"
"""
Peer address of the network connection - IP address or Unix domain socket name.
"""

NETWORK_PEER_PORT = "network.peer.port"
"""
Peer port number of the network connection.
"""

NETWORK_PROTOCOL_NAME = "network.protocol.name"
"""
[OSI application layer](https://osi-model.com/application-layer/) or non-OSI equivalent.
Note: The value SHOULD be normalized to lowercase.
"""

NETWORK_PROTOCOL_VERSION = "network.protocol.version"
"""
The actual version of the protocol used for network communication.
Note: If protocol version is subject to negotiation (for example using [ALPN](https://www.rfc-editor.org/rfc/rfc7301.html)), this attribute SHOULD be set to the negotiated version. If the actual protocol version is not known, this attribute SHOULD NOT be set.
"""

NETWORK_TRANSPORT = "network.transport"
"""
[OSI transport layer](https://osi-model.com/transport-layer/) or [inter-process communication method](https://wikipedia.org/wiki/Inter-process_communication).
Note: The value SHOULD be normalized to lowercase.

    Consider always setting the transport when setting a port number, since
    a port number is ambiguous without knowing the transport. For example
    different processes could be listening on TCP port 12345 and UDP port 12345.
"""

NETWORK_TYPE = "network.type"
"""
[OSI network layer](https://osi-model.com/network-layer/) or non-OSI equivalent.
Note: The value SHOULD be normalized to lowercase.
"""


class NetworkTransportValues(Enum):
    TCP = "tcp"
    """TCP."""
    UDP = "udp"
    """UDP."""
    PIPE = "pipe"
    """Named or anonymous pipe."""
    UNIX = "unix"
    """Unix domain socket."""


class NetworkTypeValues(Enum):
    IPV4 = "ipv4"
    """IPv4."""
    IPV6 = "ipv6"
    """IPv6."""
