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

from deprecated import deprecated

NETWORK_CARRIER_ICC = "network.carrier.icc"
"""
The ISO 3166-1 alpha-2 2-character country code associated with the mobile carrier network.
"""

NETWORK_CARRIER_MCC = "network.carrier.mcc"
"""
The mobile carrier country code.
"""

NETWORK_CARRIER_MNC = "network.carrier.mnc"
"""
The mobile carrier network code.
"""

NETWORK_CARRIER_NAME = "network.carrier.name"
"""
The name of the mobile carrier.
"""

NETWORK_CONNECTION_SUBTYPE = "network.connection.subtype"
"""
This describes more details regarding the connection.type. It may be the type of cell technology connection, but it could be used for describing details about a wifi connection.
"""

NETWORK_CONNECTION_TYPE = "network.connection.type"
"""
The internet connection type.
"""

NETWORK_IO_DIRECTION = "network.io.direction"
"""
The network IO operation direction.
"""

NETWORK_LOCAL_ADDRESS = "network.local.address"
"""
Deprecated in favor of stable :py:const:`opentelemetry.semconv.attributes.network_attributes.NETWORK_LOCAL_ADDRESS`.
"""

NETWORK_LOCAL_PORT = "network.local.port"
"""
Deprecated in favor of stable :py:const:`opentelemetry.semconv.attributes.network_attributes.NETWORK_LOCAL_PORT`.
"""

NETWORK_PEER_ADDRESS = "network.peer.address"
"""
Deprecated in favor of stable :py:const:`opentelemetry.semconv.attributes.network_attributes.NETWORK_PEER_ADDRESS`.
"""

NETWORK_PEER_PORT = "network.peer.port"
"""
Deprecated in favor of stable :py:const:`opentelemetry.semconv.attributes.network_attributes.NETWORK_PEER_PORT`.
"""

NETWORK_PROTOCOL_NAME = "network.protocol.name"
"""
Deprecated in favor of stable :py:const:`opentelemetry.semconv.attributes.network_attributes.NETWORK_PROTOCOL_NAME`.
"""

NETWORK_PROTOCOL_VERSION = "network.protocol.version"
"""
Deprecated in favor of stable :py:const:`opentelemetry.semconv.attributes.network_attributes.NETWORK_PROTOCOL_VERSION`.
"""

NETWORK_TRANSPORT = "network.transport"
"""
Deprecated in favor of stable :py:const:`opentelemetry.semconv.attributes.network_attributes.NETWORK_TRANSPORT`.
"""

NETWORK_TYPE = "network.type"
"""
Deprecated in favor of stable :py:const:`opentelemetry.semconv.attributes.network_attributes.NETWORK_TYPE`.
"""


class NetworkConnectionSubtypeValues(Enum):
    GPRS = "gprs"
    """GPRS."""
    EDGE = "edge"
    """EDGE."""
    UMTS = "umts"
    """UMTS."""
    CDMA = "cdma"
    """CDMA."""
    EVDO_0 = "evdo_0"
    """EVDO Rel. 0."""
    EVDO_A = "evdo_a"
    """EVDO Rev. A."""
    CDMA2000_1XRTT = "cdma2000_1xrtt"
    """CDMA2000 1XRTT."""
    HSDPA = "hsdpa"
    """HSDPA."""
    HSUPA = "hsupa"
    """HSUPA."""
    HSPA = "hspa"
    """HSPA."""
    IDEN = "iden"
    """IDEN."""
    EVDO_B = "evdo_b"
    """EVDO Rev. B."""
    LTE = "lte"
    """LTE."""
    EHRPD = "ehrpd"
    """EHRPD."""
    HSPAP = "hspap"
    """HSPAP."""
    GSM = "gsm"
    """GSM."""
    TD_SCDMA = "td_scdma"
    """TD-SCDMA."""
    IWLAN = "iwlan"
    """IWLAN."""
    NR = "nr"
    """5G NR (New Radio)."""
    NRNSA = "nrnsa"
    """5G NRNSA (New Radio Non-Standalone)."""
    LTE_CA = "lte_ca"
    """LTE CA."""


class NetworkConnectionTypeValues(Enum):
    WIFI = "wifi"
    """wifi."""
    WIRED = "wired"
    """wired."""
    CELL = "cell"
    """cell."""
    UNAVAILABLE = "unavailable"
    """unavailable."""
    UNKNOWN = "unknown"
    """unknown."""


class NetworkIoDirectionValues(Enum):
    TRANSMIT = "transmit"
    """transmit."""
    RECEIVE = "receive"
    """receive."""


@deprecated(
    reason="Deprecated in favor of stable :py:const:`opentelemetry.semconv.attributes.network_attributes.NetworkTransportValues`."
)
class NetworkTransportValues(Enum):
    TCP = "tcp"
    """TCP."""
    UDP = "udp"
    """UDP."""
    PIPE = "pipe"
    """Named or anonymous pipe."""
    UNIX = "unix"
    """Unix domain socket."""


@deprecated(
    reason="Deprecated in favor of stable :py:const:`opentelemetry.semconv.attributes.network_attributes.NetworkTypeValues`."
)
class NetworkTypeValues(Enum):
    IPV4 = "ipv4"
    """IPv4."""
    IPV6 = "ipv6"
    """IPv6."""
