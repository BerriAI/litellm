"""
Types and field definitions for coordination Redis settings management endpoints
"""

from typing import Literal, Optional

from pydantic import BaseModel

CoordinationRedisSection = Literal["connection", "cluster", "sentinel"]

CoordinationRedisSource = Literal["coordination_redis", "cache_backend", "environment"]


class CoordinationRedisSettingsField(BaseModel):
    field_name: str
    field_type: str
    field_value: Optional[object] = None
    field_description: str
    field_default: Optional[object] = None
    ui_field_name: str
    section: CoordinationRedisSection


COORDINATION_REDIS_SETTINGS_FIELDS: list[CoordinationRedisSettingsField] = [
    CoordinationRedisSettingsField(
        field_name="host",
        field_type="String",
        field_description="Redis server hostname or IP address",
        ui_field_name="Host",
        section="connection",
    ),
    CoordinationRedisSettingsField(
        field_name="port",
        field_type="Integer",
        field_description="Redis server port number",
        field_default=6379,
        ui_field_name="Port",
        section="connection",
    ),
    CoordinationRedisSettingsField(
        field_name="username",
        field_type="String",
        field_description="Redis server username (if required)",
        ui_field_name="Username",
        section="connection",
    ),
    CoordinationRedisSettingsField(
        field_name="password",
        field_type="String",
        field_description="Redis server password",
        ui_field_name="Password",
        section="connection",
    ),
    CoordinationRedisSettingsField(
        field_name="url",
        field_type="String",
        field_description=(
            "Full Redis connection URL (e.g. redis://:password@host:6379/1). "
            "Set this instead of the discrete host/port/username/password fields."
        ),
        ui_field_name="Redis URL",
        section="connection",
    ),
    CoordinationRedisSettingsField(
        field_name="ssl",
        field_type="Boolean",
        field_description="Connect to Redis over TLS",
        field_default=False,
        ui_field_name="SSL",
        section="connection",
    ),
    CoordinationRedisSettingsField(
        field_name="startup_nodes",
        field_type="List",
        field_description=(
            "Cluster-mode startup nodes (e.g. [{'host': '127.0.0.1', 'port': 7001}]). "
            "When set, a Redis Cluster client is used."
        ),
        ui_field_name="Cluster Startup Nodes",
        section="cluster",
    ),
    CoordinationRedisSettingsField(
        field_name="sentinel_nodes",
        field_type="List",
        field_description=(
            "Sentinel [host, port] pairs (e.g. [['localhost', 26379]]). When set, a Sentinel-managed client is used."
        ),
        ui_field_name="Sentinel Nodes",
        section="sentinel",
    ),
    CoordinationRedisSettingsField(
        field_name="sentinel_password",
        field_type="String",
        field_description="Password for the Redis Sentinel nodes",
        ui_field_name="Sentinel Password",
        section="sentinel",
    ),
    CoordinationRedisSettingsField(
        field_name="service_name",
        field_type="String",
        field_description="Master service name for Redis Sentinel",
        ui_field_name="Service Name",
        section="sentinel",
    ),
]
