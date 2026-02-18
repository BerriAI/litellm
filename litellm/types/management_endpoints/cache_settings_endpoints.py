"""
Types and field definitions for cache settings management endpoints
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class CacheSettingsField(BaseModel):
    field_name: str
    field_type: str
    field_value: Any
    field_description: str
    field_default: Any = None
    options: Optional[List[str]] = None  # For fields with predefined options/enum values
    ui_field_name: str  # User-friendly display name
    link: Optional[str] = None  # Documentation link for the field
    redis_type: Optional[str] = None  # Which Redis type this field applies to (node, cluster, sentinel)


# Redis type descriptions
REDIS_TYPE_DESCRIPTIONS: Dict[str, str] = {
    "node": "Standard Redis node/single instance",
    "cluster": "Redis Cluster mode for high availability and horizontal scaling",
    "sentinel": "Redis Sentinel mode for high availability with automatic failover",
}


# Define all available cache settings fields
CACHE_SETTINGS_FIELDS: List[CacheSettingsField] = [
    CacheSettingsField(
        field_name="redis_type",
        field_type="String",
        field_value=None,
        field_description="Type of Redis deployment",
        field_default="node",
        options=["node", "cluster", "sentinel"],
        ui_field_name="Redis Type",
        redis_type=None,
    ),
    # Common fields for all Redis types
    CacheSettingsField(
        field_name="host",
        field_type="String",
        field_value=None,
        field_description="Redis server hostname or IP address",
        field_default=None,
        ui_field_name="Host",
        redis_type=None,
    ),
    CacheSettingsField(
        field_name="port",
        field_type="String",
        field_value=None,
        field_description="Redis server port number",
        field_default="6379",
        ui_field_name="Port",
        redis_type=None,
    ),
    CacheSettingsField(
        field_name="password",
        field_type="String",
        field_value=None,
        field_description="Redis server password",
        field_default=None,
        ui_field_name="Password",
        redis_type=None,
    ),
    CacheSettingsField(
        field_name="username",
        field_type="String",
        field_value=None,
        field_description="Redis server username (if required)",
        field_default=None,
        ui_field_name="Username",
        redis_type=None,
    ),
    CacheSettingsField(
        field_name="ssl",
        field_type="Boolean",
        field_value=None,
        field_description="Enable SSL/TLS connection",
        field_default=False,
        ui_field_name="SSL",
        redis_type=None,
    ),
    CacheSettingsField(
        field_name="namespace",
        field_type="String",
        field_value=None,
        field_description="Namespace prefix for cache keys",
        field_default=None,
        ui_field_name="Namespace",
        redis_type=None,
    ),
    CacheSettingsField(
        field_name="ttl",
        field_type="Float",
        field_value=None,
        field_description="Time-to-live for cached items in seconds",
        field_default=None,
        ui_field_name="TTL (seconds)",
        redis_type=None,
    ),
    CacheSettingsField(
        field_name="max_connections",
        field_type="Integer",
        field_value=None,
        field_description="Maximum number of connections in the connection pool",
        field_default=None,
        ui_field_name="Max Connections",
        redis_type=None,
    ),
    # Cluster-specific fields
    CacheSettingsField(
        field_name="redis_startup_nodes",
        field_type="List",
        field_value=None,
        field_description="List of startup nodes for Redis Cluster (e.g., [{'host': '127.0.0.1', 'port': '7001'}])",
        field_default=None,
        ui_field_name="Startup Nodes",
        redis_type="cluster",
    ),
    # Sentinel-specific fields
    CacheSettingsField(
        field_name="sentinel_nodes",
        field_type="List",
        field_value=None,
        field_description="List of Sentinel nodes (e.g., [['localhost', 26379]])",
        field_default=None,
        ui_field_name="Sentinel Nodes",
        redis_type="sentinel",
    ),
    CacheSettingsField(
        field_name="service_name",
        field_type="String",
        field_value=None,
        field_description="Master service name for Redis Sentinel",
        field_default=None,
        ui_field_name="Service Name",
        redis_type="sentinel",
    ),
    CacheSettingsField(
        field_name="sentinel_password",
        field_type="String",
        field_value=None,
        field_description="Password for Redis Sentinel authentication",
        field_default=None,
        ui_field_name="Sentinel Password",
        redis_type="sentinel",
    ),
    # Semantic-specific fields
    CacheSettingsField(
        field_name="similarity_threshold",
        field_type="Float",
        field_value=None,
        field_description="Similarity threshold for semantic cache",
        field_default=0.8,
        ui_field_name="Similarity Threshold",
        redis_type="semantic",
    ),
    CacheSettingsField(
        field_name="redis_semantic_cache_embedding_model",
        field_type="Models_Select",
        field_value=None,
        field_description="Embedding model for semantic cache",
        field_default=None,
        ui_field_name="Embedding Model",
        redis_type="semantic",
    ),
    CacheSettingsField(
        field_name="redis_semantic_cache_embedding_dimensions",
        field_type="Integer",
        field_value=None,
        field_description="Explicit embedding dimensions for Redis semantic cache (optional, auto-detected if not provided)",
        field_default=None,
        ui_field_name="Embedding Dimensions",
        redis_type="semantic",
    ),
    # Qdrant semantic cache fields
    CacheSettingsField(
        field_name="qdrant_api_base",
        field_type="String",
        field_value=None,
        field_description="Qdrant API base URL",
        field_default=None,
        ui_field_name="Qdrant API URL",
        redis_type=None,
    ),
    CacheSettingsField(
        field_name="qdrant_api_key",
        field_type="String",
        field_value=None,
        field_description="Qdrant API key",
        field_default=None,
        ui_field_name="Qdrant API Key",
        redis_type=None,
    ),
    CacheSettingsField(
        field_name="qdrant_collection_name",
        field_type="String",
        field_value=None,
        field_description="Qdrant collection name for semantic cache",
        field_default=None,
        ui_field_name="Qdrant Collection Name",
        redis_type=None,
    ),
    CacheSettingsField(
        field_name="qdrant_quantization_config",
        field_type="String",
        field_value=None,
        field_description="Quantization config for Qdrant (binary, scalar, or product)",
        field_default="binary",
        options=["binary", "scalar", "product"],
        ui_field_name="Qdrant Quantization",
        redis_type=None,
    ),
    CacheSettingsField(
        field_name="qdrant_semantic_cache_embedding_model",
        field_type="Models_Select",
        field_value=None,
        field_description="Embedding model for Qdrant semantic cache",
        field_default=None,
        ui_field_name="Qdrant Embedding Model",
        redis_type=None,
    ),
    CacheSettingsField(
        field_name="qdrant_semantic_cache_embedding_dimensions",
        field_type="Integer",
        field_value=None,
        field_description="Explicit embedding dimensions for Qdrant semantic cache (defaults to 1536 if not provided)",
        field_default=1536,
        ui_field_name="Qdrant Embedding Dimensions",
        redis_type=None,
    ),
    # GCP IAM authentication fields
    CacheSettingsField(
        field_name="gcp_service_account",
        field_type="String",
        field_value=None,
        field_description="GCP service account for IAM authentication (e.g., projects/-/serviceAccounts/your-sa@project.iam.gserviceaccount.com)",
        field_default=None,
        ui_field_name="GCP Service Account",
        redis_type=None,
    ),
    CacheSettingsField(
        field_name="gcp_ssl_ca_certs",
        field_type="String",
        field_value=None,
        field_description="Path to SSL CA certificate file for GCP Memorystore Redis",
        field_default=None,
        ui_field_name="GCP SSL CA Certs",
        redis_type=None,
    ),
    CacheSettingsField(
        field_name="ssl_cert_reqs",
        field_type="String",
        field_value=None,
        field_description="SSL certificate requirements (None, CERT_REQUIRED, CERT_OPTIONAL)",
        field_default=None,
        ui_field_name="SSL Cert Reqs",
        redis_type=None,
    ),
    CacheSettingsField(
        field_name="ssl_check_hostname",
        field_type="Boolean",
        field_value=None,
        field_description="Enable SSL hostname verification",
        field_default=None,
        ui_field_name="SSL Check Hostname",
        redis_type=None,
    ),
]

