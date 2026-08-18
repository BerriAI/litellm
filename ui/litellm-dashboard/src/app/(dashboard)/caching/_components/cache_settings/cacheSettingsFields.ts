export type CacheFieldType = "string" | "password" | "integer" | "float" | "boolean" | "list" | "model-select";

export type RedisType = "node" | "cluster" | "sentinel" | "semantic";

export type CacheSection = "connection" | "cluster" | "sentinel" | "semantic" | "ssl" | "cacheManagement" | "gcp";

export type CacheFieldRule = (value: unknown) => string | null;

// Marker the backend returns for a configured credential and maps back to the
// stored secret on save, so the plaintext never round-trips through the form.
export const REDACTED_VALUE = "***REDACTED***";

export interface CacheField {
  readonly name: string;
  readonly label: string;
  readonly type: CacheFieldType;
  readonly section: CacheSection;
  readonly helpText: string;
  readonly redisType: RedisType | null;
  readonly defaultValue?: string | number | boolean;
  readonly rules?: CacheFieldRule[];
  // Credential field: never prefilled into the form, and dropped from the save
  // payload when left untouched so the redacted marker is never persisted.
  readonly secret?: boolean;
}

export const REDIS_TYPES: readonly RedisType[] = ["node", "cluster", "sentinel", "semantic"];

export const REDIS_TYPE_DESCRIPTIONS: Readonly<Record<RedisType, string>> = {
  node: "Standard Redis node/single instance",
  cluster: "Redis Cluster mode for high availability and horizontal scaling",
  sentinel: "Redis Sentinel mode for high availability with automatic failover",
  semantic: "Semantic caching that reuses responses for similar prompts",
};

const isBlank = (value: unknown): boolean => value === undefined || value === null || String(value).trim() === "";

const portRule: CacheFieldRule = (value) => {
  if (isBlank(value)) {
    return null;
  }
  const port = Number(value);
  return Number.isInteger(port) && port >= 1 && port <= 65535 ? null : "Port must be an integer between 1 and 65535";
};

const jsonListRule: CacheFieldRule = (value) => {
  if (isBlank(value)) {
    return null;
  }
  let parsed: unknown;
  try {
    parsed = JSON.parse(String(value));
  } catch {
    return "Must be a valid JSON array (use double quotes)";
  }
  return Array.isArray(parsed) ? null : "Must be a JSON array";
};

const nonNegativeIntegerRule: CacheFieldRule = (value) => {
  if (isBlank(value)) {
    return null;
  }
  const parsed = Number(value);
  return Number.isInteger(parsed) && parsed >= 0 ? null : "Must be a non-negative integer";
};

const numberRule: CacheFieldRule = (value) => {
  if (isBlank(value)) {
    return null;
  }
  return Number.isNaN(Number(value)) ? "Must be a number" : null;
};

export const CACHE_FIELDS: readonly CacheField[] = [
  {
    name: "url",
    label: "Redis URL",
    type: "string",
    section: "connection",
    helpText:
      "Full Redis/Valkey connection URL (e.g. redis://:password@host:6379/1). When set, it takes precedence over Host, Port, Password, and Database Index.",
    redisType: null,
    secret: true,
  },
  {
    name: "host",
    label: "Host",
    type: "string",
    section: "connection",
    helpText: "Redis server hostname or IP address",
    redisType: null,
  },
  {
    name: "port",
    label: "Port",
    type: "string",
    section: "connection",
    helpText: "Redis server port number",
    redisType: null,
    defaultValue: "6379",
    rules: [portRule],
  },
  {
    name: "db",
    label: "Database Index",
    type: "integer",
    section: "connection",
    helpText: "Logical database index to isolate the cache (e.g. 1 for redis://host:6379/1)",
    redisType: null,
    rules: [nonNegativeIntegerRule],
  },
  {
    name: "password",
    label: "Password",
    type: "password",
    section: "connection",
    helpText: "Redis server password",
    redisType: null,
    secret: true,
  },
  {
    name: "username",
    label: "Username",
    type: "string",
    section: "connection",
    helpText: "Redis server username (if required)",
    redisType: null,
  },
  {
    name: "redis_startup_nodes",
    label: "Startup Nodes",
    type: "list",
    section: "cluster",
    helpText: 'List of startup nodes for Redis Cluster (e.g., [{"host": "127.0.0.1", "port": "7001"}])',
    redisType: "cluster",
    rules: [jsonListRule],
  },
  {
    name: "sentinel_nodes",
    label: "Sentinel Nodes",
    type: "list",
    section: "sentinel",
    helpText: 'List of Sentinel nodes (e.g., [["localhost", 26379]])',
    redisType: "sentinel",
    rules: [jsonListRule],
  },
  {
    name: "service_name",
    label: "Service Name",
    type: "string",
    section: "sentinel",
    helpText: "Master service name for Redis Sentinel",
    redisType: "sentinel",
  },
  {
    name: "sentinel_password",
    label: "Sentinel Password",
    type: "password",
    section: "sentinel",
    helpText: "Password for Redis Sentinel authentication",
    redisType: "sentinel",
    secret: true,
  },
  {
    name: "similarity_threshold",
    label: "Similarity Threshold",
    type: "float",
    section: "semantic",
    helpText: "Similarity threshold for semantic cache",
    redisType: "semantic",
    defaultValue: 0.8,
    rules: [numberRule],
  },
  {
    name: "redis_semantic_cache_embedding_model",
    label: "Embedding Model",
    type: "model-select",
    section: "semantic",
    helpText: "Embedding model for semantic cache",
    redisType: "semantic",
  },
  {
    name: "ssl",
    label: "SSL",
    type: "boolean",
    section: "ssl",
    helpText: "Enable SSL/TLS connection",
    redisType: null,
    defaultValue: false,
  },
  {
    name: "ssl_cert_reqs",
    label: "SSL Cert Reqs",
    type: "string",
    section: "ssl",
    helpText: "SSL certificate requirements (None, CERT_REQUIRED, CERT_OPTIONAL)",
    redisType: null,
  },
  {
    name: "ssl_check_hostname",
    label: "SSL Check Hostname",
    type: "boolean",
    section: "ssl",
    helpText: "Enable SSL hostname verification",
    redisType: null,
    defaultValue: false,
  },
  {
    name: "namespace",
    label: "Namespace",
    type: "string",
    section: "cacheManagement",
    helpText: "Namespace prefix for cache keys",
    redisType: null,
  },
  {
    name: "ttl",
    label: "TTL (seconds)",
    type: "float",
    section: "cacheManagement",
    helpText: "Time-to-live for cached items in seconds",
    redisType: null,
    rules: [numberRule],
  },
  {
    name: "max_connections",
    label: "Max Connections",
    type: "integer",
    section: "cacheManagement",
    helpText: "Maximum number of connections in the connection pool",
    redisType: null,
    rules: [nonNegativeIntegerRule],
  },
  {
    name: "gcp_service_account",
    label: "GCP Service Account",
    type: "string",
    section: "gcp",
    helpText:
      "GCP service account for IAM authentication (e.g., projects/-/serviceAccounts/your-sa@project.iam.gserviceaccount.com)",
    redisType: null,
  },
  {
    name: "gcp_ssl_ca_certs",
    label: "GCP SSL CA Certs",
    type: "string",
    section: "gcp",
    helpText: "Path to SSL CA certificate file for GCP Memorystore Redis",
    redisType: null,
  },
];
