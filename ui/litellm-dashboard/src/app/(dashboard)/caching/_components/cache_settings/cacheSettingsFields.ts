import type { ParseKeys } from "i18next";

export type CacheFieldType =
  | "string"
  | "password"
  | "integer"
  | "float"
  | "boolean"
  | "list"
  | "model-select"
  | "select";

export interface CacheFieldOption {
  readonly value: string;
  readonly label: ParseKeys;
}

export type RedisType = "node" | "cluster" | "sentinel" | "semantic";

export type CacheSection = "connection" | "cluster" | "sentinel" | "semantic" | "ssl" | "cacheManagement" | "gcp";

export type CacheFieldRule = (value: unknown) => ParseKeys | null;

// Marker the backend returns for a configured credential and maps back to the
// stored secret on save, so the plaintext never round-trips through the form.
export const REDACTED_VALUE = "***REDACTED***";

export interface CacheField {
  readonly name: string;
  readonly label: ParseKeys;
  readonly type: CacheFieldType;
  readonly section: CacheSection;
  readonly helpText: ParseKeys;
  readonly redisType: RedisType | null;
  readonly defaultValue?: string | number | boolean;
  readonly options?: readonly CacheFieldOption[];
  readonly rules?: CacheFieldRule[];
  // Credential field: never prefilled into the form, and dropped from the save
  // payload when left untouched so the redacted marker is never persisted.
  readonly secret?: boolean;
}

export const REDIS_TYPES: readonly RedisType[] = ["node", "cluster", "sentinel", "semantic"];

export const REDIS_TYPE_DESCRIPTIONS: Readonly<Record<RedisType, ParseKeys>> = {
  node: "caching.settings.redisType.descriptions.node",
  cluster: "caching.settings.redisType.descriptions.cluster",
  sentinel: "caching.settings.redisType.descriptions.sentinel",
  semantic: "caching.settings.redisType.descriptions.semantic",
};

const isBlank = (value: unknown): boolean => value === undefined || value === null || String(value).trim() === "";

const portRule: CacheFieldRule = (value) => {
  if (isBlank(value)) {
    return null;
  }
  const port = Number(value);
  return Number.isInteger(port) && port >= 1 && port <= 65535 ? null : "caching.settings.validation.port";
};

const jsonListRule: CacheFieldRule = (value) => {
  if (isBlank(value)) {
    return null;
  }
  let parsed: unknown;
  try {
    parsed = JSON.parse(String(value));
  } catch {
    return "caching.settings.validation.jsonArrayInvalid";
  }
  return Array.isArray(parsed) ? null : "caching.settings.validation.jsonArrayRequired";
};

const nonNegativeIntegerRule: CacheFieldRule = (value) => {
  if (isBlank(value)) {
    return null;
  }
  const parsed = Number(value);
  return Number.isInteger(parsed) && parsed >= 0 ? null : "caching.settings.validation.nonNegativeInteger";
};

const numberRule: CacheFieldRule = (value) => {
  if (isBlank(value)) {
    return null;
  }
  return Number.isNaN(Number(value)) ? "caching.settings.validation.number" : null;
};

export const CACHE_FIELDS: readonly CacheField[] = [
  {
    name: "url",
    label: "caching.settings.fields.url.label",
    type: "string",
    section: "connection",
    helpText: "caching.settings.fields.url.helpText",
    redisType: null,
    secret: true,
  },
  {
    name: "host",
    label: "caching.settings.fields.host.label",
    type: "string",
    section: "connection",
    helpText: "caching.settings.fields.host.helpText",
    redisType: null,
  },
  {
    name: "port",
    label: "caching.settings.fields.port.label",
    type: "string",
    section: "connection",
    helpText: "caching.settings.fields.port.helpText",
    redisType: null,
    defaultValue: "6379",
    rules: [portRule],
  },
  {
    name: "db",
    label: "caching.settings.fields.db.label",
    type: "integer",
    section: "connection",
    helpText: "caching.settings.fields.db.helpText",
    redisType: null,
    rules: [nonNegativeIntegerRule],
  },
  {
    name: "password",
    label: "caching.settings.fields.password.label",
    type: "password",
    section: "connection",
    helpText: "caching.settings.fields.password.helpText",
    redisType: null,
    secret: true,
  },
  {
    name: "username",
    label: "caching.settings.fields.username.label",
    type: "string",
    section: "connection",
    helpText: "caching.settings.fields.username.helpText",
    redisType: null,
  },
  {
    name: "redis_startup_nodes",
    label: "caching.settings.fields.redisStartupNodes.label",
    type: "list",
    section: "cluster",
    helpText: "caching.settings.fields.redisStartupNodes.helpText",
    redisType: "cluster",
    rules: [jsonListRule],
  },
  {
    name: "sentinel_nodes",
    label: "caching.settings.fields.sentinelNodes.label",
    type: "list",
    section: "sentinel",
    helpText: "caching.settings.fields.sentinelNodes.helpText",
    redisType: "sentinel",
    rules: [jsonListRule],
  },
  {
    name: "service_name",
    label: "caching.settings.fields.serviceName.label",
    type: "string",
    section: "sentinel",
    helpText: "caching.settings.fields.serviceName.helpText",
    redisType: "sentinel",
  },
  {
    name: "sentinel_password",
    label: "caching.settings.fields.sentinelPassword.label",
    type: "password",
    section: "sentinel",
    helpText: "caching.settings.fields.sentinelPassword.helpText",
    redisType: "sentinel",
    secret: true,
  },
  {
    name: "similarity_threshold",
    label: "caching.settings.fields.similarityThreshold.label",
    type: "float",
    section: "semantic",
    helpText: "caching.settings.fields.similarityThreshold.helpText",
    redisType: "semantic",
    defaultValue: 0.8,
    rules: [numberRule],
  },
  {
    name: "redis_semantic_cache_embedding_model",
    label: "caching.settings.fields.embeddingModel.label",
    type: "model-select",
    section: "semantic",
    helpText: "caching.settings.fields.embeddingModel.helpText",
    redisType: "semantic",
  },
  {
    name: "semantic_cache_scope",
    label: "caching.settings.fields.semanticCacheScope.label",
    type: "select",
    section: "semantic",
    helpText: "caching.settings.fields.semanticCacheScope.helpText",
    redisType: "semantic",
    defaultValue: "key",
    options: [
      { value: "key", label: "caching.settings.scopeOptions.key" },
      { value: "end_user", label: "caching.settings.scopeOptions.endUser" },
    ],
  },
  {
    name: "ssl",
    label: "caching.settings.fields.ssl.label",
    type: "boolean",
    section: "ssl",
    helpText: "caching.settings.fields.ssl.helpText",
    redisType: null,
    defaultValue: false,
  },
  {
    name: "ssl_cert_reqs",
    label: "caching.settings.fields.sslCertReqs.label",
    type: "string",
    section: "ssl",
    helpText: "caching.settings.fields.sslCertReqs.helpText",
    redisType: null,
  },
  {
    name: "ssl_check_hostname",
    label: "caching.settings.fields.sslCheckHostname.label",
    type: "boolean",
    section: "ssl",
    helpText: "caching.settings.fields.sslCheckHostname.helpText",
    redisType: null,
    defaultValue: false,
  },
  {
    name: "namespace",
    label: "caching.settings.fields.namespace.label",
    type: "string",
    section: "cacheManagement",
    helpText: "caching.settings.fields.namespace.helpText",
    redisType: null,
  },
  {
    name: "ttl",
    label: "caching.settings.fields.ttl.label",
    type: "float",
    section: "cacheManagement",
    helpText: "caching.settings.fields.ttl.helpText",
    redisType: null,
    rules: [numberRule],
  },
  {
    name: "max_connections",
    label: "caching.settings.fields.maxConnections.label",
    type: "integer",
    section: "cacheManagement",
    helpText: "caching.settings.fields.maxConnections.helpText",
    redisType: null,
    rules: [nonNegativeIntegerRule],
  },
  {
    name: "gcp_service_account",
    label: "caching.settings.fields.gcpServiceAccount.label",
    type: "string",
    section: "gcp",
    helpText: "caching.settings.fields.gcpServiceAccount.helpText",
    redisType: null,
  },
  {
    name: "gcp_ssl_ca_certs",
    label: "caching.settings.fields.gcpSslCaCerts.label",
    type: "string",
    section: "gcp",
    helpText: "caching.settings.fields.gcpSslCaCerts.helpText",
    redisType: null,
  },
];
