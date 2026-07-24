import type { FormItemProps } from "antd";

export type CoordinationFieldType = "string" | "password" | "integer" | "boolean" | "list";

export type CoordinationRedisType = "node" | "cluster" | "sentinel";

export type CoordinationSection = "connection" | "cluster" | "sentinel" | "ssl";

export type CoordinationFieldRule = NonNullable<FormItemProps["rules"]>[number];

export interface CoordinationField {
  readonly name: string;
  readonly label: string;
  readonly type: CoordinationFieldType;
  readonly section: CoordinationSection;
  readonly helpText: string;
  readonly redisType: CoordinationRedisType | null;
  readonly secret: boolean;
  readonly defaultValue?: string | number | boolean;
  readonly rules?: CoordinationFieldRule[];
}

export const COORDINATION_REDIS_TYPES: readonly CoordinationRedisType[] = ["node", "cluster", "sentinel"];

export const COORDINATION_REDIS_TYPE_DESCRIPTIONS: Readonly<Record<CoordinationRedisType, string>> = {
  node: "Standard Redis node/single instance",
  cluster: "Redis Cluster mode for high availability and horizontal scaling",
  sentinel: "Redis Sentinel mode for high availability with automatic failover",
};

export const COORDINATION_REDIS_TYPE_LABELS: Readonly<Record<CoordinationRedisType, string>> = {
  node: "Node (Single Instance)",
  cluster: "Cluster",
  sentinel: "Sentinel",
};

const portRule: CoordinationFieldRule = {
  validator: (_rule, value) => {
    if (value === undefined || value === null || String(value).trim() === "") {
      return Promise.resolve();
    }
    const port = Number(value);
    if (!Number.isInteger(port) || port < 1 || port > 65535) {
      return Promise.reject(new Error("Port must be an integer between 1 and 65535"));
    }
    return Promise.resolve();
  },
};

const jsonListRule: CoordinationFieldRule = {
  validator: (_rule, value) => {
    if (value === undefined || value === null || String(value).trim() === "") {
      return Promise.resolve();
    }
    let parsed: unknown;
    try {
      parsed = JSON.parse(String(value));
    } catch {
      return Promise.reject(new Error("Must be a valid JSON array (use double quotes)"));
    }
    if (!Array.isArray(parsed)) {
      return Promise.reject(new Error("Must be a JSON array"));
    }
    return Promise.resolve();
  },
};

export const COORDINATION_FIELDS: readonly CoordinationField[] = [
  {
    name: "url",
    label: "Redis URL",
    type: "password",
    section: "connection",
    helpText:
      "Full Redis/Valkey connection URL (e.g. redis://:password@host:6379/1). When set, it takes precedence over Host, Port, Username, and Password.",
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
    secret: false,
  },
  {
    name: "port",
    label: "Port",
    type: "integer",
    section: "connection",
    helpText: "Redis server port number",
    redisType: null,
    secret: false,
    defaultValue: "6379",
    rules: [portRule],
  },
  {
    name: "username",
    label: "Username",
    type: "string",
    section: "connection",
    helpText: "Redis server username (if required)",
    redisType: null,
    secret: false,
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
    name: "startup_nodes",
    label: "Startup Nodes",
    type: "list",
    section: "cluster",
    helpText: 'List of startup nodes for Redis Cluster (e.g., [{"host": "127.0.0.1", "port": 7001}])',
    redisType: "cluster",
    secret: false,
    rules: [jsonListRule],
  },
  {
    name: "sentinel_nodes",
    label: "Sentinel Nodes",
    type: "list",
    section: "sentinel",
    helpText: 'List of Sentinel nodes (e.g., [["localhost", 26379]])',
    redisType: "sentinel",
    secret: false,
    rules: [jsonListRule],
  },
  {
    name: "service_name",
    label: "Service Name",
    type: "string",
    section: "sentinel",
    helpText: "Master service name for Redis Sentinel",
    redisType: "sentinel",
    secret: false,
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
    name: "ssl",
    label: "SSL",
    type: "boolean",
    section: "ssl",
    helpText: "Enable SSL/TLS connection",
    redisType: null,
    secret: false,
    defaultValue: false,
  },
];
