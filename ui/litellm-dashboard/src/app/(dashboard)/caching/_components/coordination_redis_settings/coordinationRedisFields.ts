import type { ParseKeys } from "i18next";

import i18n from "@/i18n/i18n";

export type CoordinationFieldType = "string" | "password" | "integer" | "boolean" | "list";

export type CoordinationRedisType = "node" | "cluster" | "sentinel";

export type CoordinationSection = "connection" | "cluster" | "sentinel" | "ssl";

export type CoordinationFieldRule = (value: unknown) => ParseKeys | null;

export interface CoordinationField {
  readonly name: string;
  readonly label: ParseKeys;
  readonly type: CoordinationFieldType;
  readonly section: CoordinationSection;
  readonly helpText: ParseKeys;
  readonly redisType: CoordinationRedisType | null;
  readonly secret: boolean;
  readonly defaultValue?: string | number | boolean;
  readonly rules?: CoordinationFieldRule[];
}

export const COORDINATION_REDIS_TYPES: readonly CoordinationRedisType[] = ["node", "cluster", "sentinel"];

export const COORDINATION_REDIS_TYPE_DESCRIPTIONS: Readonly<Record<CoordinationRedisType, string>> = {
  get node() {
    return i18n.t("caching.coordinationRedis.redisType.descriptions.node");
  },
  get cluster() {
    return i18n.t("caching.coordinationRedis.redisType.descriptions.cluster");
  },
  get sentinel() {
    return i18n.t("caching.coordinationRedis.redisType.descriptions.sentinel");
  },
};

export const COORDINATION_REDIS_TYPE_LABELS: Readonly<Record<CoordinationRedisType, ParseKeys>> = {
  node: "caching.coordinationRedis.redisType.options.node",
  cluster: "caching.coordinationRedis.redisType.options.cluster",
  sentinel: "caching.coordinationRedis.redisType.options.sentinel",
};

const isBlank = (value: unknown): boolean => value === undefined || value === null || String(value).trim() === "";

const portRule: CoordinationFieldRule = (value) => {
  if (isBlank(value)) {
    return null;
  }
  const port = Number(value);
  return Number.isInteger(port) && port >= 1 && port <= 65535 ? null : "caching.coordinationRedis.validation.port";
};

const jsonListRule: CoordinationFieldRule = (value) => {
  if (isBlank(value)) {
    return null;
  }
  let parsed: unknown;
  try {
    parsed = JSON.parse(String(value));
  } catch {
    return "caching.coordinationRedis.validation.jsonArrayInvalid";
  }
  return Array.isArray(parsed) ? null : "caching.coordinationRedis.validation.jsonArrayRequired";
};

export const COORDINATION_FIELDS: readonly CoordinationField[] = [
  {
    name: "url",
    label: "caching.coordinationRedis.fields.url.label",
    type: "password",
    section: "connection",
    helpText: "caching.coordinationRedis.fields.url.helpText",
    redisType: null,
    secret: true,
  },
  {
    name: "host",
    label: "caching.coordinationRedis.fields.host.label",
    type: "string",
    section: "connection",
    helpText: "caching.coordinationRedis.fields.host.helpText",
    redisType: null,
    secret: false,
  },
  {
    name: "port",
    label: "caching.coordinationRedis.fields.port.label",
    type: "integer",
    section: "connection",
    helpText: "caching.coordinationRedis.fields.port.helpText",
    redisType: null,
    secret: false,
    defaultValue: "6379",
    rules: [portRule],
  },
  {
    name: "username",
    label: "caching.coordinationRedis.fields.username.label",
    type: "string",
    section: "connection",
    helpText: "caching.coordinationRedis.fields.username.helpText",
    redisType: null,
    secret: false,
  },
  {
    name: "password",
    label: "caching.coordinationRedis.fields.password.label",
    type: "password",
    section: "connection",
    helpText: "caching.coordinationRedis.fields.password.helpText",
    redisType: null,
    secret: true,
  },
  {
    name: "startup_nodes",
    label: "caching.coordinationRedis.fields.startupNodes.label",
    type: "list",
    section: "cluster",
    helpText: "caching.coordinationRedis.fields.startupNodes.helpText",
    redisType: "cluster",
    secret: false,
    rules: [jsonListRule],
  },
  {
    name: "sentinel_nodes",
    label: "caching.coordinationRedis.fields.sentinelNodes.label",
    type: "list",
    section: "sentinel",
    helpText: "caching.coordinationRedis.fields.sentinelNodes.helpText",
    redisType: "sentinel",
    secret: false,
    rules: [jsonListRule],
  },
  {
    name: "service_name",
    label: "caching.coordinationRedis.fields.serviceName.label",
    type: "string",
    section: "sentinel",
    helpText: "caching.coordinationRedis.fields.serviceName.helpText",
    redisType: "sentinel",
    secret: false,
  },
  {
    name: "sentinel_password",
    label: "caching.coordinationRedis.fields.sentinelPassword.label",
    type: "password",
    section: "sentinel",
    helpText: "caching.coordinationRedis.fields.sentinelPassword.helpText",
    redisType: "sentinel",
    secret: true,
  },
  {
    name: "ssl",
    label: "caching.coordinationRedis.fields.ssl.label",
    type: "boolean",
    section: "ssl",
    helpText: "caching.coordinationRedis.fields.ssl.helpText",
    redisType: null,
    secret: false,
    defaultValue: false,
  },
];
