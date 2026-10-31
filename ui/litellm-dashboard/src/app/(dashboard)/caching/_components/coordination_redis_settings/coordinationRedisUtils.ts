import type { ParseKeys, TFunction } from "i18next";

import type { StatusTone } from "@/components/shared/table_cells/status_badge";
import i18n from "@/i18n/i18n";
import {
  COORDINATION_FIELDS,
  CoordinationField,
  CoordinationRedisType,
  CoordinationSection,
} from "./coordinationRedisFields";
import { CoordinationRedisSettings, CoordinationRedisSource, REDACTED_VALUE } from "./types";

export type CoordinationFormValue = string | number | boolean | undefined;
export type CoordinationFormValues = Record<string, CoordinationFormValue>;

export const isFieldVisible = (field: CoordinationField, redisType: CoordinationRedisType): boolean =>
  field.redisType === null || field.redisType === redisType;

export const fieldsForSection = (section: CoordinationSection, redisType: CoordinationRedisType): CoordinationField[] =>
  COORDINATION_FIELDS.filter((field) => field.section === section && isFieldVisible(field, redisType));

const hasValue = (raw: unknown): boolean => {
  const isEmptyArray = Array.isArray(raw) && raw.length === 0;
  const isBlank = raw === undefined || raw === null || raw === "";
  return !isBlank && !isEmptyArray;
};

export const inferRedisType = (values: Record<string, unknown>): CoordinationRedisType => {
  if (hasValue(values.sentinel_nodes)) {
    return "sentinel";
  }
  if (hasValue(values.startup_nodes)) {
    return "cluster";
  }
  return "node";
};

export const configuredSecretFields = (values: Record<string, unknown>): ReadonlySet<string> =>
  new Set(COORDINATION_FIELDS.filter((field) => field.secret && hasValue(values[field.name])).map((f) => f.name));

const initialValueForField = (field: CoordinationField, raw: unknown): CoordinationFormValue => {
  if (field.secret) {
    return "";
  }

  const source = raw ?? field.defaultValue;

  if (field.type === "boolean") {
    return source === true || source === "true";
  }

  if (field.type === "list") {
    if (!hasValue(source)) {
      return "";
    }
    return typeof source === "string" ? source : JSON.stringify(source, null, 2);
  }

  if (source === undefined || source === null) {
    return "";
  }
  return String(source);
};

export const buildInitialValues = (values: Record<string, unknown>): CoordinationFormValues =>
  Object.fromEntries(COORDINATION_FIELDS.map((field) => [field.name, initialValueForField(field, values[field.name])]));

const saveValueForField = (
  field: CoordinationField,
  raw: CoordinationFormValue,
): CoordinationRedisSettings[string] | undefined => {
  if (field.secret && raw === REDACTED_VALUE) {
    return undefined;
  }

  if (field.type === "boolean") {
    return Boolean(raw);
  }

  if (field.type === "list") {
    if (typeof raw !== "string" || raw.trim() === "") {
      return undefined;
    }
    try {
      return JSON.parse(raw) as unknown[];
    } catch {
      return undefined;
    }
  }

  if (field.type === "integer") {
    if (raw === undefined || raw === null || raw === "") {
      return undefined;
    }
    const parsed = Number(raw);
    return Number.isNaN(parsed) ? undefined : parsed;
  }

  if (typeof raw !== "string") {
    return raw === undefined ? undefined : String(raw);
  }
  const trimmed = raw.trim();
  return trimmed === "" ? undefined : trimmed;
};

export const buildCoordinationPayload = (
  redisType: CoordinationRedisType,
  values: CoordinationFormValues,
): CoordinationRedisSettings => {
  const entries = COORDINATION_FIELDS.filter((field) => isFieldVisible(field, redisType)).flatMap((field) => {
    const value = saveValueForField(field, values[field.name]);
    return value === undefined ? [] : [[field.name, value] as const];
  });

  return Object.fromEntries(entries);
};

export interface SourceBadgeDescriptor {
  readonly tone: StatusTone;
  readonly label: string;
  readonly tooltip: string;
}

interface SourceBadgeKeys {
  readonly tone: StatusTone;
  readonly label: ParseKeys;
  readonly tooltip: ParseKeys;
}

const SOURCE_BADGES: Readonly<Record<CoordinationRedisSource, SourceBadgeKeys>> = {
  coordination_redis: {
    tone: "success",
    label: "caching.coordinationRedis.source.coordinationRedis.label",
    tooltip: "caching.coordinationRedis.source.coordinationRedis.tooltip",
  },
  cache_backend: {
    tone: "info",
    label: "caching.coordinationRedis.source.cacheBackend.label",
    tooltip: "caching.coordinationRedis.source.cacheBackend.tooltip",
  },
  environment: {
    tone: "info",
    label: "caching.coordinationRedis.source.environment.label",
    tooltip: "caching.coordinationRedis.source.environment.tooltip",
  },
};

const NOT_CONFIGURED_BADGE: SourceBadgeKeys = {
  tone: "neutral",
  label: "caching.coordinationRedis.source.notConfigured.label",
  tooltip: "caching.coordinationRedis.source.notConfigured.tooltip",
};

export const sourceBadge = (
  source: string | null | undefined,
  t: TFunction = i18n.t.bind(i18n),
): SourceBadgeDescriptor => {
  const known: Readonly<Record<string, SourceBadgeKeys>> = SOURCE_BADGES;
  const badge = (source && known[source]) || NOT_CONFIGURED_BADGE;
  return { tone: badge.tone, label: t(badge.label), tooltip: t(badge.tooltip) };
};
