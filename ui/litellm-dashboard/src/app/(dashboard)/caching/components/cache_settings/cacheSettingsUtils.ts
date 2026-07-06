import { CACHE_FIELDS, CacheField, CacheSection, RedisType } from "./cacheSettingsFields";

export type CacheFormValue = string | number | boolean | undefined;
export type CacheFormValues = Record<string, CacheFormValue>;
export type CacheSavePayloadValue = string | number | boolean | unknown[];
export type CacheSavePayload = Record<string, CacheSavePayloadValue>;

export const isFieldVisible = (field: CacheField, redisType: RedisType): boolean =>
  field.redisType === null || field.redisType === redisType;

export const fieldsForSection = (section: CacheSection, redisType: RedisType): CacheField[] =>
  CACHE_FIELDS.filter((field) => field.section === section && isFieldVisible(field, redisType));

const initialValueForField = (field: CacheField, raw: unknown): CacheFormValue => {
  const source = raw ?? field.defaultValue;

  if (field.type === "boolean") {
    return source === true || source === "true";
  }

  if (field.type === "list") {
    if (source === undefined || source === null || source === "") {
      return "";
    }
    return typeof source === "string" ? source : JSON.stringify(source, null, 2);
  }

  if (source === undefined || source === null) {
    return "";
  }
  return String(source);
};

export const buildInitialValues = (currentValues: Record<string, unknown>): CacheFormValues =>
  Object.fromEntries(CACHE_FIELDS.map((field) => [field.name, initialValueForField(field, currentValues[field.name])]));

const saveValueForField = (field: CacheField, raw: CacheFormValue): CacheSavePayloadValue | undefined => {
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

  if (field.type === "integer" || field.type === "float") {
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

export const buildCachePayload = (
  redisType: RedisType,
  values: CacheFormValues,
  { forTesting }: { forTesting: boolean },
): CacheSavePayload => {
  const type = !forTesting && redisType === "semantic" ? "redis-semantic" : "redis";

  const entries = CACHE_FIELDS.filter((field) => isFieldVisible(field, redisType)).flatMap((field) => {
    const value = saveValueForField(field, values[field.name]);
    return value === undefined ? [] : [[field.name, value] as const];
  });

  return { type, ...Object.fromEntries(entries) };
};
