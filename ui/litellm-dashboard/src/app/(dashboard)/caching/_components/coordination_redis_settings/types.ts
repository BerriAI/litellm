export const REDACTED_VALUE = "***REDACTED***";

export type CoordinationRedisSource = "coordination_redis" | "cache_backend" | "environment";

export type CoordinationRedisSettingValue = string | number | boolean | unknown[];

export type CoordinationRedisSettings = Record<string, CoordinationRedisSettingValue>;

export type CoordinationRedisSection = "connection" | "cluster" | "sentinel";

export interface CoordinationRedisSettingsField {
  field_name: string;
  field_type: string;
  field_value: unknown;
  field_description: string;
  ui_field_name: string;
  field_default?: unknown;
  section: CoordinationRedisSection;
}

export interface CoordinationRedisSettingsResponse {
  values: Record<string, unknown>;
  fields: CoordinationRedisSettingsField[];
  source: CoordinationRedisSource | null;
}

export interface CoordinationRedisTestResponse {
  status: "healthy" | "unhealthy";
  error?: string;
}
