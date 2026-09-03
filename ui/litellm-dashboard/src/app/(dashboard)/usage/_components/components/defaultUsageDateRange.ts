import { isDateRangePresetId, resolveDateRangePreset } from "@/components/shared/date_range_presets";

export const DEFAULT_USAGE_DATE_RANGE_SETTING_KEY = "default_usage_date_range";

const FALLBACK_WINDOW_MS = 7 * 24 * 60 * 60 * 1000;

export const resolveDefaultUsageDateRange = (setting: unknown, now: Date = new Date()): { from: Date; to: Date } =>
  isDateRangePresetId(setting)
    ? resolveDateRangePreset(setting, now)
    : { from: new Date(now.getTime() - FALLBACK_WINDOW_MS), to: now };
