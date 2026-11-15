import { createParser, parseAsStringLiteral, useQueryStates, type inferParserType } from "nuqs";
import { useMemo } from "react";
import type { DateRangePickerValue } from "@/components/shared/date_picker_types";
import { relativeTimeOptions, type RelativeTimeOption } from "@/components/shared/date_range_presets";

const PRESETS = ["month", "previous", "YTD", "MTD", "today", "7d", "30d"] as const;
type Preset = (typeof PRESETS)[number];
type Range = { from: Date; to: Date };

export const CUSTOMER_USAGE_MAX_RANGE_DAYS = 400;

export const calendarDate = (date: Date): string =>
  `${date.getFullYear().toString().padStart(4, "0")}-${(date.getMonth() + 1).toString().padStart(2, "0")}-${date.getDate().toString().padStart(2, "0")}`;

const parseDate = (value: string): Date | null => {
  if (!/^\d{4}-\d{2}-\d{2}$/.test(value)) return null;
  const date = new Date(`${value}T12:00:00`);
  return Number.isFinite(date.getTime()) && calendarDate(date) === value ? date : null;
};

export const customerRangeParser = createParser<Range>({
  parse: (value) => {
    const [start, end, extra] = value.split("..");
    if (!start || !end || extra !== undefined) return null;
    const from = parseDate(start);
    const to = parseDate(end);
    if (!from || !to || from > to) return null;
    const spanDays = (Date.parse(end) - Date.parse(start)) / 86_400_000;
    return spanDays <= CUSTOMER_USAGE_MAX_RANGE_DAYS ? { from, to } : null;
  },
  serialize: ({ from, to }) => `${calendarDate(from)}..${calendarDate(to)}`,
  eq: (a, b) => calendarDate(a.from) === calendarDate(b.from) && calendarDate(a.to) === calendarDate(b.to),
});

export const customerPresetRange = (preset: Preset, now = new Date()): Range => {
  const year = now.getUTCFullYear();
  const month = now.getUTCMonth();
  const day = now.getUTCDate();
  const to = new Date(year, month, day, 23, 59, 59);
  if (preset === "YTD") return { from: new Date(year, 0, 1), to };
  if (preset === "MTD") return { from: new Date(year, month, 1), to };
  if (preset === "today" || preset === "7d" || preset === "30d") {
    const days = { today: 0, "7d": 7, "30d": 30 }[preset];
    return { from: new Date(year, month, day - days), to };
  }
  const targetMonth = preset === "previous" ? month - 1 : month;
  return { from: new Date(year, targetMonth, 1), to: new Date(year, targetMonth + 1, 0, 23, 59, 59) };
};

export const customerUsagePresets: readonly RelativeTimeOption[] = [
  {
    label: "customerPortal.range.thisMonth",
    shortLabel: "month",
    settingValue: "this_month",
    getValue: () => customerPresetRange("month"),
  },
  {
    label: "customerPortal.range.previousMonth",
    shortLabel: "previous",
    settingValue: "previous_month",
    getValue: () => customerPresetRange("previous"),
  },
  ...relativeTimeOptions.map((option) => ({
    ...option,
    getValue: () => customerPresetRange(presetParser.parse(option.shortLabel) ?? "month"),
  })),
];

const presetParser = parseAsStringLiteral(PRESETS);
const rangeParser = createParser<Range | Preset>({
  parse: (value) => presetParser.parse(value) ?? customerRangeParser.parse(value),
  serialize: (value) => (typeof value === "string" ? value : customerRangeParser.serialize(value)),
  eq: (a, b) => (typeof a === "string" || typeof b === "string" ? a === b : customerRangeParser.eq(a, b)),
});
const dateParser = createParser({
  parse: (value) => (parseDate(value) ? value : null),
  serialize: (value: string) => value,
});
export const customerUsageDateParsers = { range: rangeParser, period: presetParser, from: dateParser, to: dateParser };
type DateParams = inferParserType<typeof customerUsageDateParsers>;

export const customerDateSelection = (params: DateParams, now = new Date()) => {
  if (typeof params.range === "string") {
    return { dateValue: customerPresetRange(params.range, now), preset: params.range };
  }
  if (params.range) return { dateValue: params.range, preset: params.period };
  const custom = params.from && params.to ? customerRangeParser.parse(`${params.from}..${params.to}`) : null;
  if (custom) return { dateValue: custom, preset: null };
  return { dateValue: customerPresetRange("month", now), preset: "month" };
};

export const customerDatePatch = (value: DateRangePickerValue, preset: string | null) => {
  if (!value.from || !value.to) return null;
  const from = calendarDate(value.from);
  const to = calendarDate(value.to);
  if (!customerRangeParser.parse(`${from}..${to}`)) return null;
  const range = preset ? presetParser.parse(preset) : null;
  return { range, period: null, from: range ? null : from, to: range ? null : to };
};

export const useCustomerUsageRange = () => {
  const [params, setParams] = useQueryStates(customerUsageDateParsers);
  const selection = useMemo(() => customerDateSelection(params), [params]);
  const onDateChange = (value: DateRangePickerValue, preset: string | null) => {
    const patch = customerDatePatch(value, preset);
    if (patch) void setParams(patch, { history: "push" });
  };
  return { ...selection, onDateChange };
};
