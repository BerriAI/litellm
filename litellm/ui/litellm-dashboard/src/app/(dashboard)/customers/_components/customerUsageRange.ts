import { createParser, parseAsStringLiteral, useQueryStates } from "nuqs";
import { useMemo } from "react";
import type { DateRangePickerValue } from "@/components/shared/date_picker_types";
import type { RelativeTimeOption } from "@/components/shared/date_range_presets";

const PRESETS = ["month", "previous", "YTD"] as const;
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
  if (preset === "YTD") {
    return { from: new Date(year, 0, 1), to: new Date(year, month, now.getUTCDate(), 23, 59, 59) };
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
  {
    label: "shared.dateRangePresets.yearToDate",
    shortLabel: "YTD",
    settingValue: "year_to_date",
    getValue: () => customerPresetRange("YTD"),
  },
];

const presetParser = parseAsStringLiteral(PRESETS);
const parsers = { range: customerRangeParser, period: presetParser };

export const useCustomerUsageRange = () => {
  const [params, setParams] = useQueryStates(parsers);
  const initialRange = useMemo(() => customerPresetRange("month"), []);
  const dateValue = params.range ?? initialRange;
  const onDateChange = (value: DateRangePickerValue, preset: string | null) => {
    if (!value.from || !value.to || value.from > value.to) return;
    void setParams(
      { range: { from: value.from, to: value.to }, period: preset ? presetParser.parse(preset) : null },
      { history: "push" },
    );
  };
  return { dateValue, onDateChange, preset: params.range ? params.period : "month" };
};
