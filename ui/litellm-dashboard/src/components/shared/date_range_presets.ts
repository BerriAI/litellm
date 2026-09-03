import moment from "moment";

export const DATE_RANGE_PRESET_IDS = ["today", "last_7_days", "last_30_days", "month_to_date", "year_to_date"] as const;

export type DateRangePresetId = (typeof DATE_RANGE_PRESET_IDS)[number];

export interface DateRangePreset {
  id: DateRangePresetId;
  label: string;
  shortLabel: string;
  getValue: (now?: Date) => { from: Date; to: Date };
}

export const isDateRangePresetId = (value: unknown): value is DateRangePresetId =>
  typeof value === "string" && (DATE_RANGE_PRESET_IDS as readonly string[]).includes(value);

const PRESETS_BY_ID: Record<DateRangePresetId, DateRangePreset> = {
  today: {
    id: "today",
    label: "Today",
    shortLabel: "today",
    getValue: (now = new Date()) => ({
      from: moment(now).startOf("day").toDate(),
      to: moment(now).endOf("day").toDate(),
    }),
  },
  last_7_days: {
    id: "last_7_days",
    label: "Last 7 days",
    shortLabel: "7d",
    getValue: (now = new Date()) => ({
      from: moment(now).subtract(7, "days").startOf("day").toDate(),
      to: moment(now).endOf("day").toDate(),
    }),
  },
  last_30_days: {
    id: "last_30_days",
    label: "Last 30 days",
    shortLabel: "30d",
    getValue: (now = new Date()) => ({
      from: moment(now).subtract(30, "days").startOf("day").toDate(),
      to: moment(now).endOf("day").toDate(),
    }),
  },
  month_to_date: {
    id: "month_to_date",
    label: "Month to date",
    shortLabel: "MTD",
    getValue: (now = new Date()) => ({
      from: moment(now).startOf("month").toDate(),
      to: moment(now).endOf("day").toDate(),
    }),
  },
  year_to_date: {
    id: "year_to_date",
    label: "Year to date",
    shortLabel: "YTD",
    getValue: (now = new Date()) => ({
      from: moment(now).startOf("year").toDate(),
      to: moment(now).endOf("day").toDate(),
    }),
  },
};

export const DATE_RANGE_PRESETS: readonly DateRangePreset[] = DATE_RANGE_PRESET_IDS.map((id) => PRESETS_BY_ID[id]);

export const resolveDateRangePreset = (id: DateRangePresetId, now: Date = new Date()): { from: Date; to: Date } =>
  PRESETS_BY_ID[id].getValue(now);
