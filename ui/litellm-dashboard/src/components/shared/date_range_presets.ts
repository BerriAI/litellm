import moment from "moment";

export interface RelativeTimeOption {
  label: string;
  shortLabel: string;
  settingValue: string;
  getValue: () => { from: Date; to: Date };
}

/** Widest window the sub-day (ClickHouse rollup) usage path serves; longer ranges use the daily tables. */
export const SUB_DAY_MAX_RANGE_MS = 24 * 60 * 60 * 1000;

/**
 * A window of at most 24 hours that does not start at a local midnight is a sub-day investigation
 * window (Last 1h, 14:05 to 14:35): it is served from spend-log rollups and never snapped to day edges.
 */
export const isSubDayRange = (from: Date, to: Date): boolean =>
  !moment(from).isSame(moment(from).startOf("day")) && to.getTime() - from.getTime() <= SUB_DAY_MAX_RANGE_MS;

const lastMinutes = (minutes: number) => () => ({
  from: moment().subtract(minutes, "minutes").toDate(),
  to: moment().toDate(),
});

export const relativeTimeOptions: readonly RelativeTimeOption[] = [
  { label: "Last 15 minutes", shortLabel: "15m", settingValue: "last_15_minutes", getValue: lastMinutes(15) },
  { label: "Last hour", shortLabel: "1h", settingValue: "last_1_hour", getValue: lastMinutes(60) },
  { label: "Last 6 hours", shortLabel: "6h", settingValue: "last_6_hours", getValue: lastMinutes(6 * 60) },
  { label: "Last 24 hours", shortLabel: "24h", settingValue: "last_24_hours", getValue: lastMinutes(24 * 60) },
  {
    label: "Today",
    shortLabel: "today",
    settingValue: "today",
    getValue: () => ({
      from: moment().startOf("day").toDate(),
      to: moment().endOf("day").toDate(),
    }),
  },
  {
    label: "Last 7 days",
    shortLabel: "7d",
    settingValue: "last_7_days",
    getValue: () => ({
      from: moment().subtract(7, "days").startOf("day").toDate(),
      to: moment().endOf("day").toDate(),
    }),
  },
  {
    label: "Last 30 days",
    shortLabel: "30d",
    settingValue: "last_30_days",
    getValue: () => ({
      from: moment().subtract(30, "days").startOf("day").toDate(),
      to: moment().endOf("day").toDate(),
    }),
  },
  {
    label: "Month to date",
    shortLabel: "MTD",
    settingValue: "month_to_date",
    getValue: () => ({
      from: moment().startOf("month").toDate(),
      to: moment().endOf("day").toDate(),
    }),
  },
  {
    label: "Year to date",
    shortLabel: "YTD",
    settingValue: "year_to_date",
    getValue: () => ({
      from: moment().startOf("year").toDate(),
      to: moment().endOf("day").toDate(),
    }),
  },
];
