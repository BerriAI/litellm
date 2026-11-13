import type { ParseKeys } from "i18next";
import moment from "moment";

export interface RelativeTimeOption {
  label: ParseKeys;
  shortLabel: string;
  settingValue: string;
  getValue: () => { from: Date; to: Date };
}

export const relativeTimeOptions: readonly RelativeTimeOption[] = [
  {
    label: "shared.dateRangePresets.today",
    shortLabel: "today",
    settingValue: "today",
    getValue: () => ({
      from: moment().startOf("day").toDate(),
      to: moment().endOf("day").toDate(),
    }),
  },
  {
    label: "shared.dateRangePresets.last7Days",
    shortLabel: "7d",
    settingValue: "last_7_days",
    getValue: () => ({
      from: moment().subtract(7, "days").startOf("day").toDate(),
      to: moment().endOf("day").toDate(),
    }),
  },
  {
    label: "shared.dateRangePresets.last30Days",
    shortLabel: "30d",
    settingValue: "last_30_days",
    getValue: () => ({
      from: moment().subtract(30, "days").startOf("day").toDate(),
      to: moment().endOf("day").toDate(),
    }),
  },
  {
    label: "shared.dateRangePresets.monthToDate",
    shortLabel: "MTD",
    settingValue: "month_to_date",
    getValue: () => ({
      from: moment().startOf("month").toDate(),
      to: moment().endOf("day").toDate(),
    }),
  },
  {
    label: "shared.dateRangePresets.yearToDate",
    shortLabel: "YTD",
    settingValue: "year_to_date",
    getValue: () => ({
      from: moment().startOf("year").toDate(),
      to: moment().endOf("day").toDate(),
    }),
  },
];

const lastCalendarMonths = (months: number, labelKey: ParseKeys): RelativeTimeOption => ({
  label: labelKey,
  shortLabel: `${months}M`,
  settingValue: `last_${months}_months`,
  getValue: () => ({
    from: moment()
      .startOf("month")
      .subtract(months - 1, "months")
      .toDate(),
    to: moment().endOf("day").toDate(),
  }),
});

export const calendarMonthOptions: readonly RelativeTimeOption[] = [
  lastCalendarMonths(3, "shared.dateRangePresets.last3Months"),
  lastCalendarMonths(6, "shared.dateRangePresets.last6Months"),
  lastCalendarMonths(12, "shared.dateRangePresets.last12Months"),
];
