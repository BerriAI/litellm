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
