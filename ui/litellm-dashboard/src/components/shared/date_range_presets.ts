import moment from "moment";

interface DateRangeLike {
  from?: Date;
  to?: Date;
}

export interface RelativeTimeOption {
  label: string;
  shortLabel: string;
  getValue: () => { from: Date; to: Date };
}

export const relativeTimeOptions: RelativeTimeOption[] = [
  {
    label: "Today",
    shortLabel: "today",
    getValue: () => ({
      from: moment().startOf("day").toDate(),
      to: moment().endOf("day").toDate(),
    }),
  },
  {
    label: "Last 7 days",
    shortLabel: "7d",
    getValue: () => ({
      from: moment().subtract(7, "days").startOf("day").toDate(),
      to: moment().endOf("day").toDate(),
    }),
  },
  {
    label: "Last 30 days",
    shortLabel: "30d",
    getValue: () => ({
      from: moment().subtract(30, "days").startOf("day").toDate(),
      to: moment().endOf("day").toDate(),
    }),
  },
  {
    label: "Month to date",
    shortLabel: "MTD",
    getValue: () => ({
      from: moment().startOf("month").toDate(),
      to: moment().endOf("day").toDate(),
    }),
  },
  {
    label: "Year to date",
    shortLabel: "YTD",
    getValue: () => ({
      from: moment().startOf("year").toDate(),
      to: moment().endOf("day").toDate(),
    }),
  },
];

export function getMatchingRelativeOption(value: DateRangeLike): string | null {
  if (!value.from || !value.to) return null;

  for (const option of relativeTimeOptions) {
    const optionRange = option.getValue();
    const fromMatches = moment(value.from).isSame(moment(optionRange.from), "day");
    const toMatches = moment(value.to).isSame(moment(optionRange.to), "day");
    if (fromMatches && toMatches) {
      return option.shortLabel;
    }
  }

  return null;
}

export function getRelativeRangeByShortLabel(shortLabel: string): { from: Date; to: Date } | null {
  const option = relativeTimeOptions.find((o) => o.shortLabel === shortLabel);
  return option ? option.getValue() : null;
}
