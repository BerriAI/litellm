import moment from "moment";

// Structurally compatible with @tremor/react's DateRangePickerValue, defined
// locally so this module doesn't need a (now-discouraged) tremor import.
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

/** Returns the shortLabel of the relative preset that matches `value`, or null if it's a custom range. */
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

/** Recomputes a fresh {from, to} for a stored preset shortLabel (e.g. "today" always resolves to the current day). */
export function getRelativeRangeByShortLabel(shortLabel: string): { from: Date; to: Date } | null {
  const option = relativeTimeOptions.find((o) => o.shortLabel === shortLabel);
  return option ? option.getValue() : null;
}
