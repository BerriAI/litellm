import { endOfDay, format, isValid, parse } from "date-fns";
import { createParser } from "nuqs";
import { useCallback, useMemo } from "react";

import type { DateRangePickerValue } from "@/components/shared/date_picker_types";

const DAY_FORMAT = "yyyy-MM-dd";
const DAY_PATTERN = /^\d{4}-\d{2}-\d{2}$/;

const parseLocalDay = (day: string): Date => parse(day, DAY_FORMAT, new Date());

const formatLocalDay = (date: Date): string => format(date, DAY_FORMAT);

export const parseAsLocalDay = createParser({
  parse: (value: string) => (DAY_PATTERN.test(value) && isValid(parseLocalDay(value)) ? value : null),
  serialize: (value: string) => value,
});

export interface UrlDays {
  start: string | null;
  end: string | null;
}

export interface UrlDayRange {
  dateValue: DateRangePickerValue;
  onDateChange: (value: DateRangePickerValue) => void;
}

const rangeOfDays = (start: string | null, end: string | null): DateRangePickerValue | null =>
  start !== null && end !== null && start <= end
    ? { from: parseLocalDay(start), to: endOfDay(parseLocalDay(end)) }
    : null;

const daysOfRange = ({ from, to }: DateRangePickerValue): UrlDays =>
  from && to ? { start: formatLocalDay(from), end: formatLocalDay(to) } : { start: null, end: null };

export const useUrlDayRange = (
  { start, end }: UrlDays,
  setDays: (days: UrlDays) => unknown,
  fallback: DateRangePickerValue,
): UrlDayRange => {
  const dateValue = useMemo(() => rangeOfDays(start, end) ?? fallback, [start, end, fallback]);
  const onDateChange = useCallback((value: DateRangePickerValue) => void setDays(daysOfRange(value)), [setDays]);
  return { dateValue, onDateChange };
};
