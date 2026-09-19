import moment from "moment";
import { parseAsString, useQueryStates } from "nuqs";
import { useCallback, useEffect, useMemo } from "react";
import type { DateRangePickerValue } from "@/components/shared/date_picker_types";

const DAY_FORMAT = "YYYY-MM-DD";
const DEFAULT_RANGE_DAYS = 7;

const rangeParsers = { start_date: parseAsString, end_date: parseAsString };

interface DayRange {
  startDate: string;
  endDate: string;
}

const formatDay = (date: Date) => moment(date).format(DAY_FORMAT);

const isDay = (value: string | null): value is string => value !== null && moment(value, DAY_FORMAT, true).isValid();

const defaultRange = (today: Date): DayRange => ({
  startDate: formatDay(moment(today).subtract(DEFAULT_RANGE_DAYS, "days").toDate()),
  endDate: formatDay(today),
});

const toRange = (startDate: string | null, endDate: string | null): DayRange | null =>
  isDay(startDate) && isDay(endDate) && startDate <= endDate ? { startDate, endDate } : null;

export interface MonitorDateRange {
  startDate: string;
  endDate: string;
  pickerValue: DateRangePickerValue;
  setPickerValue: (value: DateRangePickerValue) => void;
}

export function useMonitorDateRange(): MonitorDateRange {
  const defaults = useMemo(() => defaultRange(new Date()), []);
  const [{ start_date, end_date }, setRange] = useQueryStates(rangeParsers);
  const urlRange = toRange(start_date, end_date);
  const { startDate, endDate } = urlRange ?? defaults;
  const hasUnusableRange = urlRange === null && (start_date !== null || end_date !== null);

  useEffect(() => {
    if (hasUnusableRange) void setRange(null);
  }, [hasUnusableRange, setRange]);

  const pickerValue = useMemo<DateRangePickerValue>(
    () => ({
      from: moment(startDate, DAY_FORMAT).startOf("day").toDate(),
      to: moment(endDate, DAY_FORMAT).endOf("day").toDate(),
    }),
    [startDate, endDate],
  );

  const setPickerValue = useCallback(
    ({ from, to }: DateRangePickerValue) => {
      const next = toRange(from ? formatDay(from) : defaults.startDate, to ? formatDay(to) : defaults.endDate);
      const isDefault = next?.startDate === defaults.startDate && next.endDate === defaults.endDate;
      void setRange(next === null || isDefault ? null : { start_date: next.startDate, end_date: next.endDate });
    },
    [defaults, setRange],
  );

  return { startDate, endDate, pickerValue, setPickerValue };
}
