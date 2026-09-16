import moment from "moment";
import { createParser, useQueryStates } from "nuqs";
import { useCallback, useMemo } from "react";
import type { DateRangePickerValue } from "@/components/shared/date_picker_types";

const DAY_FORMAT = "YYYY-MM-DD";
const DEFAULT_RANGE_DAYS = 7;

const formatDay = (date: Date) => moment(date).format(DAY_FORMAT);

const parseAsDay = createParser({
  parse: (value: string) => {
    const day = moment(value, DAY_FORMAT, true);
    return day.isValid() ? day.format(DAY_FORMAT) : null;
  },
  serialize: (value: string) => value,
});

const buildParsers = (today: Date) => ({
  start_date: parseAsDay.withDefault(formatDay(moment(today).subtract(DEFAULT_RANGE_DAYS, "days").toDate())),
  end_date: parseAsDay.withDefault(formatDay(today)),
});

export interface MonitorDateRange {
  startDate: string;
  endDate: string;
  pickerValue: DateRangePickerValue;
  setPickerValue: (value: DateRangePickerValue) => void;
}

export function useMonitorDateRange(): MonitorDateRange {
  const parsers = useMemo(() => buildParsers(new Date()), []);
  const [{ start_date: startDate, end_date: endDate }, setRange] = useQueryStates(parsers);

  const pickerValue = useMemo<DateRangePickerValue>(
    () => ({
      from: moment(startDate, DAY_FORMAT).startOf("day").toDate(),
      to: moment(endDate, DAY_FORMAT).endOf("day").toDate(),
    }),
    [startDate, endDate],
  );

  const setPickerValue = useCallback(
    ({ from, to }: DateRangePickerValue) => {
      void setRange({
        start_date: from ? formatDay(from) : null,
        end_date: to ? formatDay(to) : null,
      });
    },
    [setRange],
  );

  return { startDate, endDate, pickerValue, setPickerValue };
}
