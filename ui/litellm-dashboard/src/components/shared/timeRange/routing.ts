import { parseAsInteger, parseAsNumberLiteral, useQueryState, useQueryStates } from "nuqs";
import { useCallback } from "react";

import { DEFAULT_RANGE_HOURS, isRangeHours, RANGE_HOURS, type TimeWindow } from "./timeRange";

export const TIME_RANGE_PARSERS = {
  hours: parseAsNumberLiteral(RANGE_HOURS).withDefault(DEFAULT_RANGE_HOURS),
  from: parseAsInteger,
  to: parseAsInteger,
};

export function useRangeHoursRouting(): [number, (hours: number) => void] {
  const [hours, setHours] = useQueryState("hours", TIME_RANGE_PARSERS.hours);
  const setRangeHours = useCallback((next: number) => void (isRangeHours(next) && setHours(next)), [setHours]);
  return [hours, setRangeHours];
}

/** A timeline brush narrows the list to a window inside the range; it is dropped whenever the range changes. */
export function useZoomRouting(): [TimeWindow | null, (zoom: TimeWindow | null) => void] {
  const [{ from, to }, setParams] = useQueryStates({ from: TIME_RANGE_PARSERS.from, to: TIME_RANGE_PARSERS.to });
  const setZoom = useCallback(
    (zoom: TimeWindow | null) => void setParams({ from: zoom?.startMs ?? null, to: zoom?.endMs ?? null }),
    [setParams],
  );
  return [from !== null && to !== null && from < to ? { startMs: from, endMs: to } : null, setZoom];
}
