import { useState } from "react";

export const LOG_FILTERS = ["all", "blocked", "flagged", "passed"] as const;
export type LogFilter = (typeof LOG_FILTERS)[number];
export const LOG_SAMPLE_SIZES = [10, 50, 100] as const;
export type LogSampleSize = (typeof LOG_SAMPLE_SIZES)[number];

export interface LogViewerState {
  filter: LogFilter;
  setFilter: (filter: LogFilter) => void;
  sampleSize: LogSampleSize;
  setSampleSize: (size: LogSampleSize) => void;
  requestId: string | null;
  setRequestId: (requestId: string | null) => void;
}

export function useLocalLogViewerState(initialFilter: LogFilter): LogViewerState {
  const [filter, setFilter] = useState<LogFilter>(initialFilter);
  const [sampleSize, setSampleSize] = useState<LogSampleSize>(10);
  const [requestId, setRequestId] = useState<string | null>(null);
  return { filter, setFilter, sampleSize, setSampleSize, requestId, setRequestId };
}
