import { useMemo, useState } from "react";
import type { DailyData } from "@/components/UsagePage/types";
import {
  coveringRange,
  defaultGranularity,
  type DayRange,
  type TrendGranularity,
  type TrendMetric,
} from "./usageTrendData";

export interface TrendSelection {
  readonly metric: TrendMetric;
  readonly granularity: TrendGranularity;
}

export interface TrendSelectionState {
  readonly selection: TrendSelection;
  readonly setMetric: (metric: TrendMetric) => void;
  readonly setGranularity: (granularity: TrendGranularity) => void;
}

export function useTrendSelection(range: DayRange | null, results: readonly DailyData[]): TrendSelectionState {
  const [metric, setMetric] = useState<TrendMetric>("tokens");
  const [chosenGranularity, setGranularity] = useState<TrendGranularity | null>(null);
  const autoGranularity = useMemo(
    () => (range ? defaultGranularity(coveringRange(range, results)) : "day"),
    [range, results],
  );
  const selection = useMemo(
    () => ({ metric, granularity: chosenGranularity ?? autoGranularity }),
    [metric, chosenGranularity, autoGranularity],
  );
  return { selection, setMetric, setGranularity };
}
