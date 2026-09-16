import moment from "moment";
import { createParser, parseAsStringLiteral, useQueryStates } from "nuqs";
import { useCallback, useMemo, useState } from "react";

import {
  DEFAULT_QUICK_SELECT_PRESET,
  QUICK_SELECT_OPTIONS,
  QUICK_SELECT_PRESET_IDS,
  type QuickSelectPresetId,
} from "./constants";

export const CUSTOM_RANGE = "custom";
export type LogsRange = QuickSelectPresetId | typeof CUSTOM_RANGE;

export const LOCAL_DATETIME_FORMAT = "YYYY-MM-DDTHH:mm";
const LOCAL_DATETIME_INPUT_FORMATS = [LOCAL_DATETIME_FORMAT, `${LOCAL_DATETIME_FORMAT}:ss`];
const URL_DATETIME_FORMAT = "YYYY-MM-DDTHH:mm[Z]";

const RANGE_IDS: readonly LogsRange[] = [...QUICK_SELECT_PRESET_IDS, CUSTOM_RANGE];

const isLocalDateTime = (value: string): boolean => moment(value, LOCAL_DATETIME_INPUT_FORMATS, true).isValid();

const parseAsLocalDateTime = createParser<string>({
  parse: (value) => {
    const parsed = moment(value, moment.ISO_8601, true);
    return parsed.isValid() ? parsed.format(LOCAL_DATETIME_FORMAT) : null;
  },
  serialize: (value) => moment(value, LOCAL_DATETIME_INPUT_FORMATS, true).utc().format(URL_DATETIME_FORMAT),
});

const TIME_RANGE_PARSERS = {
  range: parseAsStringLiteral(RANGE_IDS).withDefault(DEFAULT_QUICK_SELECT_PRESET),
  start: parseAsLocalDateTime,
  end: parseAsLocalDateTime,
};

export interface LogsTimeRange {
  range: LogsRange;
  startTime: string;
  endTime: string;
}

interface PresetWindow {
  preset: QuickSelectPresetId;
  startTime: string;
  endTime: string;
}

interface Anchor {
  range: LogsRange;
  window: PresetWindow;
}

const presetWindow = (preset: QuickSelectPresetId): PresetWindow => {
  const option = QUICK_SELECT_OPTIONS.find((candidate) => candidate.id === preset) ?? QUICK_SELECT_OPTIONS[0];
  const now = moment();
  return {
    preset,
    startTime: now.clone().subtract(option.value, option.unit).format(LOCAL_DATETIME_FORMAT),
    endTime: now.format(LOCAL_DATETIME_FORMAT),
  };
};

const anchorFor = (range: LogsRange, previous: PresetWindow | null): Anchor => {
  if (range !== CUSTOM_RANGE) return { range, window: presetWindow(range) };
  return { range, window: previous ?? presetWindow(DEFAULT_QUICK_SELECT_PRESET) };
};

export interface LogsTimeRangeState {
  timeRange: LogsTimeRange;
  selectPreset: (preset: QuickSelectPresetId) => void;
  toggleCustomRange: () => void;
  setStartTime: (value: string) => void;
  setEndTime: (value: string) => void;
  reset: () => void;
}

export function useLogsTimeRange(onChange: () => void): LogsTimeRangeState {
  const [{ range, start, end }, setParams] = useQueryStates(TIME_RANGE_PARSERS);
  const [anchor, setAnchor] = useState<Anchor>(() => anchorFor(range, null));
  if (anchor.range !== range) setAnchor(anchorFor(range, anchor.window));

  const isCustom = range === CUSTOM_RANGE;
  const lastPreset = anchor.window.preset;
  const startTime = isCustom ? start ?? "" : anchor.window.startTime;
  const endTime = isCustom ? end ?? "" : anchor.window.endTime;
  const timeRange = useMemo<LogsTimeRange>(() => ({ range, startTime, endTime }), [range, startTime, endTime]);

  const selectPreset = useCallback(
    (preset: QuickSelectPresetId) => {
      void setParams({ range: preset, start: null, end: null });
      setAnchor(anchorFor(preset, null));
      onChange();
    },
    [setParams, onChange],
  );

  const toggleCustomRange = useCallback(() => {
    void setParams(
      isCustom
        ? { range: lastPreset, start: null, end: null }
        : { range: CUSTOM_RANGE, start: startTime || null, end: endTime || null },
    );
    onChange();
  }, [setParams, isCustom, lastPreset, startTime, endTime, onChange]);

  const setStartTime = useCallback(
    (value: string) => {
      void setParams({ start: isLocalDateTime(value) ? value : null });
      onChange();
    },
    [setParams, onChange],
  );

  const setEndTime = useCallback(
    (value: string) => {
      void setParams({ end: isLocalDateTime(value) ? value : null });
      onChange();
    },
    [setParams, onChange],
  );

  const reset = useCallback(() => {
    void setParams(null);
    setAnchor(anchorFor(DEFAULT_QUICK_SELECT_PRESET, null));
    onChange();
  }, [setParams, onChange]);

  return useMemo<LogsTimeRangeState>(
    () => ({ timeRange, selectPreset, toggleCustomRange, setStartTime, setEndTime, reset }),
    [timeRange, selectPreset, toggleCustomRange, setStartTime, setEndTime, reset],
  );
}
