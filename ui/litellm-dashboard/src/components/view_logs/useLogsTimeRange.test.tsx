import { act, render, renderHook, waitFor } from "@testing-library/react";
import moment from "moment";
import { NuqsTestingAdapter, withNuqsTestingAdapter, type OnUrlUpdateFunction } from "nuqs/adapters/testing";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { LOCAL_DATETIME_FORMAT, useLogsTimeRange, type LogsRange, type LogsTimeRange } from "./useLogsTimeRange";

const renderTimeRange = (searchParams = "") => {
  const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
  const onChange = vi.fn();
  const adapterProps = { searchParams, onUrlUpdate, hasMemory: true, resetUrlUpdateQueueOnMount: false };
  const hook = renderHook(() => useLogsTimeRange(onChange), { wrapper: withNuqsTestingAdapter(adapterProps) });
  return { ...hook, onUrlUpdate, onChange };
};

const lastParams = (onUrlUpdate: ReturnType<typeof vi.fn<OnUrlUpdateFunction>>) =>
  onUrlUpdate.mock.calls.at(-1)?.[0].searchParams ?? new URLSearchParams();

const expectedRange = (
  range: LogsRange,
  [startTime, endTime]: readonly [string, string],
  [startInput, endInput]: readonly [string, string] = ["", ""],
): LogsTimeRange => ({ range, startTime, endTime, startInput, endInput });

const minutesBetween = (startTime: string, endTime: string) =>
  moment(endTime, LOCAL_DATETIME_FORMAT).diff(moment(startTime, LOCAL_DATETIME_FORMAT), "minutes");

describe("useLogsTimeRange", () => {
  beforeEach(() => {
    vi.stubEnv("TZ", "America/New_York");
    vi.useFakeTimers({ shouldAdvanceTime: true });
    vi.setSystemTime(new Date("2026-07-07T10:00:00Z"));
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.unstubAllEnvs();
  });

  it("anchors a preset's bounds at mount time so the query window does not drift per render", () => {
    const { result, rerender } = renderTimeRange("?range=15m");

    const initial = result.current.timeRange;
    expect(initial).toEqual(expectedRange("15m", ["2026-07-07T05:45", "2026-07-07T06:00"]));

    vi.setSystemTime(new Date("2026-07-07T10:05:00Z"));
    rerender();

    expect(result.current.timeRange).toEqual(initial);
  });

  it("re-anchors from now when the same preset is picked again", async () => {
    const { result, onChange } = renderTimeRange("?range=15m");
    const initial = result.current.timeRange;

    vi.setSystemTime(new Date("2026-07-07T10:05:00Z"));
    act(() => result.current.selectPreset("15m"));

    await waitFor(() => expect(result.current.timeRange.endTime).toBe(moment().format(LOCAL_DATETIME_FORMAT)));
    expect(minutesBetween(result.current.timeRange.startTime, result.current.timeRange.endTime)).toBe(15);
    expect(result.current.timeRange.startTime).not.toBe(initial.startTime);
    expect(onChange).toHaveBeenCalledTimes(1);
  });

  it("recomputes the bounds from now when the URL range changes underneath it, as on Back", () => {
    const seen: LogsTimeRange[] = [];
    function Probe() {
      seen.push(useLogsTimeRange(vi.fn()).timeRange);
      return null;
    }
    const tree = (searchParams: string) => (
      <NuqsTestingAdapter searchParams={searchParams} hasMemory>
        <Probe />
      </NuqsTestingAdapter>
    );
    const view = render(tree("?range=1h"));
    const initial = seen.at(-1);
    if (!initial) throw new Error("hook did not render");
    expect(minutesBetween(initial.startTime, initial.endTime)).toBe(60);

    vi.setSystemTime(new Date("2026-07-07T12:00:00Z"));
    view.rerender(tree("?range=15m"));

    const afterBack = seen.at(-1);
    if (!afterBack) throw new Error("hook did not re-render");
    expect(afterBack.range).toBe("15m");
    expect(afterBack.endTime).toBe(moment().format(LOCAL_DATETIME_FORMAT));
    expect(minutesBetween(afterBack.startTime, afterBack.endTime)).toBe(15);
  });

  it("round-trips a custom range through UTC minute-precision ISO strings", async () => {
    const { result, onUrlUpdate } = renderTimeRange("?range=custom&start=2026-07-01T10:00Z&end=2026-07-02T10:30Z");

    const bounds = ["2026-07-01T06:00", "2026-07-02T06:30"] as const;
    expect(result.current.timeRange).toEqual(expectedRange("custom", bounds, bounds));

    act(() => result.current.setEndTime("2026-07-03T08:15"));

    await waitFor(() => expect(lastParams(onUrlUpdate).get("end")).toBe("2026-07-03T12:15Z"));
    expect(result.current.timeRange.endTime).toBe("2026-07-03T08:15");
    expect(result.current.timeRange.endInput).toBe("2026-07-03T08:15");
  });

  it("keeps an unparsable or cleared custom bound empty in the input but queries the default window for it", async () => {
    const { result, onUrlUpdate } = renderTimeRange("?range=custom&start=not-a-date&end=2026-07-02T10:30Z");

    expect(result.current.timeRange).toEqual(
      expectedRange("custom", ["2026-07-06T06:00", "2026-07-02T06:30"], ["", "2026-07-02T06:30"]),
    );

    act(() => result.current.setEndTime(""));

    await waitFor(() => expect(lastParams(onUrlUpdate).has("end")).toBe(false));
    expect(result.current.timeRange.endInput).toBe("");
    expect(result.current.timeRange.endTime).toBe("2026-07-07T06:00");
  });

  it("ignores ?start= and ?end= unless the range is custom", () => {
    const { result } = renderTimeRange("?range=1h&start=2026-07-01T10:00Z&end=2026-07-02T10:30Z");

    expect(result.current.timeRange).toEqual(expectedRange("1h", ["2026-07-07T05:00", "2026-07-07T06:00"]));
  });

  it("picking a preset while in a custom range drops ?start= and ?end=", async () => {
    const { result, onUrlUpdate } = renderTimeRange("?range=custom&start=2026-07-01T10:00Z&end=2026-07-02T10:30Z");

    act(() => result.current.selectPreset("1h"));

    await waitFor(() => expect(lastParams(onUrlUpdate).get("range")).toBe("1h"));
    expect(lastParams(onUrlUpdate).has("start")).toBe(false);
    expect(lastParams(onUrlUpdate).has("end")).toBe(false);
    await waitFor(() => expect(result.current.timeRange.startTime).toBe("2026-07-07T05:00"));
    expect(result.current.timeRange.endTime).toBe("2026-07-07T06:00");
  });

  it("carries the current preset bounds into ?start= and ?end= when Custom Range is toggled on", async () => {
    const { result, onUrlUpdate, onChange } = renderTimeRange("?range=4h");
    const { startTime, endTime } = result.current.timeRange;

    act(() => result.current.toggleCustomRange());

    await waitFor(() => expect(lastParams(onUrlUpdate).get("range")).toBe("custom"));
    expect(lastParams(onUrlUpdate).get("start")).toBe("2026-07-07T06:00Z");
    expect(lastParams(onUrlUpdate).get("end")).toBe("2026-07-07T10:00Z");
    const bounds = ["2026-07-07T02:00", "2026-07-07T06:00"] as const;
    expect([startTime, endTime]).toEqual(bounds);
    expect(result.current.timeRange).toEqual(expectedRange("custom", bounds, bounds));
    expect(onChange).toHaveBeenCalledTimes(1);
  });

  it("toggling Custom Range off returns to the preset picked before it, with bounds refreshed from now", async () => {
    const { result, onUrlUpdate, onChange } = renderTimeRange("?range=4h");

    act(() => result.current.toggleCustomRange());
    await waitFor(() => expect(lastParams(onUrlUpdate).get("range")).toBe("custom"));
    act(() => result.current.setStartTime("2026-07-01T10:00"));
    await waitFor(() => expect(lastParams(onUrlUpdate).get("start")).toBe("2026-07-01T14:00Z"));
    expect(result.current.timeRange.startTime).toBe("2026-07-01T10:00");

    vi.setSystemTime(new Date("2026-07-07T11:30:00Z"));
    act(() => result.current.toggleCustomRange());

    await waitFor(() => expect(lastParams(onUrlUpdate).get("range")).toBe("4h"));
    expect(lastParams(onUrlUpdate).has("start")).toBe(false);
    expect(lastParams(onUrlUpdate).has("end")).toBe(false);
    await waitFor(() => expect(result.current.timeRange.range).toBe("4h"));
    expect(result.current.timeRange.endTime).toBe(moment().format(LOCAL_DATETIME_FORMAT));
    expect(minutesBetween(result.current.timeRange.startTime, result.current.timeRange.endTime)).toBe(4 * 60);
    expect(onChange).toHaveBeenCalledTimes(3);
  });

  it("falls back to the default preset when a custom range restored from the URL is toggled off", async () => {
    const { result, onUrlUpdate } = renderTimeRange("?range=custom&start=2026-07-01T10:00Z&end=2026-07-02T10:30Z");

    act(() => result.current.toggleCustomRange());

    await waitFor(() => expect(result.current.timeRange.range).toBe("24h"));
    expect(lastParams(onUrlUpdate).has("range")).toBe(false);
    expect(lastParams(onUrlUpdate).has("start")).toBe(false);
    expect(result.current.timeRange.endTime).toBe(moment().format(LOCAL_DATETIME_FORMAT));
    expect(minutesBetween(result.current.timeRange.startTime, result.current.timeRange.endTime)).toBe(24 * 60);
  });

  it("re-anchors from now when the URL leaves a custom range for the preset it started from", () => {
    const seen: LogsTimeRange[] = [];
    function Probe() {
      seen.push(useLogsTimeRange(vi.fn()).timeRange);
      return null;
    }
    const tree = (searchParams: string) => (
      <NuqsTestingAdapter searchParams={searchParams} hasMemory>
        <Probe />
      </NuqsTestingAdapter>
    );
    const view = render(tree("?range=1h"));
    view.rerender(tree("?range=custom&start=2026-07-01T10:00Z&end=2026-07-02T10:30Z"));
    expect(seen.at(-1)?.range).toBe("custom");

    vi.setSystemTime(new Date("2026-07-07T12:00:00Z"));
    view.rerender(tree("?range=1h"));

    const restored = seen.at(-1);
    if (!restored) throw new Error("hook did not re-render");
    expect(restored.range).toBe("1h");
    expect(restored.endTime).toBe(moment().format(LOCAL_DATETIME_FORMAT));
    expect(minutesBetween(restored.startTime, restored.endTime)).toBe(60);
  });

  it("reset clears every time range key and refreshes the default bounds from now", async () => {
    const { result, onUrlUpdate, onChange } = renderTimeRange(
      "?range=custom&start=2026-07-01T10:00Z&end=2026-07-02T10:30Z",
    );

    vi.setSystemTime(new Date("2026-07-07T13:00:00Z"));
    act(() => result.current.reset());

    await waitFor(() => expect(lastParams(onUrlUpdate).has("range")).toBe(false));
    expect(lastParams(onUrlUpdate).has("start")).toBe(false);
    expect(lastParams(onUrlUpdate).has("end")).toBe(false);
    expect(result.current.timeRange.range).toBe("24h");
    expect(result.current.timeRange.endTime).toBe(moment().format(LOCAL_DATETIME_FORMAT));
    expect(onChange).toHaveBeenCalledTimes(1);
  });
});
