import { act, render, renderHook, waitFor } from "@testing-library/react";
import moment from "moment";
import { NuqsTestingAdapter, withNuqsTestingAdapter, type OnUrlUpdateFunction } from "nuqs/adapters/testing";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { LOCAL_DATETIME_FORMAT, useLogsTimeRange, type LogsTimeRange } from "./useLogsTimeRange";

const renderTimeRange = (searchParams = "") => {
  const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
  const onChange = vi.fn();
  const hook = renderHook(() => useLogsTimeRange(onChange), {
    wrapper: withNuqsTestingAdapter({ searchParams, onUrlUpdate, hasMemory: true }),
  });
  return { ...hook, onUrlUpdate, onChange };
};

const lastParams = (onUrlUpdate: ReturnType<typeof vi.fn<OnUrlUpdateFunction>>) =>
  onUrlUpdate.mock.calls.at(-1)?.[0].searchParams ?? new URLSearchParams();

const minutesBetween = (startTime: string, endTime: string) =>
  moment(endTime, LOCAL_DATETIME_FORMAT).diff(moment(startTime, LOCAL_DATETIME_FORMAT), "minutes");

describe("useLogsTimeRange", () => {
  beforeEach(() => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    vi.setSystemTime(new Date("2026-07-07T10:00:00Z"));
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("anchors a preset's bounds at mount time so the query window does not drift per render", () => {
    const { result, rerender } = renderTimeRange("?range=15m");

    const initial = result.current.timeRange;
    expect(minutesBetween(initial.startTime, initial.endTime)).toBe(15);
    expect(initial.endTime).toBe(moment().format(LOCAL_DATETIME_FORMAT));

    vi.setSystemTime(new Date("2026-07-07T10:05:00Z"));
    rerender();

    expect(result.current.timeRange).toBe(initial);
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

    expect(result.current.timeRange).toEqual({
      range: "custom",
      startTime: moment("2026-07-01T10:00Z").format(LOCAL_DATETIME_FORMAT),
      endTime: moment("2026-07-02T10:30Z").format(LOCAL_DATETIME_FORMAT),
    });

    act(() => result.current.setEndTime("2026-07-03T08:15"));

    await waitFor(() =>
      expect(lastParams(onUrlUpdate).get("end")).toBe(moment("2026-07-03T08:15").utc().format("YYYY-MM-DDTHH:mm[Z]")),
    );
    expect(result.current.timeRange.endTime).toBe("2026-07-03T08:15");
  });

  it("treats an unparsable or cleared custom bound as empty instead of inventing a date", async () => {
    const { result, onUrlUpdate } = renderTimeRange("?range=custom&start=not-a-date&end=2026-07-02T10:30Z");

    expect(result.current.timeRange.startTime).toBe("");

    act(() => result.current.setEndTime(""));

    await waitFor(() => expect(lastParams(onUrlUpdate).has("end")).toBe(false));
    expect(result.current.timeRange.endTime).toBe("");
  });

  it("carries the current preset bounds into ?start= and ?end= when Custom Range is toggled on", async () => {
    const { result, onUrlUpdate, onChange } = renderTimeRange("?range=4h");
    const { startTime, endTime } = result.current.timeRange;

    act(() => result.current.toggleCustomRange());

    await waitFor(() => expect(lastParams(onUrlUpdate).get("range")).toBe("custom"));
    expect(lastParams(onUrlUpdate).get("start")).toBe(
      moment(startTime, LOCAL_DATETIME_FORMAT).utc().format("YYYY-MM-DDTHH:mm[Z]"),
    );
    expect(lastParams(onUrlUpdate).get("end")).toBe(
      moment(endTime, LOCAL_DATETIME_FORMAT).utc().format("YYYY-MM-DDTHH:mm[Z]"),
    );
    expect(result.current.timeRange).toEqual({ range: "custom", startTime, endTime });
    expect(onChange).toHaveBeenCalledTimes(1);

    act(() => result.current.toggleCustomRange());

    await waitFor(() => expect(lastParams(onUrlUpdate).has("range")).toBe(false));
    expect(lastParams(onUrlUpdate).has("start")).toBe(false);
    expect(result.current.timeRange.range).toBe("24h");
    expect(minutesBetween(result.current.timeRange.startTime, result.current.timeRange.endTime)).toBe(24 * 60);
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
