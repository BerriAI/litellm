import { act, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useDebouncedValue } from "./useDebouncedValue";

beforeEach(() => vi.useFakeTimers());
afterEach(() => vi.useRealTimers());

describe("useDebouncedValue", () => {
  it("settles once on the latest value when changes arrive within the wait", () => {
    const { result, rerender } = renderHook(({ value }) => useDebouncedValue(value, 350), {
      initialProps: { value: { team: "" } },
    });
    const initialSettledAt = result.current.settledAt;
    rerender({ value: { team: "a" } });
    act(() => vi.advanceTimersByTime(200));
    rerender({ value: { team: "ab" } });
    act(() => vi.advanceTimersByTime(340));
    expect(result.current).toMatchObject({ value: { team: "" }, pending: true });
    act(() => vi.advanceTimersByTime(10));
    expect(result.current).toMatchObject({ value: { team: "ab" }, pending: false });
    expect(result.current.settledAt > initialSettledAt).toBe(true);
  });

  it("ignores a new reference with the same content", () => {
    const { result, rerender } = renderHook(({ value }) => useDebouncedValue(value, 350), {
      initialProps: { value: { filters: [{ key: "env", value: "prod" }] } },
    });
    const settledAt = result.current.settledAt;
    rerender({ value: { filters: [{ key: "env", value: "prod" }] } });
    expect(result.current.pending).toBe(false);
    rerender({ value: { filters: [{ key: "env", value: "stage" }] } });
    act(() => vi.advanceTimersByTime(300));
    rerender({ value: { filters: [{ key: "env", value: "stage" }] } });
    act(() => vi.advanceTimersByTime(50));
    expect(result.current).toMatchObject({ value: { filters: [{ key: "env", value: "stage" }] }, pending: false });
    expect(result.current.settledAt).not.toBe(settledAt);
  });
});
