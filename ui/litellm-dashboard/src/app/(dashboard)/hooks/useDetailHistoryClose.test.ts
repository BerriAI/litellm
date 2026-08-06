import { act, renderHook } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { useDetailHistoryClose } from "./useDetailHistoryClose";

const backSpy = vi.spyOn(window.history, "back").mockImplementation(() => {});

afterEach(() => {
  backSpy.mockClear();
});

describe("useDetailHistoryClose", () => {
  it("close after an in-session open pops the pushed entry instead of clearing params", () => {
    const clearParams = vi.fn();
    const { result } = renderHook(() => useDetailHistoryClose(clearParams));

    act(() => result.current.markOpened());
    act(() => result.current.close());

    expect(backSpy).toHaveBeenCalledTimes(1);
    expect(clearParams).not.toHaveBeenCalled();
  });

  it("close without an in-session open clears params so a deep-linked visitor is not ejected", () => {
    const clearParams = vi.fn();
    const { result } = renderHook(() => useDetailHistoryClose(clearParams));

    act(() => result.current.close());

    expect(backSpy).not.toHaveBeenCalled();
    expect(clearParams).toHaveBeenCalledTimes(1);
  });

  it("a browser Back consumes the pushed entry, so a later close falls back to clearing", () => {
    const clearParams = vi.fn();
    const { result } = renderHook(() => useDetailHistoryClose(clearParams));

    act(() => result.current.markOpened());
    act(() => window.dispatchEvent(new PopStateEvent("popstate")));
    act(() => result.current.close());

    expect(backSpy).not.toHaveBeenCalled();
    expect(clearParams).toHaveBeenCalledTimes(1);
  });

  it("the popstate caused by close itself does not consume a second entry", () => {
    const clearParams = vi.fn();
    const { result } = renderHook(() => useDetailHistoryClose(clearParams));

    act(() => result.current.markOpened());
    act(() => result.current.markOpened());
    act(() => result.current.close());
    act(() => window.dispatchEvent(new PopStateEvent("popstate")));
    act(() => result.current.close());

    expect(backSpy).toHaveBeenCalledTimes(2);
    expect(clearParams).not.toHaveBeenCalled();
  });
});
