import { act, renderHook } from "@testing-library/react";
import type { UIEvent } from "react";
import { describe, expect, it, vi } from "vitest";

import { usePaginatedCombobox } from "./usePaginatedCombobox";

const scrollEvent = (scrollTop: number, clientHeight: number, scrollHeight: number) =>
  ({ currentTarget: { scrollTop, clientHeight, scrollHeight } }) as UIEvent<HTMLDivElement>;

const setup = (overrides: Partial<Parameters<typeof usePaginatedCombobox>[0]> = {}) => {
  const onLoadMore = vi.fn();
  const onSearchChange = vi.fn();
  const options = { onSearchChange, onLoadMore, hasNextPage: true, isFetchingNextPage: false, ...overrides };
  const { result } = renderHook(() => usePaginatedCombobox(options));
  return { result, onLoadMore, onSearchChange };
};

describe("usePaginatedCombobox", () => {
  it("loads the next page only once the list is scrolled most of the way down", () => {
    const { result, onLoadMore } = setup();

    result.current.handleScroll(scrollEvent(0, 100, 1000));
    expect(onLoadMore).not.toHaveBeenCalled();

    result.current.handleScroll(scrollEvent(690, 100, 1000));
    expect(onLoadMore).not.toHaveBeenCalled();

    result.current.handleScroll(scrollEvent(700, 100, 1000));
    expect(onLoadMore).toHaveBeenCalledTimes(1);
  });

  it("does not ask for a page that does not exist or is already in flight", () => {
    const exhausted = setup({ hasNextPage: false });
    exhausted.result.current.handleScroll(scrollEvent(900, 100, 1000));
    expect(exhausted.onLoadMore).not.toHaveBeenCalled();

    const inFlight = setup({ isFetchingNextPage: true });
    inFlight.result.current.handleScroll(scrollEvent(900, 100, 1000));
    expect(inFlight.onLoadMore).not.toHaveBeenCalled();
  });

  it("ignores a scroll on an unmeasured list, whose ratio divides by a zero scroll height", () => {
    const { result, onLoadMore } = setup();
    result.current.handleScroll(scrollEvent(0, 100, 0));
    expect(onLoadMore).not.toHaveBeenCalled();
  });

  it("searches on typing and clearing, but not on reasons that leave the query unchanged", async () => {
    vi.useFakeTimers();
    try {
      const { result, onSearchChange } = setup();

      result.current.handleSearchInput("alpha", "item-press");
      result.current.handleSearchInput("alpha", "list-navigation");
      await act(async () => {
        vi.advanceTimersByTime(1000);
      });
      expect(onSearchChange).not.toHaveBeenCalled();

      result.current.handleSearchInput("alpha", "input-change");
      await act(async () => {
        vi.advanceTimersByTime(1000);
      });
      expect(onSearchChange).toHaveBeenCalledWith("alpha");

      result.current.handleSearchInput("", "input-clear");
      await act(async () => {
        vi.advanceTimersByTime(1000);
      });
      expect(onSearchChange).toHaveBeenLastCalledWith("");
    } finally {
      vi.useRealTimers();
    }
  });

  it("debounces typing so one search leaves for a burst of keystrokes", async () => {
    vi.useFakeTimers();
    try {
      const { result, onSearchChange } = setup();

      result.current.handleSearchInput("a", "input-change");
      result.current.handleSearchInput("al", "input-change");
      result.current.handleSearchInput("alp", "input-change");
      await act(async () => {
        vi.advanceTimersByTime(299);
      });
      expect(onSearchChange).not.toHaveBeenCalled();

      await act(async () => {
        vi.advanceTimersByTime(1);
      });
      expect(onSearchChange).toHaveBeenCalledTimes(1);
      expect(onSearchChange).toHaveBeenCalledWith("alp");
    } finally {
      vi.useRealTimers();
    }
  });
});
