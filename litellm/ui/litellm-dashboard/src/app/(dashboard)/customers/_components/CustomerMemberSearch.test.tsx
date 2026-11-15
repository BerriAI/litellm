import { act, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { DEBOUNCE_WAIT_MS } from "@/utils/debounceConstants";
import CustomerMemberSearch from "./CustomerMemberSearch";

describe("CustomerMemberSearch", () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });
  afterEach(() => {
    vi.useRealTimers();
  });

  it("drops a pending search when the URL value changes underneath it", () => {
    const onSearch = vi.fn();
    const { rerender } = render(<CustomerMemberSearch value="zoe" onSearch={onSearch} />);
    const search = screen.getByRole("searchbox");

    fireEvent.change(search, { target: { value: "9001" } });
    rerender(<CustomerMemberSearch value="" onSearch={onSearch} />);
    act(() => {
      vi.advanceTimersByTime(DEBOUNCE_WAIT_MS * 2);
    });
    expect(onSearch).not.toHaveBeenCalled();
    expect(search).toHaveValue("");

    fireEvent.change(search, { target: { value: " emp " } });
    act(() => {
      vi.advanceTimersByTime(DEBOUNCE_WAIT_MS * 2);
    });
    expect(onSearch.mock.calls).toEqual([["emp"]]);
  });
});
