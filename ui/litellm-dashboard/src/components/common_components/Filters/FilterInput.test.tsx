import { act, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { FilterInput } from "./FilterInput";

describe("FilterInput", () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.runOnlyPendingTimers();
    vi.useRealTimers();
  });

  it("should render", () => {
    const onChange = vi.fn();
    render(<FilterInput value="" onChange={onChange} placeholder="Search..." />);
    expect(screen.getByPlaceholderText("Search...")).toBeInTheDocument();
  });

  it("should call onChange with debounced value", async () => {
    const onChange = vi.fn();
    render(<FilterInput value="" onChange={onChange} placeholder="Search..." />);

    const input = screen.getByPlaceholderText("Search...");

    act(() => {
      fireEvent.change(input, { target: { value: "test" } });
    });

    expect(onChange).not.toHaveBeenCalled();

    act(() => {
      vi.advanceTimersByTime(300);
    });

    expect(onChange).toHaveBeenCalledWith("test");
  });

  it("should display the value prop", () => {
    const onChange = vi.fn();
    render(<FilterInput value="initial value" onChange={onChange} placeholder="Search..." />);
    const input = screen.getByPlaceholderText("Search...") as HTMLInputElement;
    expect(input.value).toBe("initial value");
  });

  it("should update local value immediately when typing", async () => {
    const onChange = vi.fn();
    render(<FilterInput value="" onChange={onChange} placeholder="Search..." />);

    const input = screen.getByPlaceholderText("Search...") as HTMLInputElement;

    act(() => {
      fireEvent.change(input, { target: { value: "a" } });
    });

    expect(input.value).toBe("a");
  });
});
