import { act, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import ModelSelector from "./ModelSelector";

vi.mock("@/components/llm_calls/fetch_models", () => ({
  fetchAvailableModels: vi.fn().mockResolvedValue([]),
}));

const openCustomModelInput = () => {
  const selector = document.querySelector(".ant-select-selector");
  expect(selector).toBeTruthy();
  act(() => {
    fireEvent.mouseDown(selector!);
  });
  act(() => {
    fireEvent.click(screen.getByText("Enter custom model"));
  });
  return screen.getByPlaceholderText("Enter custom model name");
};

describe("ModelSelector custom model debounce", () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    act(() => {
      vi.runOnlyPendingTimers();
    });
    vi.useRealTimers();
  });

  it("does not call onChange before the debounce wait elapses", () => {
    const onChange = vi.fn();
    render(<ModelSelector accessToken="test-token" onChange={onChange} />);

    const input = openCustomModelInput();

    act(() => {
      fireEvent.change(input, { target: { value: "gpt-4o" } });
    });

    expect(onChange).not.toHaveBeenCalled();

    act(() => {
      vi.advanceTimersByTime(499);
    });

    expect(onChange).not.toHaveBeenCalled();
  });

  it("calls onChange exactly once with the last typed value after the wait", () => {
    const onChange = vi.fn();
    render(<ModelSelector accessToken="test-token" onChange={onChange} />);

    const input = openCustomModelInput();

    act(() => {
      fireEvent.change(input, { target: { value: "g" } });
      fireEvent.change(input, { target: { value: "gp" } });
      fireEvent.change(input, { target: { value: "gpt-5.2" } });
    });

    expect(onChange).not.toHaveBeenCalled();

    act(() => {
      vi.advanceTimersByTime(500);
    });

    expect(onChange).toHaveBeenCalledTimes(1);
    expect(onChange).toHaveBeenCalledWith("gpt-5.2");
  });

  it("does not call onChange when unmounted mid-wait", () => {
    const onChange = vi.fn();
    const { unmount } = render(<ModelSelector accessToken="test-token" onChange={onChange} />);

    const input = openCustomModelInput();

    act(() => {
      fireEvent.change(input, { target: { value: "gpt-4o" } });
    });

    unmount();

    act(() => {
      vi.advanceTimersByTime(500);
    });

    expect(onChange).not.toHaveBeenCalled();
  });
});
