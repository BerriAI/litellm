import { act, fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import ModelSelector from "./ModelSelector";

vi.mock("@/components/llm_calls/fetch_models", () => ({
  fetchAvailableModels: vi.fn().mockResolvedValue([]),
}));

const openCustomModelInput = async () => {
  const user = userEvent.setup();
  await user.click(screen.getByRole("combobox"));
  await user.click(await screen.findByText("Enter custom model"));
  return screen.getByPlaceholderText("Enter custom model name");
};

describe("ModelSelector custom model debounce", () => {
  afterEach(() => {
    act(() => {
      vi.runOnlyPendingTimers();
    });
    vi.useRealTimers();
  });

  it("does not call onChange before the debounce wait elapses", async () => {
    const onChange = vi.fn();
    render(<ModelSelector accessToken="test-token" onChange={onChange} />);

    const input = await openCustomModelInput();
    vi.useFakeTimers();

    act(() => {
      fireEvent.change(input, { target: { value: "gpt-4o" } });
    });

    expect(onChange).not.toHaveBeenCalled();

    act(() => {
      vi.advanceTimersByTime(499);
    });

    expect(onChange).not.toHaveBeenCalled();
  });

  it("calls onChange exactly once with the last typed value after the wait", async () => {
    const onChange = vi.fn();
    render(<ModelSelector accessToken="test-token" onChange={onChange} />);

    const input = await openCustomModelInput();
    vi.useFakeTimers();

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

  it("does not call onChange when unmounted mid-wait", async () => {
    const onChange = vi.fn();
    const { unmount } = render(<ModelSelector accessToken="test-token" onChange={onChange} />);

    const input = await openCustomModelInput();
    vi.useFakeTimers();

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
