import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { BudgetFallbacksEditor } from "./BudgetFallbacksEditor";

const MODELS = ["gpt-4", "gpt-3.5-turbo", "claude-3", "claude-haiku"];

describe("BudgetFallbacksEditor", () => {
  it("renders empty state with add button", () => {
    const onChange = vi.fn();
    render(<BudgetFallbacksEditor value={{}} onChange={onChange} availableModels={MODELS} />);
    expect(screen.getByText("Add Budget Fallback")).toBeTruthy();
    expect(screen.getByText(/reroute to fallback models/)).toBeTruthy();
  });

  it("renders existing entries from value prop", () => {
    const onChange = vi.fn();
    render(
      <BudgetFallbacksEditor
        value={{ "gpt-4": ["gpt-3.5-turbo", "claude-3"] }}
        onChange={onChange}
        availableModels={MODELS}
      />,
    );
    expect(screen.getByText("IF BUDGET EXCEEDED, TRY")).toBeTruthy();
    expect(screen.getByText("Primary Model")).toBeTruthy();
    expect(screen.getByText("Fallback Models")).toBeTruthy();
  });

  it("adds a new empty entry when clicking add button", async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    render(<BudgetFallbacksEditor value={{}} onChange={onChange} availableModels={MODELS} />);

    await user.click(screen.getByText("Add Budget Fallback"));
    expect(screen.getByText("Primary Model")).toBeTruthy();
    expect(onChange).toHaveBeenCalledWith({});
  });

  it("removes an entry and emits updated dict", async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    const { container } = render(
      <BudgetFallbacksEditor
        value={{ "gpt-4": ["gpt-3.5-turbo"], "claude-3": ["claude-haiku"] }}
        onChange={onChange}
        availableModels={MODELS}
      />,
    );

    const removeButtons = container.querySelectorAll<HTMLButtonElement>(".relative > button[type='button']");
    expect(removeButtons.length).toBe(2);

    await user.click(removeButtons[0]);
    expect(onChange).toHaveBeenLastCalledWith({ "claude-3": ["claude-haiku"] });
  });

  it("renders multiple entries for multiple fallback groups", () => {
    const onChange = vi.fn();
    render(
      <BudgetFallbacksEditor
        value={{ "gpt-4": ["claude-3"], "gpt-3.5-turbo": ["claude-haiku"] }}
        onChange={onChange}
        availableModels={MODELS}
      />,
    );
    const labels = screen.getAllByText("Primary Model");
    expect(labels.length).toBe(2);
  });

  it("resets internal state when remounted with empty value via key prop", () => {
    const onChange = vi.fn();
    const { rerender } = render(
      <BudgetFallbacksEditor
        key={1}
        value={{ "gpt-4": ["gpt-3.5-turbo"] }}
        onChange={onChange}
        availableModels={MODELS}
      />,
    );
    expect(screen.getAllByText("Primary Model").length).toBe(1);

    rerender(<BudgetFallbacksEditor key={2} value={{}} onChange={onChange} availableModels={MODELS} />);
    expect(screen.queryByText("Primary Model")).toBeNull();
    expect(screen.getByText("Add Budget Fallback")).toBeTruthy();
  });

  it("shows ordering hint when multiple fallback models are configured", () => {
    const onChange = vi.fn();
    render(
      <BudgetFallbacksEditor
        value={{ "gpt-4": ["gpt-3.5-turbo", "claude-3"] }}
        onChange={onChange}
        availableModels={MODELS}
      />,
    );
    expect(screen.getByText(/first model still within its own budget/)).toBeTruthy();
  });
});
