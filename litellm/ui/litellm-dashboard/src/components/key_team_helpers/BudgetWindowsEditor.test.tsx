import { describe, expect, it, vi } from "vitest";
import { fireEvent, renderWithProviders, screen } from "../../../tests/test-utils";
import { BudgetWindowsEditor } from "./BudgetWindowsEditor";

describe("BudgetWindowsEditor", () => {
  it("shows a compact budget placeholder next to the currency addon", () => {
    const onChange = vi.fn();
    renderWithProviders(
      <BudgetWindowsEditor value={[{ budget_duration: "24h", max_budget: null }]} onChange={onChange} />,
    );

    expect(screen.getByText("$")).toBeInTheDocument();
    fireEvent.change(screen.getByPlaceholderText("Max Budget"), { target: { value: "12.5" } });
    expect(onChange).toHaveBeenCalledWith([{ budget_duration: "24h", max_budget: 12.5 }]);
  });

  it("clears a budget without storing an invalid number", () => {
    const onChange = vi.fn();
    renderWithProviders(
      <BudgetWindowsEditor value={[{ budget_duration: "24h", max_budget: 10 }]} onChange={onChange} />,
    );

    fireEvent.change(screen.getByPlaceholderText("Max Budget"), { target: { value: "" } });
    expect(onChange).toHaveBeenCalledWith([{ budget_duration: "24h", max_budget: null }]);
  });

  it("allows adding a window when none exist", () => {
    const onChange = vi.fn();
    renderWithProviders(<BudgetWindowsEditor value={[]} onChange={onChange} />);

    fireEvent.click(screen.getByRole("button", { name: "+ Add Budget Window" }));
    expect(onChange).toHaveBeenCalledWith([{ budget_duration: "24h", max_budget: null }]);
  });
});
