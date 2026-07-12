import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { SpendBudgetCell } from "./spend_budget_cell";

const indicator = (container: HTMLElement) => container.querySelector('[data-slot="meter-indicator"]');

describe("SpendBudgetCell", () => {
  it("shows Unlimited and renders no meter when there is no budget", () => {
    const { container } = render(<SpendBudgetCell spend={0.5} maxBudget={null} />);
    expect(screen.getByText("· Unlimited")).toBeInTheDocument();
    expect(screen.queryByRole("meter")).not.toBeInTheDocument();
    expect(indicator(container)).toBeNull();
  });

  it("shows $0.00 for zero or undefined spend, never a hyphen", () => {
    const { rerender } = render(<SpendBudgetCell spend={0} maxBudget={100} />);
    expect(screen.getByText("$0.00")).toBeInTheDocument();
    expect(screen.queryByText("-")).not.toBeInTheDocument();
    rerender(<SpendBudgetCell spend={null} maxBudget={null} />);
    expect(screen.getByText("$0.00")).toBeInTheDocument();
    expect(screen.queryByText("-")).not.toBeInTheDocument();
  });

  it("renders a meter carrying the spend and budget when a budget exists", () => {
    render(<SpendBudgetCell spend={25} maxBudget={100} />);
    const meter = screen.getByRole("meter");
    expect(meter).toHaveAttribute("aria-valuenow", "25");
    expect(meter).toHaveAttribute("aria-valuemax", "100");
    expect(screen.getByText("of $100")).toBeInTheDocument();
  });

  it("keeps the default tone below 80% usage", () => {
    const { container } = render(<SpendBudgetCell spend={50} maxBudget={100} />);
    expect(indicator(container)?.className).toContain("bg-primary");
  });

  it("switches to the warning tone at 80% usage", () => {
    const { container } = render(<SpendBudgetCell spend={80} maxBudget={100} />);
    expect(indicator(container)?.className).toContain("bg-amber-500");
  });

  it("switches to the over tone above 100% usage", () => {
    const { container } = render(<SpendBudgetCell spend={150} maxBudget={100} />);
    expect(indicator(container)?.className).toContain("bg-destructive");
  });

  it("falls back to the team budget and labels it", () => {
    render(<SpendBudgetCell spend={10} maxBudget={null} teamMaxBudget={200} />);
    expect(screen.getByText("of $200 (Team)")).toBeInTheDocument();
    expect(screen.getByRole("meter")).toHaveAttribute("aria-valuemax", "200");
  });
});
