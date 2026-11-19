import { describe, expect, it } from "vitest";
import { renderWithProviders, screen, within } from "../../../../../tests/test-utils";
import CustomerBudgetSpend from "./CustomerBudgetSpend";

describe("CustomerBudgetSpend usage colors", () => {
  it.each([
    [50, "#16A34A"],
    [70, "#EAB308"],
    [85, "#F97316"],
    [92, "#F05A2E"],
    [98, "#E5484D"],
  ] as const)("matches the remaining pill to the filled bar at %s%% usage", (spend, color) => {
    renderWithProviders(<CustomerBudgetSpend spend={spend} limit={100} label="Monthly spend / limit" />);

    const progress = screen.getByRole("progressbar", { name: "Monthly spend / limit" });
    expect(progress).toHaveAttribute("aria-valuenow", String(spend));
    expect(within(progress).getByRole("generic")).toHaveStyle({
      backgroundColor: color,
      width: `${spend}%`,
    });
    expect(screen.getByText(`$${(100 - spend).toFixed(2)} remains`)).toHaveStyle({
      backgroundColor: `color-mix(in oklab, ${color} 12%, transparent)`,
      color: `color-mix(in oklab, ${color} 60%, var(--foreground))`,
    });
  });
});
