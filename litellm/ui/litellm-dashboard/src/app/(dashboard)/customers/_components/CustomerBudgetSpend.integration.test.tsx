import { describe, expect, it } from "vitest";
import { renderWithProviders, screen, within } from "../../../../../tests/test-utils";
import CustomerBudgetSpend from "./CustomerBudgetSpend";

describe("CustomerBudgetSpend", () => {
  it.each([
    [50, "#16A34A"],
    [70, "#EAB308"],
    [85, "#F97316"],
    [92, "#F05A2E"],
    [98, "#E5484D"],
  ] as const)("shows budget usage and remaining spend at %s%% usage", (spend, color) => {
    renderWithProviders(<CustomerBudgetSpend spend={spend} limit={100} label="Monthly spend / limit" />);

    const progress = screen.getByRole("progressbar", { name: "Monthly spend / limit" });
    expect(progress).toHaveAttribute("aria-valuenow", String(spend));
    expect(within(progress).getByRole("generic")).toHaveStyle({
      backgroundColor: color,
      width: `${spend}%`,
    });
    expect(screen.getByText(`${spend}% used`)).toBeInTheDocument();
    expect(screen.getByText(`$${(100 - spend).toFixed(2)} left`)).toBeInTheDocument();
  });

  it("shows the actual percentage over budget while capping the bar", () => {
    renderWithProviders(<CustomerBudgetSpend spend={120} limit={100} label="Monthly spend / limit" />);

    expect(screen.getByText("120% used")).toBeInTheDocument();
    expect(screen.getByText("$20.00 over")).toBeInTheDocument();
    expect(screen.getByRole("progressbar")).toHaveAttribute("aria-valuenow", "100");
  });

  it("identifies a zero budget without inventing a percentage", () => {
    renderWithProviders(<CustomerBudgetSpend spend={0} limit={0} label="Monthly spend / limit" />);

    expect(screen.getByText("Percentage unavailable for a $0 budget")).toBeInTheDocument();
    expect(screen.getByText("$0.00 left")).toBeInTheDocument();
    expect(screen.getByRole("progressbar")).toHaveAttribute("aria-valuenow", "100");
  });

  it.each([
    { spend: null, limit: 100, label: "Unavailable / $100.00" },
    { spend: 10, limit: null, label: "No limit" },
  ])("does not show a percentage without known spend and a limit", ({ spend, limit, label }) => {
    renderWithProviders(<CustomerBudgetSpend spend={spend} limit={limit} label="Monthly spend / limit" />);

    expect(screen.getByText(label)).toBeInTheDocument();
    expect(screen.queryByRole("progressbar")).not.toBeInTheDocument();
    expect(screen.queryByText(/% used$/)).not.toBeInTheDocument();
  });
});
