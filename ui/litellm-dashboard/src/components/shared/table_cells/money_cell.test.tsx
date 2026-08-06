import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { MoneyCell } from "./money_cell";

describe("MoneyCell", () => {
  it("renders '-' for null and undefined", () => {
    const { rerender } = render(<MoneyCell value={null} />);
    expect(screen.getByText("-")).toBeInTheDocument();
    rerender(<MoneyCell value={undefined} />);
    expect(screen.getByText("-")).toBeInTheDocument();
  });

  it("renders the custom emptyText for null budgets", () => {
    render(<MoneyCell value={null} emptyText="Unlimited" />);
    expect(screen.getByText("Unlimited")).toBeInTheDocument();
  });

  it("renders '-' for zero by default", () => {
    render(<MoneyCell value={0} />);
    expect(screen.getByText("-")).toBeInTheDocument();
  });

  it("renders a formatted zero when showZero is set, never the emptyText", () => {
    render(<MoneyCell value={0} showZero emptyText="Unlimited" decimals={2} />);
    expect(screen.getByText("$0.00")).toBeInTheDocument();
    expect(screen.queryByText("Unlimited")).not.toBeInTheDocument();
  });

  it("formats amounts with commas, a dollar sign and the given decimals", () => {
    render(<MoneyCell value={1234.5678} decimals={2} />);
    expect(screen.getByText("$1,234.57")).toBeInTheDocument();
  });

  it("defaults to 4 decimals", () => {
    render(<MoneyCell value={42} />);
    expect(screen.getByText("$42.0000")).toBeInTheDocument();
  });

  it("renders the sub-threshold form for amounts that round to zero", () => {
    render(<MoneyCell value={0.0000001} decimals={6} />);
    expect(screen.getByText("< $0.000001")).toBeInTheDocument();
  });
});
