import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import React from "react";
import ViewUserSpend from "./view_user_spend";

vi.mock("./networking", () => ({
  modelAvailableCall: vi.fn().mockResolvedValue({ data: [] }),
}));

vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  default: () => ({ accessToken: "tok", userRole: "Admin", userId: "admin-id" }),
}));

describe("ViewUserSpend — Max Budget tile", () => {
  beforeEach(() => vi.clearAllMocks());

  it("shows a finite cap with its reset period when budgetDuration is set", () => {
    render(<ViewUserSpend userSpend={10} userMaxBudget={600} selectedTeam={null} budgetDuration="30d" />);
    expect(screen.getByText(/\$600\.0000 limit/)).toBeInTheDocument();
    expect(screen.getByText(/over monthly/)).toBeInTheDocument();
  });

  it('shows "No limit" for an unlimited (null) budget and no period', () => {
    render(<ViewUserSpend userSpend={10} userMaxBudget={null} selectedTeam={null} budgetDuration="30d" />);
    expect(screen.getByText("No limit")).toBeInTheDocument();
    expect(screen.queryByText(/over/)).not.toBeInTheDocument();
  });

  it("shows a finite cap with no period when budgetDuration is absent", () => {
    render(<ViewUserSpend userSpend={10} userMaxBudget={300} selectedTeam={null} />);
    expect(screen.getByText(/\$300\.0000 limit/)).toBeInTheDocument();
    expect(screen.queryByText(/over/)).not.toBeInTheDocument();
  });

  it('does not render "No limit" while the budget is loading', () => {
    render(<ViewUserSpend userSpend={10} userMaxBudget={null} selectedTeam={null} budgetLoading={true} />);
    expect(screen.queryByText("No limit")).not.toBeInTheDocument();
    expect(screen.getByText("—")).toBeInTheDocument();
  });
});
