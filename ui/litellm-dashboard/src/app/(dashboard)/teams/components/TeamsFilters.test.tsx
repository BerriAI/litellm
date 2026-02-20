import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import React from "react";
import { describe, expect, it, vi } from "vitest";
import { Organization } from "@/components/networking";
import TeamsFilters from "./TeamsFilters";

type FilterState = {
  team_id: string;
  team_alias: string;
  organization_id: string;
  sort_by: string;
  sort_order: "asc" | "desc";
};

const emptyFilters: FilterState = {
  team_alias: "",
  team_id: "",
  organization_id: "",
  sort_by: "",
  sort_order: "asc",
};

const mockOrganizations: Organization[] = [
  { organization_id: "org-1", organization_alias: "Acme Corp" } as Organization,
  { organization_id: "org-2", organization_alias: "Globex" } as Organization,
];

const renderFilters = (overrides: Partial<Parameters<typeof TeamsFilters>[0]> = {}) => {
  const defaults = {
    filters: emptyFilters,
    organizations: mockOrganizations,
    showFilters: false,
    onToggleFilters: vi.fn(),
    onChange: vi.fn(),
    onReset: vi.fn(),
  };
  return render(<TeamsFilters {...defaults} {...overrides} />);
};

describe("TeamsFilters", () => {
  it("renders the team name search input, Filters button, and Reset Filters button", () => {
    renderFilters();

    expect(screen.getByPlaceholderText("Search by Team Name...")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /^filters$/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /reset filters/i })).toBeInTheDocument();
  });

  it("reflects the current team_alias filter value in the search input", () => {
    renderFilters({ filters: { ...emptyFilters, team_alias: "Platform" } });

    expect(screen.getByPlaceholderText("Search by Team Name...")).toHaveValue("Platform");
  });

  it("calls onChange with 'team_alias' key when the search input changes", async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    renderFilters({ onChange });

    await user.type(screen.getByPlaceholderText("Search by Team Name..."), "Dev");

    expect(onChange).toHaveBeenCalledWith("team_alias", expect.stringContaining("D"));
  });

  it("calls onToggleFilters with the inverted boolean when the Filters button is clicked", async () => {
    const user = userEvent.setup();
    const onToggleFilters = vi.fn();
    renderFilters({ showFilters: false, onToggleFilters });

    await user.click(screen.getByRole("button", { name: /^filters$/i }));

    expect(onToggleFilters).toHaveBeenCalledWith(true);
  });

  it("calls onToggleFilters(false) when filters are currently expanded", async () => {
    const user = userEvent.setup();
    const onToggleFilters = vi.fn();
    renderFilters({ showFilters: true, onToggleFilters });

    await user.click(screen.getByRole("button", { name: /^filters$/i }));

    expect(onToggleFilters).toHaveBeenCalledWith(false);
  });

  it("calls onReset when the Reset Filters button is clicked", async () => {
    const user = userEvent.setup();
    const onReset = vi.fn();
    renderFilters({ onReset });

    await user.click(screen.getByRole("button", { name: /reset filters/i }));

    expect(onReset).toHaveBeenCalledTimes(1);
  });

  it("does not show the Team ID input when showFilters is false", () => {
    renderFilters({ showFilters: false });

    expect(screen.queryByPlaceholderText("Enter Team ID")).not.toBeInTheDocument();
  });

  it("shows the Team ID input when showFilters is true", () => {
    renderFilters({ showFilters: true });

    expect(screen.getByPlaceholderText("Enter Team ID")).toBeInTheDocument();
  });

  it("calls onChange with 'team_id' key when the Team ID input changes", async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    renderFilters({ showFilters: true, onChange });

    await user.type(screen.getByPlaceholderText("Enter Team ID"), "abc");

    expect(onChange).toHaveBeenCalledWith("team_id", expect.stringContaining("a"));
  });

  it("reflects the current team_id filter value in the Team ID input", () => {
    renderFilters({ showFilters: true, filters: { ...emptyFilters, team_id: "team-xyz" } });

    expect(screen.getByPlaceholderText("Enter Team ID")).toHaveValue("team-xyz");
  });

  it("shows a blue dot indicator on the Filters button when team_alias filter is active", () => {
    renderFilters({ filters: { ...emptyFilters, team_alias: "Platform" } });

    const filtersButton = screen.getByRole("button", { name: /^filters$/i });
    expect(filtersButton.querySelector(".bg-blue-500")).toBeInTheDocument();
  });

  it("shows a blue dot indicator on the Filters button when team_id filter is active", () => {
    renderFilters({ filters: { ...emptyFilters, team_id: "team-123" } });

    const filtersButton = screen.getByRole("button", { name: /^filters$/i });
    expect(filtersButton.querySelector(".bg-blue-500")).toBeInTheDocument();
  });

  it("shows a blue dot indicator on the Filters button when organization_id filter is active", () => {
    renderFilters({ filters: { ...emptyFilters, organization_id: "org-1" } });

    const filtersButton = screen.getByRole("button", { name: /^filters$/i });
    expect(filtersButton.querySelector(".bg-blue-500")).toBeInTheDocument();
  });

  it("does not show the blue dot indicator when all filters are empty", () => {
    renderFilters({ filters: emptyFilters });

    const filtersButton = screen.getByRole("button", { name: /^filters$/i });
    expect(filtersButton.querySelector(".bg-blue-500")).not.toBeInTheDocument();
  });
});
