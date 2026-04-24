import { render, screen } from "@testing-library/react";
import React from "react";
import { describe, expect, it, vi } from "vitest";
import TeamsHeaderTabs from "./TeamsHeaderTabs";

const renderTabs = (props: Partial<Parameters<typeof TeamsHeaderTabs>[0]> = {}) => {
  const defaults = {
    lastRefreshed: "",
    onRefresh: vi.fn(),
    userRole: "Internal User",
    yourTeamsPanel: <div data-testid="your-teams-panel">Your</div>,
    availableTeamsPanel: <div data-testid="available-teams-panel">Available</div>,
    defaultTeamSettingsPanel: <div data-testid="default-settings-panel">Settings</div>,
  };
  return render(<TeamsHeaderTabs {...defaults} {...props} />);
};

describe("TeamsHeaderTabs", () => {
  it("should render 'Your Teams' and 'Available Teams' tabs", () => {
    renderTabs();

    expect(screen.getByRole("tab", { name: "Your Teams" })).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: "Available Teams" })).toBeInTheDocument();
  });

  it("should render 'Default Team Settings' tab when user is Admin", () => {
    renderTabs({ userRole: "Admin" });

    expect(
      screen.getByRole("tab", { name: "Default Team Settings" }),
    ).toBeInTheDocument();
  });

  it("should not render 'Default Team Settings' tab for non-admin users", () => {
    renderTabs({ userRole: "Internal User" });

    expect(
      screen.queryByRole("tab", { name: "Default Team Settings" }),
    ).not.toBeInTheDocument();
  });

  it("should display last refreshed time when provided", () => {
    renderTabs({ lastRefreshed: "2024-06-01 12:00:00" });

    expect(screen.getByText("Last Refreshed: 2024-06-01 12:00:00")).toBeInTheDocument();
  });
});
