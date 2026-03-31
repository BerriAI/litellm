import { render, screen } from "@testing-library/react";
import React from "react";
import { describe, expect, it, vi } from "vitest";
import TeamsHeaderTabs from "./TeamsHeaderTabs";

vi.mock("@tremor/react", () => ({
  TabGroup: ({ children, ...props }: any) => <div data-testid="tab-group" {...props}>{children}</div>,
  TabList: ({ children, ...props }: any) => <div data-testid="tab-list" {...props}>{children}</div>,
  Tab: ({ children, ...props }: any) => <button {...props}>{children}</button>,
  TabPanels: ({ children, ...props }: any) => <div data-testid="tab-panels" {...props}>{children}</div>,
  Text: ({ children, ...props }: any) => <span {...props}>{children}</span>,
  Icon: ({ onClick, ...props }: any) => <button data-testid="refresh-icon" onClick={onClick} />,
}));

vi.mock("@heroicons/react/outline", () => ({
  RefreshIcon: () => <svg data-testid="refresh-svg" />,
}));

const renderTabs = (props: Partial<Parameters<typeof TeamsHeaderTabs>[0]> = {}) => {
  const defaults = {
    lastRefreshed: "",
    onRefresh: vi.fn(),
    userRole: "Internal User",
    children: <div data-testid="panel-content">Panel</div>,
  };
  return render(<TeamsHeaderTabs {...defaults} {...props} />);
};

describe("TeamsHeaderTabs", () => {
  it("should render 'Your Teams' and 'Available Teams' tabs", () => {
    renderTabs();

    expect(screen.getByText("Your Teams")).toBeInTheDocument();
    expect(screen.getByText("Available Teams")).toBeInTheDocument();
  });

  it("should render 'Default Team Settings' tab when user is Admin", () => {
    renderTabs({ userRole: "Admin" });

    expect(screen.getByText("Default Team Settings")).toBeInTheDocument();
  });

  it("should not render 'Default Team Settings' tab for non-admin users", () => {
    renderTabs({ userRole: "Internal User" });

    expect(screen.queryByText("Default Team Settings")).not.toBeInTheDocument();
  });

  it("should display last refreshed time when provided", () => {
    renderTabs({ lastRefreshed: "2024-06-01 12:00:00" });

    expect(screen.getByText("Last Refreshed: 2024-06-01 12:00:00")).toBeInTheDocument();
  });
});
