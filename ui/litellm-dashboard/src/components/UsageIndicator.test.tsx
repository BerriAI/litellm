import React from "react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import UsageIndicator from "./UsageIndicator";

vi.mock("./networking", () => ({
  getRemainingUsers: vi.fn(),
  getLicenseInfo: vi.fn().mockResolvedValue(null),
}));

vi.mock("@/app/(dashboard)/hooks/useDisableUsageIndicator", () => ({
  useDisableUsageIndicator: vi.fn(() => false),
}));

import { getRemainingUsers } from "./networking";

const mockGetRemainingUsers = vi.mocked(getRemainingUsers);

const DEFAULT_USAGE_DATA = {
  total_users: 100,
  total_users_used: 1,
  total_users_remaining: 99,
  total_teams: null,
  total_teams_used: 0,
  total_teams_remaining: null,
};

describe("UsageIndicator", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockGetRemainingUsers.mockResolvedValue(DEFAULT_USAGE_DATA);
  });

  it("should render when given access token and usage data loads", async () => {
    render(<UsageIndicator accessToken="token" width={220} />);

    await screen.findByText("Usage");

    expect(screen.getByText("Usage")).toBeInTheDocument();
  });

  it("should not show Near limit when users usage is below 80% (1/100 -> 1%)", async () => {
    render(<UsageIndicator accessToken="token" width={220} />);

    await screen.findByText("Usage");

    expect(screen.queryByText("Near limit")).not.toBeInTheDocument();
  });

  it("should render nothing when both total_users and total_teams are null", async () => {
    mockGetRemainingUsers.mockResolvedValue({
      total_users: null,
      total_teams: null,
      total_users_used: 520,
      total_teams_used: 4,
      total_teams_remaining: null,
      total_users_remaining: null,
    });

    render(<UsageIndicator accessToken="token" width={220} />);

    await waitFor(() => {
      expect(screen.queryByText("Usage")).not.toBeInTheDocument();
      expect(screen.queryByText("Loading...")).not.toBeInTheDocument();
    });
  });

  it("should show Near limit for Teams when at 80% usage (4/5)", async () => {
    mockGetRemainingUsers.mockResolvedValue({
      total_users: null,
      total_users_used: 0,
      total_users_remaining: null,
      total_teams: 5,
      total_teams_used: 4,
      total_teams_remaining: 1,
    });

    render(<UsageIndicator accessToken="token" width={220} />);

    await screen.findByText("Usage");

    expect(screen.getByText("Teams")).toBeInTheDocument();
    expect(screen.getByText("Near limit")).toBeInTheDocument();
  });

  it("should show Over limit for Users when usage exceeds 100% (105/100)", async () => {
    mockGetRemainingUsers.mockResolvedValue({
      total_users: 100,
      total_users_used: 105,
      total_users_remaining: -5,
      total_teams: null,
      total_teams_used: 0,
      total_teams_remaining: null,
    });

    render(<UsageIndicator accessToken="token" width={220} />);

    await screen.findByText("Usage");

    expect(screen.getByText("Users")).toBeInTheDocument();
    expect(screen.getByText("Over limit")).toBeInTheDocument();
  });

  it("should show Over limit for Teams when usage exceeds 100%", async () => {
    mockGetRemainingUsers.mockResolvedValue({
      total_users: null,
      total_users_used: 0,
      total_users_remaining: null,
      total_teams: 10,
      total_teams_used: 12,
      total_teams_remaining: -2,
    });

    render(<UsageIndicator accessToken="token" width={220} />);

    await screen.findByText("Usage");

    expect(screen.getByText("Teams")).toBeInTheDocument();
    expect(screen.getByText("Over limit")).toBeInTheDocument();
  });

  it("should render nothing when accessToken is null", () => {
    render(<UsageIndicator accessToken={null} width={220} />);

    expect(mockGetRemainingUsers).not.toHaveBeenCalled();
    expect(screen.queryByText("Usage")).not.toBeInTheDocument();
  });

  it("should render nothing when disableUsageIndicator is true", async () => {
    const { useDisableUsageIndicator } = await import("@/app/(dashboard)/hooks/useDisableUsageIndicator");
    (useDisableUsageIndicator as ReturnType<typeof vi.fn>).mockReturnValue(true);

    render(<UsageIndicator accessToken="token" width={220} />);

    await waitFor(() => {
      expect(screen.queryByText("Usage")).not.toBeInTheDocument();
    });

    (useDisableUsageIndicator as ReturnType<typeof vi.fn>).mockReturnValue(false);
  });

  it("should show Loading while fetching", () => {
    mockGetRemainingUsers.mockImplementation(() => new Promise(() => {}));

    render(<UsageIndicator accessToken="token" width={220} />);

    expect(screen.getByText("Loading...")).toBeInTheDocument();
  });

  it("should show error message when fetch fails", async () => {
    const consoleSpy = vi.spyOn(console, "error").mockImplementation(() => {});
    mockGetRemainingUsers.mockRejectedValue(new Error("Network error"));

    render(<UsageIndicator accessToken="token" width={220} />);

    expect(await screen.findByText("Failed to load usage data")).toBeInTheDocument();

    consoleSpy.mockRestore();
  });

  it("should minimize when user clicks minimize button", async () => {
    const user = userEvent.setup();
    render(<UsageIndicator accessToken="token" width={220} />);

    await screen.findByText("Usage");

    const minimizeButton = screen.getByTitle("Minimize");
    await user.click(minimizeButton);

    expect(screen.queryByText("Users")).not.toBeInTheDocument();
    expect(screen.getByTitle("Show usage details")).toBeInTheDocument();
  });

  it("should restore from minimized when user clicks restore button", async () => {
    const user = userEvent.setup();
    render(<UsageIndicator accessToken="token" width={220} />);

    await screen.findByText("Usage");

    await user.click(screen.getByTitle("Minimize"));
    await user.click(screen.getByTitle("Show usage details"));

    expect(screen.getByText("Usage")).toBeInTheDocument();
    expect(screen.getByText("Users")).toBeInTheDocument();
  });
});
