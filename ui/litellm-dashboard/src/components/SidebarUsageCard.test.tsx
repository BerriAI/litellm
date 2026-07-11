import React from "react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import SidebarUsageCard from "./SidebarUsageCard";
import type { LicenseInfo } from "./networking";

vi.mock("./networking", () => ({ getRemainingUsers: vi.fn() }));

vi.mock("@/app/(dashboard)/hooks/useDisableUsageIndicator", () => ({
  useDisableUsageIndicator: vi.fn(() => false),
}));

vi.mock("@/app/(dashboard)/hooks/license/useLicenseInfo", () => ({
  useLicenseInfo: vi.fn(),
}));

import { getRemainingUsers } from "./networking";
import { useLicenseInfo } from "@/app/(dashboard)/hooks/license/useLicenseInfo";

const mockGetRemainingUsers = vi.mocked(getRemainingUsers);
const mockUseLicenseInfo = vi.mocked(useLicenseInfo);

const licenseResult = (data: LicenseInfo | null) => ({ data }) as unknown as ReturnType<typeof useLicenseInfo>;
const ACTIVE_LICENSE: LicenseInfo = {
  has_license: true,
  license_type: null,
  expiration_date: null,
  allowed_features: [],
  limits: { max_users: null, max_teams: null },
};

const renderWithClient = (ui: React.ReactElement) => {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(<QueryClientProvider client={queryClient}>{ui}</QueryClientProvider>);
};

const SEATS_DATA = {
  total_users: 100,
  total_users_used: 20,
  total_users_remaining: 80,
  total_teams: null,
  total_teams_used: 0,
  total_teams_remaining: null,
};

const OVER_LIMIT_DATA = {
  total_users: 100,
  total_users_used: 130,
  total_users_remaining: -30,
  total_teams: null,
  total_teams_used: 0,
  total_teams_remaining: null,
};

const NO_LIMITS_DATA = {
  total_users: null,
  total_users_used: 186,
  total_users_remaining: null,
  total_teams: null,
  total_teams_used: 125,
  total_teams_remaining: null,
};

describe("SidebarUsageCard", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockGetRemainingUsers.mockResolvedValue(SEATS_DATA);
    mockUseLicenseInfo.mockReturnValue(licenseResult(ACTIVE_LICENSE));
  });

  it("renders an expanded seat meter reporting value and range when data loads", async () => {
    const { container } = renderWithClient(
      <SidebarUsageCard accessToken="token" collapsed={false} onExpandRail={() => {}} />,
    );

    await screen.findByText("Enterprise usage");
    const meter = await screen.findByRole("meter");
    expect(meter).toHaveAttribute("aria-valuenow", "20");
    expect(meter).toHaveAttribute("aria-valuemax", "100");
    expect(screen.getByText("Seats")).toBeInTheDocument();

    const indicator = container.querySelector('[data-slot="meter-indicator"]');
    expect(indicator).toHaveStyle({ width: "20%" });
  });

  it("collapses the meter panel when the trigger is toggled", async () => {
    const user = userEvent.setup();
    renderWithClient(<SidebarUsageCard accessToken="token" collapsed={false} onExpandRail={() => {}} />);

    await screen.findByRole("meter");
    await user.click(screen.getByRole("button", { name: /Enterprise usage/i }));

    await waitFor(() => expect(screen.queryByRole("meter")).not.toBeInTheDocument());
  });

  it("flags over-limit usage with a destructive, capped indicator", async () => {
    mockGetRemainingUsers.mockResolvedValue(OVER_LIMIT_DATA);

    const { container } = renderWithClient(
      <SidebarUsageCard accessToken="token" collapsed={false} onExpandRail={() => {}} />,
    );

    await screen.findByRole("meter");
    const indicator = container.querySelector('[data-slot="meter-indicator"]');
    expect(indicator).toHaveClass("bg-destructive");
    expect(indicator).toHaveStyle({ width: "100%" });
  });

  it("renders nothing when neither seat nor team limits are set", async () => {
    mockGetRemainingUsers.mockResolvedValue(NO_LIMITS_DATA);

    renderWithClient(<SidebarUsageCard accessToken="token" collapsed={false} onExpandRail={() => {}} />);

    await waitFor(() => expect(screen.queryByText("Enterprise usage")).not.toBeInTheDocument());
  });

  it("renders nothing without an enterprise license even when seat limits exist", async () => {
    mockUseLicenseInfo.mockReturnValue(licenseResult(null));

    const { container } = renderWithClient(
      <SidebarUsageCard accessToken="token" collapsed={false} onExpandRail={() => {}} />,
    );

    await waitFor(() => expect(mockGetRemainingUsers).toHaveBeenCalled());
    expect(screen.queryByText("Enterprise usage")).not.toBeInTheDocument();
    expect(container.querySelector('[data-slot="meter"]')).toBeNull();
  });

  it("shows a collapsed rail button that expands the sidebar", async () => {
    const onExpandRail = vi.fn();
    const user = userEvent.setup();
    renderWithClient(<SidebarUsageCard accessToken="token" collapsed onExpandRail={onExpandRail} />);

    const rail = await screen.findByTitle("Enterprise usage");
    await user.click(rail);
    expect(onExpandRail).toHaveBeenCalledOnce();
  });
});
