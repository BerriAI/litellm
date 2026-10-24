import { screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { renderWithProviders } from "@/../tests/test-utils";
import UsagePage from "./page";

const settings = vi.hoisted(() => vi.fn());
vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({ default: () => ({}) }));
vi.mock("@/app/(dashboard)/hooks/teams/useTeams", () => ({ useTeams: () => ({ data: [] }) }));
vi.mock("@/app/(dashboard)/hooks/organizations/useOrganizations", () => ({ useOrganizations: () => ({ data: [] }) }));
vi.mock("@/app/(dashboard)/hooks/uiSettings/useUISettings", () => ({ useUISettings: settings }));
vi.mock("./_components/components/UsagePageView", () => ({
  default: ({ defaultTimeRange }: { defaultTimeRange?: string }) => (
    <div>Usage range: {defaultTimeRange ?? "last_7_days"}</div>
  ),
}));

describe("Usage settings readiness", () => {
  beforeEach(() => settings.mockReturnValue({ isLoading: true }));

  it("waits for the configured range before mounting the usage data consumers", () => {
    const { rerender } = renderWithProviders(<UsagePage />);
    expect(screen.getByRole("status", { name: "Loading usage settings" })).toBeInTheDocument();
    expect(screen.queryByText(/Usage range/)).not.toBeInTheDocument();
    settings.mockReturnValue({ isLoading: false, data: { values: { usage_default_time_range: "month_to_date" } } });
    rerender(<UsagePage />);
    expect(screen.getByText("Usage range: month_to_date")).toBeInTheDocument();
  });

  it.each(["?range=MTD", "?from=2026-01-01&to=2026-01-15"])(
    "does not delay an explicit URL range %s",
    (searchParams) => {
      renderWithProviders(<UsagePage />, { searchParams });
      expect(screen.queryByRole("status")).not.toBeInTheDocument();
      expect(screen.getByText(/Usage range/)).toBeInTheDocument();
    },
  );

  it.each(["?range=unknown", "?from=2026-02-30&to=2026-03-01"])(
    "waits for settings when the URL range is invalid %s",
    (searchParams) => {
      renderWithProviders(<UsagePage />, { searchParams });
      expect(screen.getByRole("status", { name: "Loading usage settings" })).toBeInTheDocument();
    },
  );

  it("uses the existing fallback if settings cannot be fetched", () => {
    settings.mockReturnValue({ isLoading: false, isError: true });
    renderWithProviders(<UsagePage />);
    expect(screen.getByText("Usage range: last_7_days")).toBeInTheDocument();
  });
});
