import { fireEvent, render, waitFor, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

const mockGetGeneralSettingsCall = vi.fn();

vi.mock("@/components/networking", () => ({
  getGeneralSettingsCall: (...args: unknown[]) => mockGetGeneralSettingsCall(...args),
}));

vi.mock("@/app/(dashboard)/router-settings/_components/general_settings", () => ({
  PromptCachingPanel: () => <div data-testid="caching-settings" />,
}));

const mockCacheLeakageCard = vi.fn();
const mockRequestsTable = vi.fn();
const nextDateRange = { from: new Date(2026, 8, 1), to: new Date(2026, 8, 2) };

vi.mock("./PromptCachingRequestsTable", () => ({
  default: (props: unknown) => {
    mockRequestsTable(props);
    return <div data-testid="caching-requests" />;
  },
}));

vi.mock("@/components/shared/advanced_date_picker", () => ({
  default: ({ onValueChange }: { onValueChange: (range: typeof nextDateRange) => void }) => (
    <button onClick={() => onValueChange(nextDateRange)}>Change caching dates</button>
  ),
}));

vi.mock("./CacheLeakageCard", () => ({
  __esModule: true,
  default: (props: unknown) => {
    mockCacheLeakageCard(props);
    return <div data-testid="cache-leakage-card" />;
  },
}));

import PromptCachingTab from "./PromptCachingTab";

describe("PromptCachingTab", () => {
  it("shares the selected dates between requests and cache leakage alongside caching settings", async () => {
    mockGetGeneralSettingsCall.mockResolvedValue([]);

    const activity = {
      dateValue: {},
      onDateChange: vi.fn(),
      results: [],
      loading: false,
      isFetchingMore: false,
      progress: { currentPage: 1, totalPages: 1 },
      cancelled: false,
      failed: false,
      cancel: vi.fn(),
    };
    render(<PromptCachingTab accessToken="test-token" activity={activity} />);

    expect(screen.getByTestId("caching-settings")).toBeInTheDocument();
    expect(screen.getByTestId("cache-leakage-card")).toBeInTheDocument();
    expect(screen.getByTestId("caching-requests")).toBeInTheDocument();
    expect(mockRequestsTable).toHaveBeenCalledWith({ accessToken: "test-token", dateValue: activity.dateValue });
    fireEvent.click(screen.getByRole("button", { name: "Change caching dates" }));
    expect(activity.onDateChange).toHaveBeenCalledWith(nextDateRange);
    await waitFor(() => expect(mockCacheLeakageCard).toHaveBeenCalledWith(expect.objectContaining({ activity })));
  });
});
