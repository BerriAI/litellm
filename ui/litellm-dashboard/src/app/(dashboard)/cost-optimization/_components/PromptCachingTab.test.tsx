import { render, waitFor, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

const mockGetGeneralSettingsCall = vi.fn();

vi.mock("@/components/networking", () => ({
  getGeneralSettingsCall: (...args: unknown[]) => mockGetGeneralSettingsCall(...args),
}));

vi.mock("@/app/(dashboard)/router-settings/_components/general_settings", () => ({
  PromptCachingPanel: () => <div data-testid="caching-settings" />,
}));

const mockCacheLeakageCard = vi.fn();

vi.mock("./CacheLeakageCard", () => ({
  __esModule: true,
  default: (props: unknown) => {
    mockCacheLeakageCard(props);
    return <div data-testid="cache-leakage-card" />;
  },
}));

import PromptCachingTab from "./PromptCachingTab";

describe("PromptCachingTab", () => {
  it("renders the cache leakage table alongside the caching settings", async () => {
    mockGetGeneralSettingsCall.mockResolvedValue([]);

    const activity = {
      dateValue: {},
      onDateChange: vi.fn(),
      results: [],
      loading: false,
      isFetchingMore: false,
      progress: { currentPage: 1, totalPages: 1 },
      cancelled: false,
      cancel: vi.fn(),
    };
    render(<PromptCachingTab accessToken="test-token" activity={activity} />);

    expect(screen.getByTestId("caching-settings")).toBeInTheDocument();
    expect(screen.getByTestId("cache-leakage-card")).toBeInTheDocument();
    await waitFor(() => expect(mockCacheLeakageCard).toHaveBeenCalledWith(expect.objectContaining({ activity })));
  });
});
