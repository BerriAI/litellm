import { render, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

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

const activityFixture = () => ({
  dateValue: {},
  onDateChange: vi.fn(),
  results: [],
  loading: false,
  isFetchingMore: false,
});

describe("PromptCachingTab", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("renders the cache leakage table alongside the caching settings", async () => {
    mockGetGeneralSettingsCall.mockResolvedValue([]);

    const activity = activityFixture();
    const { getByTestId } = render(
      <PromptCachingTab accessToken="test-token" activity={activity} canViewProxyConfig />,
    );

    expect(getByTestId("caching-settings")).toBeInTheDocument();
    expect(getByTestId("cache-leakage-card")).toBeInTheDocument();
    await waitFor(() => expect(mockCacheLeakageCard).toHaveBeenCalledWith(expect.objectContaining({ activity })));
  });

  it("keeps the cache leakage table but skips /config/list when the caller cannot read proxy config", async () => {
    mockGetGeneralSettingsCall.mockResolvedValue([]);

    const { getByTestId, queryByTestId } = render(
      <PromptCachingTab accessToken="test-token" activity={activityFixture()} canViewProxyConfig={false} />,
    );

    expect(queryByTestId("caching-settings")).not.toBeInTheDocument();
    expect(getByTestId("cache-leakage-card")).toBeInTheDocument();
    await waitFor(() => expect(mockCacheLeakageCard).toHaveBeenCalled());
    expect(mockGetGeneralSettingsCall).not.toHaveBeenCalled();
  });
});
