import { render, waitFor, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

const mockGetGeneralSettingsCall = vi.fn();

vi.mock("@/components/networking", () => ({
  getGeneralSettingsCall: (...args: unknown[]) => mockGetGeneralSettingsCall(...args),
}));

vi.mock("@/app/(dashboard)/router-settings/_components/general_settings", async (importOriginal) => {
  const original =
    await importOriginal<typeof import("@/app/(dashboard)/router-settings/_components/general_settings")>();
  return {
    ...original,
    PromptCachingPanel: () => <div data-testid="caching-settings" />,
  };
});

const mockCacheLeakageCard = vi.fn();
const mockPromptCachingTestCard = vi.fn();

vi.mock("./CacheLeakageCard", () => ({
  __esModule: true,
  default: (props: unknown) => {
    mockCacheLeakageCard(props);
    return <div data-testid="cache-leakage-card" />;
  },
}));

vi.mock("./PromptCachingTestCard", () => ({
  __esModule: true,
  default: (props: unknown) => {
    mockPromptCachingTestCard(props);
    return <div data-testid="prompt-caching-test-card" />;
  },
}));

import PromptCachingTab from "./PromptCachingTab";

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

describe("PromptCachingTab", () => {
  it("renders the cache leakage table alongside the caching settings", async () => {
    mockGetGeneralSettingsCall.mockResolvedValue([]);

    render(<PromptCachingTab accessToken="test-token" activity={activity} />);

    expect(screen.getByTestId("caching-settings")).toBeInTheDocument();
    expect(screen.getByTestId("cache-leakage-card")).toBeInTheDocument();
    await waitFor(() => expect(mockCacheLeakageCard).toHaveBeenCalledWith(expect.objectContaining({ activity })));
  });

  it("passes enabled=true to the test card when the caching setting is on", async () => {
    mockGetGeneralSettingsCall.mockResolvedValue([
      {
        field_name: "enable_anthropic_prompt_caching",
        field_type: "Boolean",
        field_value: true,
        field_description: "",
        stored_in_db: true,
      },
    ]);

    render(<PromptCachingTab accessToken="test-token" activity={activity} />);

    await waitFor(() =>
      expect(mockPromptCachingTestCard).toHaveBeenCalledWith(expect.objectContaining({ enabled: true })),
    );
  });

  it("passes enabled=false to the test card when the caching setting is off", async () => {
    mockGetGeneralSettingsCall.mockResolvedValue([
      {
        field_name: "enable_anthropic_prompt_caching",
        field_type: "Boolean",
        field_value: false,
        field_description: "",
        stored_in_db: true,
      },
    ]);

    render(<PromptCachingTab accessToken="test-token" activity={activity} />);

    await waitFor(() =>
      expect(mockPromptCachingTestCard).toHaveBeenCalledWith(expect.objectContaining({ enabled: false })),
    );
  });
});
