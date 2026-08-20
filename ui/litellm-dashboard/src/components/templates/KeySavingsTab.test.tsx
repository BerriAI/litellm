import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import KeySavingsTab from "./KeySavingsTab";
import * as useScopedDailyActivityRangeModule from "@/app/(dashboard)/cost-optimization/_components/useDailyActivityRange";

const mockActivity = (overrides = {}) => ({
  dateValue: { from: new Date("2025-01-01"), to: new Date("2025-01-31") },
  onDateChange: vi.fn(),
  results: [],
  loading: false,
  isFetchingMore: false,
  ...overrides,
});

describe("KeySavingsTab", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("renders without crashing", () => {
    vi.spyOn(useScopedDailyActivityRangeModule, "useScopedDailyActivityRange").mockReturnValue(mockActivity());

    render(
      <KeySavingsTab
        accessToken="test-token"
        keyToken="key-abc123"
        userId="user-123"
        userRole="user"
      />
    );

    expect(screen.getByText("Total saved")).toBeInTheDocument();
    expect(screen.getByText("Compression savings")).toBeInTheDocument();
    expect(screen.getByText("Prompt caching savings")).toBeInTheDocument();
    expect(screen.getByText("Cache hit rate")).toBeInTheDocument();
  });

  it("shows empty state when no results in range", () => {
    vi.spyOn(useScopedDailyActivityRangeModule, "useScopedDailyActivityRange").mockReturnValue(mockActivity());

    render(
      <KeySavingsTab
        accessToken="test-token"
        keyToken="key-abc123"
        userId="user-123"
        userRole="user"
      />
    );

    expect(screen.getByTestId("key-savings-empty")).toHaveTextContent("No usage recorded for this key");
  });

  it("shows loading state", () => {
    vi.spyOn(useScopedDailyActivityRangeModule, "useScopedDailyActivityRange").mockReturnValue(
      mockActivity({ loading: true })
    );

    render(
      <KeySavingsTab
        accessToken="test-token"
        keyToken="key-abc123"
        userId="user-123"
        userRole="user"
      />
    );

    expect(screen.getByTestId("key-savings-empty")).toHaveTextContent("Loading savings");
    expect(screen.getByText("Loading...")).toBeInTheDocument();
  });

  it("passes key token as apiKey to scoped hook", () => {
    const mockUseScopedDailyActivityRange = vi.spyOn(
      useScopedDailyActivityRangeModule,
      "useScopedDailyActivityRange"
    );
    mockUseScopedDailyActivityRange.mockReturnValue(mockActivity());

    render(
      <KeySavingsTab
        accessToken="test-token"
        keyToken="key-abc123"
        userId="user-456"
        userRole="user"
      />
    );

    expect(mockUseScopedDailyActivityRange).toHaveBeenCalledWith("test-token", {
      userId: "user-456",
      apiKey: "key-abc123",
    });
  });

  it("passes null userId for admin viewers", () => {
    const mockUseScopedDailyActivityRange = vi.spyOn(
      useScopedDailyActivityRangeModule,
      "useScopedDailyActivityRange"
    );
    mockUseScopedDailyActivityRange.mockReturnValue(mockActivity());

    render(
      <KeySavingsTab
        accessToken="test-token"
        keyToken="key-abc123"
        userId="admin-123"
        userRole="proxy_admin"
      />
    );

    expect(mockUseScopedDailyActivityRange).toHaveBeenCalledWith("test-token", {
      userId: null,
      apiKey: "key-abc123",
    });
  });

  it("shows scope note for non-admin viewers", () => {
    vi.spyOn(useScopedDailyActivityRangeModule, "useScopedDailyActivityRange").mockReturnValue(mockActivity());

    render(
      <KeySavingsTab
        accessToken="test-token"
        keyToken="key-abc123"
        userId="user-123"
        userRole="user"
      />
    );

    expect(screen.getByTestId("key-savings-scope-note")).toHaveTextContent("Showing your own requests");
  });

  it("does not show scope note for admin viewers", () => {
    vi.spyOn(useScopedDailyActivityRangeModule, "useScopedDailyActivityRange").mockReturnValue(mockActivity());

    render(
      <KeySavingsTab
        accessToken="test-token"
        keyToken="key-abc123"
        userId="admin-123"
        userRole="proxy_admin"
      />
    );

    expect(screen.queryByTestId("key-savings-scope-note")).not.toBeInTheDocument();
  });
});
