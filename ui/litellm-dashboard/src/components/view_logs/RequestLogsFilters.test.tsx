import { screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { renderWithProviders, testQueryClient } from "../../../tests/test-utils";
import { LOG_FILTER_IDS } from "./log_filter_logic";
import { RequestLogsFilters } from "./RequestLogsFilters";

vi.mock("@/app/(dashboard)/hooks/keys/useKeyAliases", () => ({
  useInfiniteKeyAliases: vi.fn(),
}));

vi.mock("@/app/(dashboard)/hooks/models/useModels", () => ({
  useInfiniteModelInfo: vi.fn(),
}));

vi.mock("../networking", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../networking")>();
  return { ...actual, allEndUsersCall: vi.fn().mockResolvedValue([]) };
});

import { useInfiniteKeyAliases } from "@/app/(dashboard)/hooks/keys/useKeyAliases";
import { useInfiniteModelInfo } from "@/app/(dashboard)/hooks/models/useModels";

const emptyInfiniteQuery = {
  data: { pages: [], pageParams: [] },
  fetchNextPage: vi.fn(),
  hasNextPage: false,
  isFetchingNextPage: false,
  isLoading: false,
};

function renderFilters(filters: Record<string, string> = {}) {
  const set = vi.fn();
  renderWithProviders(
    <RequestLogsFilters get={(id: string) => filters[id]} set={set} teams={[]} accessToken="test-token" />,
  );
  return { set };
}

describe("RequestLogsFilters", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    testQueryClient.clear();
    vi.mocked(useInfiniteKeyAliases).mockReturnValue(
      emptyInfiniteQuery as unknown as ReturnType<typeof useInfiniteKeyAliases>,
    );
    vi.mocked(useInfiniteModelInfo).mockReturnValue(
      emptyInfiniteQuery as unknown as ReturnType<typeof useInfiniteModelInfo>,
    );
  });

  it("renders every backend-supported filter field", async () => {
    renderFilters();

    for (const label of [
      "Team ID",
      "Status",
      "Key Alias",
      "End User",
      "Error Code",
      "Error Message",
      "Key Hash",
      "Session ID",
      "Model",
      "Public model / search tool",
    ]) {
      expect(await screen.findByText(label)).toBeInTheDocument();
    }
  });

  it("scopes the Key Alias lookup to the selected team", async () => {
    renderFilters({ [LOG_FILTER_IDS.TEAM_ID]: "team-42" });

    await waitFor(() => expect(useInfiniteKeyAliases).toHaveBeenCalled());
    expect(useInfiniteKeyAliases).toHaveBeenCalledWith(50, undefined, "team-42");
  });

  it("leaves the Key Alias lookup unscoped when no team is selected", async () => {
    renderFilters();

    await waitFor(() => expect(useInfiniteKeyAliases).toHaveBeenCalled());
    expect(useInfiniteKeyAliases).toHaveBeenCalledWith(50, undefined, undefined);
  });

  it("does not leak the team scope into the Model lookup", async () => {
    renderFilters({ [LOG_FILTER_IDS.TEAM_ID]: "team-42" });

    await waitFor(() => expect(useInfiniteModelInfo).toHaveBeenCalled());
    expect(useInfiniteModelInfo).toHaveBeenCalledWith(50, undefined);
  });
});
