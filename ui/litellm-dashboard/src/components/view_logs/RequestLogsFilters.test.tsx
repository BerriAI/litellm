import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
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

vi.mock("@/app/(dashboard)/hooks/spendLogs/useSpendLogEndUsers", () => ({
  useInfiniteSpendLogEndUsers: vi.fn(),
}));

import { useInfiniteSpendLogEndUsers } from "@/app/(dashboard)/hooks/spendLogs/useSpendLogEndUsers";
import { useInfiniteKeyAliases } from "@/app/(dashboard)/hooks/keys/useKeyAliases";
import { useInfiniteModelInfo } from "@/app/(dashboard)/hooks/models/useModels";

const emptyInfiniteQuery = {
  data: { pages: [], pageParams: [] },
  fetchNextPage: vi.fn(),
  hasNextPage: false,
  isFetchingNextPage: false,
  isLoading: false,
};

const LOGS_WINDOW = { start_date: "2026-07-23 00:00:00", end_date: "2026-07-24 00:00:00" };

function renderFilters(filters: Record<string, string> = {}) {
  const set = vi.fn();
  renderWithProviders(
    <RequestLogsFilters get={(id: string) => filters[id]} set={set} teams={[]} logsWindow={LOGS_WINDOW} />,
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
    vi.mocked(useInfiniteSpendLogEndUsers).mockReturnValue(
      emptyInfiniteQuery as unknown as ReturnType<typeof useInfiniteSpendLogEndUsers>,
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

  it("asks the server for a bounded page of end users scoped to the visible time window", async () => {
    renderFilters();

    await waitFor(() => expect(useInfiniteSpendLogEndUsers).toHaveBeenCalled());
    expect(useInfiniteSpendLogEndUsers).toHaveBeenCalledWith(LOGS_WINDOW, 50, undefined);
  });

  it("pushes the End User query to the server rather than filtering a preloaded list", async () => {
    const user = userEvent.setup();
    renderFilters();

    const input = await screen.findByPlaceholderText("Search an end user");
    await user.click(input);
    await user.type(input, "acme");

    await waitFor(() => expect(useInfiniteSpendLogEndUsers).toHaveBeenCalledWith(LOGS_WINDOW, 50, "acme"));
  });

  it("renders only the end users the current page returned", async () => {
    vi.mocked(useInfiniteSpendLogEndUsers).mockReturnValue({
      ...emptyInfiniteQuery,
      data: {
        pages: [
          {
            data: ["cust-a", "cust-b"],
            meta: { page: 1, page_size: 50, has_more: true },
            links: { self: "", next: "?page=2" },
          },
        ],
        pageParams: [1],
      },
    } as unknown as ReturnType<typeof useInfiniteSpendLogEndUsers>);
    const user = userEvent.setup();
    renderFilters();

    await user.click(await screen.findByPlaceholderText("Search an end user"));

    expect(await screen.findByText("cust-a")).toBeInTheDocument();
    expect(screen.getByText("cust-b")).toBeInTheDocument();
  });

  it("loads the next page when the End User list is scrolled near the end", async () => {
    const fetchNextPage = vi.fn();
    vi.mocked(useInfiniteSpendLogEndUsers).mockReturnValue({
      ...emptyInfiniteQuery,
      fetchNextPage,
      hasNextPage: true,
      data: {
        pages: [
          { data: ["cust-a"], meta: { page: 1, page_size: 50, has_more: true }, links: { self: "", next: "?page=2" } },
        ],
        pageParams: [1],
      },
    } as unknown as ReturnType<typeof useInfiniteSpendLogEndUsers>);
    const user = userEvent.setup();
    renderFilters();

    await user.click(await screen.findByPlaceholderText("Search an end user"));
    const list = await screen.findByTestId("paginated-search-select-list");
    Object.defineProperty(list, "scrollTop", { value: 90, configurable: true });
    Object.defineProperty(list, "clientHeight", { value: 10, configurable: true });
    Object.defineProperty(list, "scrollHeight", { value: 100, configurable: true });
    list.dispatchEvent(new Event("scroll", { bubbles: true }));

    await waitFor(() => expect(fetchNextPage).toHaveBeenCalled());
  });

  it("scopes the End User lookup to the window the logs table is showing", async () => {
    const otherWindow = { start_date: "2026-01-01 00:00:00", end_date: "2026-01-02 00:00:00" };
    renderWithProviders(<RequestLogsFilters get={() => undefined} set={vi.fn()} teams={[]} logsWindow={otherWindow} />);

    await waitFor(() => expect(useInfiniteSpendLogEndUsers).toHaveBeenCalledWith(otherWindow, 50, undefined));
  });
});
