import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { components } from "@/lib/http/schema";
import { type DailyActivityKeyPageResponse, type KeyActivityRow, type KeySpendActivityRow } from "../dailyActivityApi";
import type { ModelActivityData } from "../types";
import KeyActivityPanel from "./KeyActivityPanel";

let triggerIntersection: (() => void) | undefined;

const intersectSentinel = async () => {
  await waitFor(() => expect(triggerIntersection).toBeDefined());
  await act(async () => {
    triggerIntersection?.();
  });
};

class TestIntersectionObserver implements IntersectionObserver {
  readonly root: Element | Document | null = null;
  readonly rootMargin = "";
  readonly thresholds: readonly number[] = [];

  constructor(private readonly callback: IntersectionObserverCallback) {}

  observe(target: Element): void {
    triggerIntersection = () =>
      this.callback(
        [
          {
            boundingClientRect: target.getBoundingClientRect(),
            intersectionRect: target.getBoundingClientRect(),
            intersectionRatio: 1,
            isIntersecting: true,
            rootBounds: null,
            target,
            time: 0,
          },
        ],
        this,
      );
  }

  unobserve(): void {}

  disconnect(): void {
    triggerIntersection = undefined;
  }

  takeRecords(): IntersectionObserverEntry[] {
    return [];
  }
}

const metrics: components["schemas"]["SpendMetrics"] = {
  api_requests: 2,
  autorouter_savings_spend: 0,
  cache_creation_input_tokens: 0,
  cache_read_input_tokens: 0,
  completion_tokens: 3,
  compression_saved_tokens: 0,
  compression_savings_spend: 0,
  failed_requests: 0,
  flat_cost: 0,
  gateway_injected_caching_savings_spend: 0,
  prompt_caching_savings_spend: 0,
  prompt_tokens: 4,
  spend: 1.25,
  successful_requests: 2,
  timed_requests: 0,
  total_response_time_ms: 0,
  total_tokens: 7,
};

const pageRow = (apiKey: string): KeySpendActivityRow => ({
  api_key: apiKey,
  metrics: {
    api_requests: 2,
    cache_creation_input_tokens: 0,
    cache_read_input_tokens: 0,
    completion_tokens: 3,
    failed_requests: 0,
    prompt_tokens: 4,
    spend: 1.25,
    successful_requests: 2,
    total_tokens: 7,
  },
  metadata: { key_alias: apiKey, team_id: null },
});

const searchRow = (apiKey: string, alias: string): KeyActivityRow => ({
  api_key: apiKey,
  metrics,
  metadata: { key_alias: alias, team_id: null },
});

const pageResponse = (apiKeys: KeySpendActivityRow[], total: number, offset = 0): DailyActivityKeyPageResponse => ({
  api_keys: apiKeys,
  total_api_keys: total,
  offset,
  limit: 50,
});

const summary: ModelActivityData = {
  label: "Overall Usage",
  total_requests: 200,
  total_successful_requests: 198,
  total_failed_requests: 2,
  total_cache_read_input_tokens: 0,
  total_cache_creation_input_tokens: 0,
  total_tokens: 700,
  prompt_tokens: 400,
  completion_tokens: 300,
  total_spend: 500,
  total_response_time_ms: 0,
  total_timed_requests: 0,
  top_models: [],
  daily_data: [
    {
      date: "2026-09-27",
      metrics: {
        prompt_tokens: 4,
        completion_tokens: 3,
        total_tokens: 7,
        api_requests: 2,
        spend: 1.25,
        successful_requests: 2,
        failed_requests: 0,
        cache_read_input_tokens: 0,
        cache_creation_input_tokens: 0,
      },
    },
  ],
};

const detail = (apiKey: string): ModelActivityData => ({
  label: apiKey,
  total_requests: 2,
  total_successful_requests: 2,
  total_failed_requests: 0,
  total_cache_read_input_tokens: 0,
  total_cache_creation_input_tokens: 0,
  total_tokens: 7,
  prompt_tokens: 4,
  completion_tokens: 3,
  total_spend: 1.25,
  total_response_time_ms: 0,
  total_timed_requests: 0,
  top_models: [
    { model: "gpt-4o-mini", spend: 1.25, requests: 2, successful_requests: 2, failed_requests: 0, tokens: 7 },
  ],
  daily_data: [
    {
      date: "2026-09-27",
      metrics: {
        prompt_tokens: 4,
        completion_tokens: 3,
        total_tokens: 7,
        api_requests: 2,
        spend: 1.25,
        successful_requests: 2,
        failed_requests: 0,
        cache_read_input_tokens: 0,
        cache_creation_input_tokens: 0,
        avg_response_time_ms: null,
      },
    },
  ],
});

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
  vi.useRealTimers();
  triggerIntersection = undefined;
});

beforeEach(() => {
  vi.stubGlobal("IntersectionObserver", TestIntersectionObserver);
});

describe("KeyActivityPanel", () => {
  it("loads the first page, appends the next page, and keeps limit copy out of the UI", async () => {
    let resolveNextPage: (response: DailyActivityKeyPageResponse) => void = () => {};
    const nextPage = new Promise<DailyActivityKeyPageResponse>((resolve) => {
      resolveNextPage = resolve;
    });
    const firstPageRows = Array.from({ length: 50 }, (_, index) => pageRow(`key-${index}`));
    const fetchKeyPage = vi.fn((offset: number, _limit: number) =>
      offset === 0 ? Promise.resolve(pageResponse(firstPageRows, 52)) : nextPage,
    );

    render(
      <KeyActivityPanel
        summary={summary}
        fetchKeyPage={fetchKeyPage}
        fetchKeyDetail={vi.fn().mockResolvedValue(detail("unused"))}
        searchKeys={vi.fn().mockResolvedValue({ api_keys: [] })}
        teams={[]}
      />,
    );

    expect(await screen.findByRole("button", { name: /key-49/ })).toBeInTheDocument();
    expect(screen.getByText("52 keys")).toBeInTheDocument();
    expect(screen.getByText("$500.00")).toBeInTheDocument();
    expect(fetchKeyPage).toHaveBeenCalledWith(0, 50);
    expect(screen.queryByText(/limit|truncat|highest-spend|load top/i)).not.toBeInTheDocument();

    await intersectSentinel();
    expect(fetchKeyPage).toHaveBeenCalledWith(50, 50);
    expect(screen.getByText("Loading more keys...")).toBeInTheDocument();

    await act(async () => {
      resolveNextPage(pageResponse([pageRow("key-50"), pageRow("key-51")], 52, 50));
      await nextPage;
    });
    expect(await screen.findByRole("button", { name: /key-51/ })).toBeInTheDocument();
    expect(screen.queryByText("Loading more keys...")).not.toBeInTheDocument();
  });

  it("fetches full detail on first expansion and renders charts with daily data", async () => {
    const fetchKeyPage = vi.fn().mockResolvedValue(pageResponse([pageRow("key-chart")], 1));
    let resolveDetail: (metrics: ModelActivityData) => void = () => {};
    const detailResponse = new Promise<ModelActivityData>((resolve) => {
      resolveDetail = resolve;
    });
    const fetchKeyDetail = vi.fn().mockReturnValue(detailResponse);
    render(
      <KeyActivityPanel
        summary={summary}
        fetchKeyPage={fetchKeyPage}
        fetchKeyDetail={fetchKeyDetail}
        searchKeys={vi.fn().mockResolvedValue({ api_keys: [] })}
        teams={[]}
      />,
    );

    fireEvent.click(await screen.findByRole("button", { name: /key-chart/ }));

    expect(fetchKeyDetail).toHaveBeenCalledWith("key-chart");
    expect(await screen.findByText("Loading key details...")).toBeInTheDocument();
    await act(async () => {
      resolveDetail(detail("key-chart"));
      await detailResponse;
    });
    expect(await screen.findByText("Spend per day")).toBeInTheDocument();
    expect(screen.getByText("Requests per day")).toBeInTheDocument();
    expect(screen.queryByText("No data")).not.toBeInTheDocument();
  });

  it("shows a detail error with Retry when the detail request fails and refetches on retry", async () => {
    const fetchKeyPage = vi.fn().mockResolvedValue(pageResponse([pageRow("key-broken")], 1));
    const fetchKeyDetail = vi.fn().mockRejectedValueOnce(new Error("boom")).mockResolvedValueOnce(detail("key-broken"));
    const consoleError = vi.spyOn(console, "error").mockImplementation(() => {});
    render(
      <KeyActivityPanel
        summary={summary}
        fetchKeyPage={fetchKeyPage}
        fetchKeyDetail={fetchKeyDetail}
        searchKeys={vi.fn().mockResolvedValue({ api_keys: [] })}
        teams={[]}
      />,
    );

    fireEvent.click(await screen.findByRole("button", { name: /key-broken/ }));

    expect(await screen.findByText(/Could not load key details\./)).toBeInTheDocument();
    expect(fetchKeyDetail).toHaveBeenCalledTimes(1);
    expect(screen.queryByText("Spend per day")).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Retry" }));

    expect(fetchKeyDetail).toHaveBeenCalledTimes(2);
    expect(fetchKeyDetail).toHaveBeenLastCalledWith("key-broken");
    expect(await screen.findByText("Spend per day")).toBeInTheDocument();
    expect(screen.queryByText(/Could not load key details\./)).not.toBeInTheDocument();
    consoleError.mockRestore();
  });

  it("shows a loader instead of zero totals while the summary is in flight", async () => {
    const fetchKeyPage = vi.fn().mockResolvedValue(pageResponse([pageRow("key-local")], 1));
    const zeroSummary: ModelActivityData = { ...summary, total_spend: 0, total_requests: 0, total_tokens: 0 };
    const props = {
      fetchKeyPage,
      fetchKeyDetail: vi.fn(),
      searchKeys: vi.fn().mockResolvedValue({ api_keys: [] }),
      teams: [],
    };
    const { rerender } = render(<KeyActivityPanel summary={zeroSummary} summaryLoading {...props} />);

    expect(await screen.findByRole("button", { name: /key-local/ })).toBeInTheDocument();
    expect(screen.getByText("Loading chart data...")).toBeInTheDocument();
    expect(screen.queryByText("Overall Usage")).not.toBeInTheDocument();
    expect(screen.queryByText("$0.00")).not.toBeInTheDocument();

    rerender(<KeyActivityPanel summary={summary} summaryLoading={false} {...props} />);

    expect(screen.queryByText("Loading chart data...")).not.toBeInTheDocument();
    expect(screen.getByText("Overall Usage")).toBeInTheDocument();
    expect(screen.getAllByText("$500.00").length).toBeGreaterThan(0);
  });

  it("loads details for remote search results and merges local matches", async () => {
    vi.useFakeTimers();
    const fetchKeyPage = vi.fn().mockResolvedValue(pageResponse([pageRow("key-local-remote")], 2));
    const fetchKeyDetail = vi.fn().mockResolvedValue(detail("key-remote"));
    const searchKeys = vi.fn().mockResolvedValue({
      api_keys: [searchRow("key-remote", "remote server result")],
    });
    render(
      <KeyActivityPanel
        summary={summary}
        fetchKeyPage={fetchKeyPage}
        fetchKeyDetail={fetchKeyDetail}
        searchKeys={searchKeys}
        teams={[]}
      />,
    );

    fireEvent.change(screen.getByLabelText("Search keys"), { target: { value: "remote" } });
    await act(async () => {
      await vi.advanceTimersByTimeAsync(300);
    });
    vi.useRealTimers();
    expect(searchKeys).toHaveBeenCalledWith("remote");
    expect(screen.getByText("2 matching keys")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /key-local-remote/ })).toBeInTheDocument();
    const remoteButton = screen.getByRole("button", { name: /remote server result/ });
    fireEvent.click(remoteButton);

    expect(fetchKeyDetail).toHaveBeenCalledWith("key-remote");
    expect(await screen.findByText("Spend per day")).toBeInTheDocument();
    expect(screen.queryByText("No data")).not.toBeInTheDocument();
  });

  it("shows an error with Retry when the first page fails and recovers on retry", async () => {
    let resolveRetry: (page: DailyActivityKeyPageResponse) => void = () => undefined;
    const fetchKeyPage = vi
      .fn()
      .mockRejectedValueOnce(new Error("boom"))
      .mockImplementationOnce(
        () =>
          new Promise<DailyActivityKeyPageResponse>((resolve) => {
            resolveRetry = resolve;
          }),
      );
    render(
      <KeyActivityPanel
        summary={summary}
        fetchKeyPage={fetchKeyPage}
        fetchKeyDetail={vi.fn().mockResolvedValue(detail("unused"))}
        searchKeys={vi.fn().mockResolvedValue({ api_keys: [] })}
        teams={[]}
      />,
    );

    expect(await screen.findByText("Could not load keys for this range.")).toBeInTheDocument();
    expect(screen.queryByText("0 keys")).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Retry" }));

    expect(screen.getByText("Loading keys...")).toBeInTheDocument();
    expect(screen.queryByText("Could not load keys for this range.")).not.toBeInTheDocument();
    expect(fetchKeyPage).toHaveBeenCalledTimes(2);
    expect(fetchKeyPage).toHaveBeenNthCalledWith(2, 0, 50);

    await act(async () => {
      resolveRetry(pageResponse([pageRow("key-after-retry")], 1));
    });
    expect(await screen.findByRole("button", { name: /key-after-retry/ })).toBeInTheDocument();
    expect(screen.queryByText("Loading keys...")).not.toBeInTheDocument();
  });

  it("stops auto-retrying after a failed next page until Retry is clicked", async () => {
    const firstPageRows = Array.from({ length: 50 }, (_, index) => pageRow(`key-${index}`));
    const fetchKeyPage = vi
      .fn()
      .mockResolvedValueOnce(pageResponse(firstPageRows, 52))
      .mockRejectedValueOnce(new Error("boom"))
      .mockResolvedValue(pageResponse([pageRow("key-50"), pageRow("key-51")], 52, 50));
    render(
      <KeyActivityPanel
        summary={summary}
        fetchKeyPage={fetchKeyPage}
        fetchKeyDetail={vi.fn().mockResolvedValue(detail("unused"))}
        searchKeys={vi.fn().mockResolvedValue({ api_keys: [] })}
        teams={[]}
      />,
    );

    expect(await screen.findByRole("button", { name: /key-49/ })).toBeInTheDocument();
    await intersectSentinel();
    expect(await screen.findByText("Could not load more keys.")).toBeInTheDocument();
    expect(fetchKeyPage).toHaveBeenCalledTimes(2);

    await act(async () => {
      triggerIntersection?.();
    });
    expect(fetchKeyPage).toHaveBeenCalledTimes(2);

    fireEvent.click(screen.getByRole("button", { name: "Retry" }));
    expect(await screen.findByRole("button", { name: /key-51/ })).toBeInTheDocument();
    expect(fetchKeyPage).toHaveBeenCalledTimes(3);
    expect(fetchKeyPage).toHaveBeenNthCalledWith(3, 50, 50);
  });

  it("hides the next-page Retry while searching and refetches the failed page once the search is cleared", async () => {
    const firstPageRows = Array.from({ length: 50 }, (_, index) => pageRow(`key-${index}`));
    const fetchKeyPage = vi
      .fn()
      .mockResolvedValueOnce(pageResponse(firstPageRows, 52))
      .mockRejectedValueOnce(new Error("boom"))
      .mockResolvedValue(pageResponse([pageRow("key-50"), pageRow("key-51")], 52, 50));
    render(
      <KeyActivityPanel
        summary={summary}
        fetchKeyPage={fetchKeyPage}
        fetchKeyDetail={vi.fn().mockResolvedValue(detail("unused"))}
        searchKeys={vi.fn().mockResolvedValue({ api_keys: [] })}
        teams={[]}
      />,
    );

    expect(await screen.findByRole("button", { name: /key-49/ })).toBeInTheDocument();
    await intersectSentinel();
    expect(await screen.findByText("Could not load more keys.")).toBeInTheDocument();

    fireEvent.change(screen.getByLabelText("Search keys"), { target: { value: "key-1" } });
    expect(screen.queryByText("Could not load more keys.")).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Retry" })).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Clear key search" }));
    expect(await screen.findByText("Could not load more keys.")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Retry" }));
    expect(await screen.findByRole("button", { name: /key-51/ })).toBeInTheDocument();
    expect(fetchKeyPage).toHaveBeenCalledTimes(3);
    expect(fetchKeyPage).toHaveBeenNthCalledWith(3, 50, 50);
  });

  it("shows server search matches even when the first key page failed", async () => {
    vi.useFakeTimers();
    const fetchKeyPage = vi.fn().mockRejectedValue(new Error("boom"));
    const searchKeys = vi.fn().mockResolvedValue({
      api_keys: [searchRow("key-remote", "remote server result")],
    });
    render(
      <KeyActivityPanel
        summary={summary}
        fetchKeyPage={fetchKeyPage}
        fetchKeyDetail={vi.fn().mockResolvedValue(detail("key-remote"))}
        searchKeys={searchKeys}
        teams={[]}
      />,
    );
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });
    expect(screen.getByText("Could not load keys for this range.")).toBeInTheDocument();

    fireEvent.change(screen.getByLabelText("Search keys"), { target: { value: "remote" } });
    await act(async () => {
      await vi.advanceTimersByTimeAsync(300);
    });
    vi.useRealTimers();

    expect(searchKeys).toHaveBeenCalledWith("remote");
    expect(screen.getByRole("button", { name: /remote server result/ })).toBeInTheDocument();
    expect(screen.getByText("1 matching keys")).toBeInTheDocument();
    expect(screen.queryByText("Could not load keys for this range.")).not.toBeInTheDocument();
  });

  it("keeps the loader and the first page error visible for a short query instead of No keys match", async () => {
    let resolveFirstPage: (page: DailyActivityKeyPageResponse) => void = () => undefined;
    const fetchKeyPage = vi.fn().mockImplementationOnce(
      () =>
        new Promise<DailyActivityKeyPageResponse>((resolve) => {
          resolveFirstPage = resolve;
        }),
    );
    render(
      <KeyActivityPanel
        summary={summary}
        fetchKeyPage={fetchKeyPage}
        fetchKeyDetail={vi.fn().mockResolvedValue(detail("unused"))}
        searchKeys={vi.fn().mockResolvedValue({ api_keys: [] })}
        teams={[]}
      />,
    );

    fireEvent.change(screen.getByLabelText("Search keys"), { target: { value: "a" } });
    expect(screen.getByText("Loading keys...")).toBeInTheDocument();
    expect(screen.queryByText(/No keys match/)).not.toBeInTheDocument();
    expect(screen.queryByText("0 matching keys")).not.toBeInTheDocument();

    await act(async () => {
      resolveFirstPage(pageResponse([pageRow("key-alpha")], 1));
    });
    expect(await screen.findByRole("button", { name: /key-alpha/ })).toBeInTheDocument();
    expect(screen.getByText("1 matching keys")).toBeInTheDocument();

    cleanup();
    const failingFetch = vi.fn().mockRejectedValue(new Error("boom"));
    render(
      <KeyActivityPanel
        summary={summary}
        fetchKeyPage={failingFetch}
        fetchKeyDetail={vi.fn().mockResolvedValue(detail("unused"))}
        searchKeys={vi.fn().mockResolvedValue({ api_keys: [] })}
        teams={[]}
      />,
    );
    expect(await screen.findByText("Could not load keys for this range.")).toBeInTheDocument();
    fireEvent.change(screen.getByLabelText("Search keys"), { target: { value: "a" } });
    expect(screen.getByText("Could not load keys for this range.")).toBeInTheDocument();
    expect(screen.queryByText(/No keys match/)).not.toBeInTheDocument();
    expect(screen.queryByText("0 matching keys")).not.toBeInTheDocument();
  });

  it("does not restart an in-flight search when the parent passes a new teams array", async () => {
    vi.useFakeTimers();
    const fetchKeyPage = vi.fn().mockResolvedValue(pageResponse([pageRow("key-local")], 1));
    let resolveSearch: (response: { api_keys: KeyActivityRow[] }) => void = () => undefined;
    const searchKeys = vi.fn().mockImplementation(
      () =>
        new Promise<{ api_keys: KeyActivityRow[] }>((resolve) => {
          resolveSearch = resolve;
        }),
    );
    const props = {
      summary,
      fetchKeyPage,
      fetchKeyDetail: vi.fn().mockResolvedValue(detail("key-remote")),
      searchKeys,
    };
    const { rerender } = render(<KeyActivityPanel {...props} teams={[]} />);

    fireEvent.change(screen.getByLabelText("Search keys"), { target: { value: "remote" } });
    await act(async () => {
      await vi.advanceTimersByTimeAsync(300);
    });
    expect(searchKeys).toHaveBeenCalledTimes(1);

    rerender(<KeyActivityPanel {...props} teams={[]} />);
    await act(async () => {
      await vi.advanceTimersByTimeAsync(300);
    });
    expect(searchKeys).toHaveBeenCalledTimes(1);

    await act(async () => {
      resolveSearch({ api_keys: [searchRow("key-remote", "remote server result")] });
    });
    vi.useRealTimers();
    expect(await screen.findByRole("button", { name: /remote server result/ })).toBeInTheDocument();
    expect(screen.getByText("1 matching keys")).toBeInTheDocument();
  });

  it("keeps paging past a page of already loaded keys by server offset instead of loaded row count", async () => {
    const fetchKeyPage = vi
      .fn()
      .mockResolvedValueOnce(pageResponse([pageRow("key-1"), pageRow("key-2")], 5))
      .mockResolvedValueOnce(pageResponse([pageRow("key-1"), pageRow("key-2")], 5, 2))
      .mockResolvedValueOnce(pageResponse([pageRow("key-5")], 5, 4));
    render(
      <KeyActivityPanel
        summary={summary}
        fetchKeyPage={fetchKeyPage}
        fetchKeyDetail={vi.fn().mockResolvedValue(detail("unused"))}
        searchKeys={vi.fn().mockResolvedValue({ api_keys: [] })}
        teams={[]}
      />,
    );
    expect(await screen.findByRole("button", { name: /key-2/ })).toBeInTheDocument();

    await intersectSentinel();
    expect(fetchKeyPage).toHaveBeenNthCalledWith(2, 2, 50);
    await intersectSentinel();
    expect(fetchKeyPage).toHaveBeenNthCalledWith(3, 4, 50);
    expect(await screen.findByRole("button", { name: /key-5/ })).toBeInTheDocument();
    expect(fetchKeyPage).toHaveBeenCalledTimes(3);
    expect(screen.getAllByRole("button", { name: /key-/ })).toHaveLength(3);
  });

  it("stops requesting more keys when a later page is empty even though the total says more exist", async () => {
    const fetchKeyPage = vi
      .fn()
      .mockResolvedValueOnce(pageResponse([pageRow("key-1"), pageRow("key-2")], 3))
      .mockResolvedValue(pageResponse([], 3, 2));
    render(
      <KeyActivityPanel
        summary={summary}
        fetchKeyPage={fetchKeyPage}
        fetchKeyDetail={vi.fn().mockResolvedValue(detail("unused"))}
        searchKeys={vi.fn().mockResolvedValue({ api_keys: [] })}
        teams={[]}
      />,
    );
    expect(await screen.findByRole("button", { name: /key-2/ })).toBeInTheDocument();

    await intersectSentinel();
    await act(async () => {
      triggerIntersection?.();
    });
    expect(fetchKeyPage).toHaveBeenCalledTimes(2);
    expect(fetchKeyPage).toHaveBeenLastCalledWith(2, 50);
    expect(triggerIntersection).toBeUndefined();
    expect(screen.getByText("3 keys")).toBeInTheDocument();
  });

  it("shows a search error with Retry search instead of No keys match when the search fails", async () => {
    vi.useFakeTimers();
    const fetchKeyPage = vi.fn().mockResolvedValue(pageResponse([pageRow("key-local")], 1));
    const searchKeys = vi
      .fn()
      .mockRejectedValueOnce(new Error("boom"))
      .mockResolvedValue({ api_keys: [searchRow("key-remote", "remote server result")] });
    render(
      <KeyActivityPanel
        summary={summary}
        fetchKeyPage={fetchKeyPage}
        fetchKeyDetail={vi.fn().mockResolvedValue(detail("key-remote"))}
        searchKeys={searchKeys}
        teams={[]}
      />,
    );
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });

    fireEvent.change(screen.getByLabelText("Search keys"), { target: { value: "remote" } });
    await act(async () => {
      await vi.advanceTimersByTimeAsync(300);
    });
    expect(searchKeys).toHaveBeenCalledTimes(1);
    expect(screen.getByText(/Could not search keys\./)).toBeInTheDocument();
    expect(screen.queryByText(/No keys match/)).not.toBeInTheDocument();
    expect(screen.queryByText("0 matching keys")).not.toBeInTheDocument();
    expect(screen.getByText("Search failed")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Retry search" }));
    expect(screen.getByText("Searching...")).toBeInTheDocument();
    await act(async () => {
      await vi.advanceTimersByTimeAsync(300);
    });
    vi.useRealTimers();

    expect(searchKeys).toHaveBeenCalledTimes(2);
    expect(screen.getByRole("button", { name: /remote server result/ })).toBeInTheDocument();
    expect(screen.queryByText(/Could not search keys\./)).not.toBeInTheDocument();
  });

  it("discards stale pages and clears loaded keys when the scope changes", async () => {
    let resolveOldPage: (response: DailyActivityKeyPageResponse) => void = () => {};
    const oldPage = new Promise<DailyActivityKeyPageResponse>((resolve) => {
      resolveOldPage = resolve;
    });
    const firstScopeFetch = vi.fn().mockReturnValue(oldPage);
    const secondScopeFetch = vi.fn().mockResolvedValue(pageResponse([pageRow("new-scope-key")], 1));
    const props = {
      summary,
      fetchKeyDetail: vi.fn().mockResolvedValue(detail("new-scope-key")),
      searchKeys: vi.fn().mockResolvedValue({ api_keys: [] }),
      teams: [],
    };
    const { rerender } = render(<KeyActivityPanel {...props} fetchKeyPage={firstScopeFetch} />);
    rerender(<KeyActivityPanel {...props} fetchKeyPage={secondScopeFetch} />);

    expect(await screen.findByRole("button", { name: /new-scope-key/ })).toBeInTheDocument();
    await act(async () => {
      resolveOldPage(pageResponse([pageRow("old-scope-key")], 1));
      await oldPage;
    });
    expect(screen.queryByRole("button", { name: /old-scope-key/ })).not.toBeInTheDocument();
    expect(secondScopeFetch).toHaveBeenCalledWith(0, 50);
  });

  it("renders key search before the summary metrics", async () => {
    render(
      <KeyActivityPanel
        summary={summary}
        fetchKeyPage={vi.fn().mockResolvedValue(pageResponse([], 0))}
        fetchKeyDetail={vi.fn()}
        searchKeys={vi.fn().mockResolvedValue({ api_keys: [] })}
        teams={[]}
      />,
    );

    const searchInput = await screen.findByRole("textbox", { name: "Search keys" });
    const summaryHeading = await screen.findByRole("heading", { name: "Overall Usage" });

    expect(searchInput.compareDocumentPosition(summaryHeading) & Node.DOCUMENT_POSITION_FOLLOWING).toBe(
      Node.DOCUMENT_POSITION_FOLLOWING,
    );
  });
});
