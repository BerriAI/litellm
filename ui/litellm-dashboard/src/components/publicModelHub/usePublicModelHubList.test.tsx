import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, renderHook, waitFor } from "@testing-library/react";
import { NuqsTestingAdapter, type OnUrlUpdateFunction, type UrlUpdateEvent } from "nuqs/adapters/testing";
import React, { type PropsWithChildren } from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { PUBLIC_MODEL_HUB_PATH, usePublicModelHubList } from "./usePublicModelHubList";

const { apiGetMock } = vi.hoisted(() => ({ apiGetMock: vi.fn() }));

vi.mock("@/components/networking", () => ({
  apiClient: { get: apiGetMock },
}));

type QueryRecord = Record<string, string | number>;

const queries = (): QueryRecord[] =>
  apiGetMock.mock.calls
    .filter((call) => call[0] === PUBLIC_MODEL_HUB_PATH)
    .map((call) => (call[1] as { query: QueryRecord }).query);

const renderList = (searchParams: string, onUrlUpdate?: OnUrlUpdateFunction) => {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 } } });
  const wrapper = ({ children }: PropsWithChildren) => (
    <NuqsTestingAdapter
      searchParams={searchParams}
      onUrlUpdate={onUrlUpdate}
      hasMemory
      resetUrlUpdateQueueOnMount={false}
    >
      <QueryClientProvider client={client}>{children}</QueryClientProvider>
    </NuqsTestingAdapter>
  );
  return renderHook(() => usePublicModelHubList(true), { wrapper });
};

const lastUrl = (onUrlUpdate: ReturnType<typeof vi.fn<OnUrlUpdateFunction>>): URLSearchParams => {
  const events = onUrlUpdate.mock.calls.map(([event]: [UrlUpdateEvent]) => event);
  return events[events.length - 1].searchParams;
};

describe("usePublicModelHubList", () => {
  beforeEach(() => {
    apiGetMock.mockReset();
    apiGetMock.mockResolvedValue({
      data: [],
      meta: { total_count: 500, page: 1, page_size: 50, total_pages: 10 },
    });
  });

  it("opens a shared link on the same page, sort, search and filters", async () => {
    const { result } = renderList(
      "?page=4&page_size=25&sort_by=rpm&sort_order=desc&q=gpt&filter_providers=openai,azure&filter_mode=chat&filter_features=vision",
    );

    const expectedQuery: QueryRecord = {
      page: 4,
      page_size: 25,
      sort: "-rpm",
      q: "gpt",
      "filter[providers][in]": "openai,azure",
      "filter[mode][in]": "chat",
      "filter[features][in]": "vision",
    };
    await waitFor(() => expect(queries()).toHaveLength(1));
    expect(queries()[0]).toEqual(expectedQuery);
    expect(result.current.providerValues).toEqual(["openai", "azure"]);
    expect(result.current.modeValues).toEqual(["chat"]);
    expect(result.current.featureValues).toEqual(["vision"]);
    expect(result.current.searchValue).toBe("gpt");
    expect(result.current.hasActiveQuery).toBe(true);
  });

  it("sorts by model name when the link names a field the endpoint cannot sort on", async () => {
    renderList("?sort_by=health_status&sort_order=desc");

    await waitFor(() => expect(queries()).toHaveLength(1));
    expect(queries()[0].sort).toBe("-model_group");
  });

  it("writes each multi-select comma separated and returns to the first page", async () => {
    const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
    const { result } = renderList("?page=3", onUrlUpdate);
    await waitFor(() => expect(queries()).toHaveLength(1));

    act(() => result.current.onProvidersChange(["openai", "vertex_ai"]));
    await waitFor(() => expect(lastUrl(onUrlUpdate).get("filter_providers")).toBe("openai,vertex_ai"));
    expect(lastUrl(onUrlUpdate).has("page")).toBe(false);

    act(() => result.current.onModesChange(["embedding"]));
    await waitFor(() => expect(lastUrl(onUrlUpdate).get("filter_mode")).toBe("embedding"));
    expect(lastUrl(onUrlUpdate).get("filter_providers")).toBe("openai,vertex_ai");

    act(() => result.current.onFeaturesChange(["vision", "function_calling"]));
    await waitFor(() => expect(lastUrl(onUrlUpdate).get("filter_features")).toBe("vision,function_calling"));

    await waitFor(() => expect(result.current.providerValues).toEqual(["openai", "vertex_ai"]));
    expect(queries()[queries().length - 1]["filter[features][in]"]).toBe("vision,function_calling");
  });

  it("removes a filter from the URL once its last value is cleared", async () => {
    const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
    const { result } = renderList("?filter_providers=openai&filter_mode=chat", onUrlUpdate);
    await waitFor(() => expect(queries()).toHaveLength(1));

    act(() => result.current.onProvidersChange([]));
    await waitFor(() => expect(onUrlUpdate).toHaveBeenCalled());
    expect(lastUrl(onUrlUpdate).has("filter_providers")).toBe(false);
    expect(lastUrl(onUrlUpdate).get("filter_mode")).toBe("chat");
    await waitFor(() => expect(result.current.providerValues).toEqual([]));
  });
});
