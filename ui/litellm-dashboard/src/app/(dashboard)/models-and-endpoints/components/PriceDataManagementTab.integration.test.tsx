import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import ModelDeprecationBanner from "./ModelDeprecationBanner";
import PriceDataManagementTab from "./PriceDataManagementTab";

const TEST_TODAY = new Date(2026, 10, 4);

vi.mock("@/components/price_data_reload", () => ({
  default: ({ onReloadSuccess }: { onReloadSuccess: () => void }) => <button onClick={onReloadSuccess}>reload</button>,
}));
vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  default: () => ({ accessToken: "sk-test", userRole: "Admin" }),
}));

const renderTabWithBanner = (queryClient: QueryClient) =>
  render(
    <QueryClientProvider client={queryClient}>
      <>
        <PriceDataManagementTab />
        <ModelDeprecationBanner today={TEST_TODAY} />
      </>
    </QueryClientProvider>,
  );

describe("PriceDataManagementTab", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("refetches deprecations after reloading price data", async () => {
    const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    const deprecatedResponse = {
      checked_at: "2026-06-01T00:00:00Z",
      warn_within_days: 30,
      deprecated: [
        {
          model_name: "old-model",
          litellm_provider: null,
          deprecation_date: "2026-06-01",
          days_until_deprecation: -1,
          status: "deprecated",
          successor_model: null,
        },
      ],
      imminent: [],
      upcoming: [],
    };
    const currentResponse = {
      checked_at: "2026-06-02T00:00:00Z",
      warn_within_days: 30,
      deprecated: [],
      imminent: [
        {
          model_name: "current-model",
          litellm_provider: null,
          deprecation_date: "2026-11-30",
          days_until_deprecation: 26,
          status: "imminent",
          successor_model: null,
        },
      ],
      upcoming: [],
    };
    const fetchMock = vi
      .fn<typeof fetch>()
      .mockResolvedValueOnce(Response.json(deprecatedResponse))
      .mockResolvedValueOnce(Response.json(currentResponse));
    vi.stubGlobal("fetch", fetchMock);
    renderTabWithBanner(queryClient);

    expect(await screen.findByText("old-model was deprecated on 2026-06-01")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "reload" }));
    expect(await screen.findByText("current-model deprecates on 2026-11-30, in 26 days")).toBeInTheDocument();
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2));
    expect(
      fetchMock.mock.calls.map(
        ([input]) => new URL(input instanceof Request ? input.url : String(input), "http://localhost").pathname,
      ),
    ).toEqual(["/model/deprecations", "/model/deprecations"]);
  });
});
