import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { uiHref } from "@/utils/uiHref";
import { useBuilderInsights } from "./useBuilderInsights";

function BuilderInsightsStatus() {
  const { data, isError } = useBuilderInsights();
  if (data) return <div>Loaded</div>;
  if (isError) return <div>Failed</div>;
  return <div>Loading</div>;
}

describe("useBuilderInsights", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("loads the sample from the configured dashboard UI path", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: vi.fn().mockResolvedValue({ builders: [] }),
    });
    vi.stubGlobal("fetch", fetchMock);
    const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });

    render(
      <QueryClientProvider client={queryClient}>
        <BuilderInsightsStatus />
      </QueryClientProvider>,
    );

    expect(await screen.findByText("Loaded")).toBeInTheDocument();
    expect(fetchMock).toHaveBeenCalledWith(uiHref("builder-insights-sample.json"));
    queryClient.clear();
  });

  it("reports an unavailable sample when the asset request fails", async () => {
    const fetchMock = vi.fn().mockResolvedValue({ ok: false });
    vi.stubGlobal("fetch", fetchMock);
    const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });

    render(
      <QueryClientProvider client={queryClient}>
        <BuilderInsightsStatus />
      </QueryClientProvider>,
    );

    expect(await screen.findByText("Failed")).toBeInTheDocument();
    queryClient.clear();
  });
});
