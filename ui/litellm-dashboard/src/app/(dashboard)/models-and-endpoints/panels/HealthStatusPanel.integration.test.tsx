import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { NuqsTestingAdapter, type OnUrlUpdateFunction } from "nuqs/adapters/testing";
import type { ReactNode } from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { act, render, screen, waitFor } from "@/../tests/test-utils";

import HealthStatusPanel from "./HealthStatusPanel";

const { modelInfoCall, latestHealthChecksCall } = vi.hoisted(() => ({
  modelInfoCall: vi.fn(),
  latestHealthChecksCall: vi.fn(),
}));

vi.mock("@/components/networking", () => ({
  modelInfoCall,
  latestHealthChecksCall,
  individualModelHealthCheckCall: vi.fn(),
  modelHubCall: vi.fn(),
  modelAvailableCall: vi.fn(),
}));

vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  default: () => ({ accessToken: "sk-test", userId: "u-admin", userRole: "Admin" }),
}));
vi.mock("@/app/(dashboard)/hooks/teams/useTeams", () => ({ useTeams: () => ({ data: [] }) }));
vi.mock("@/app/(dashboard)/hooks/models/useModelCostMap", () => ({ useModelCostMap: () => ({ data: {} }) }));

const PAGE_ARG = 3;
const TEN_MODELS_FIRST_PAGE = { data: [], total_count: 10, current_page: 1, total_pages: 1, size: 50 };

const renderPanel = (searchParams: string, onUrlUpdate: OnUrlUpdateFunction) => {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(<HealthStatusPanel />, {
    wrapper: ({ children }: { children: ReactNode }) => (
      <NuqsTestingAdapter
        searchParams={searchParams}
        onUrlUpdate={onUrlUpdate}
        hasMemory
        resetUrlUpdateQueueOnMount={false}
      >
        <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
      </NuqsTestingAdapter>
    ),
  });
  return queryClient;
};

const flushUrlUpdates = () => act(() => new Promise((resolve) => setTimeout(resolve, 50)));

describe("HealthStatusPanel health_page", () => {
  beforeEach(() => {
    modelInfoCall.mockReset();
    latestHealthChecksCall.mockReset();
    latestHealthChecksCall.mockResolvedValue({ latest_health_checks: {} });
  });

  it("snaps a page past the end back to the first page once the models load", async () => {
    modelInfoCall.mockResolvedValue(TEN_MODELS_FIRST_PAGE);
    const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();

    renderPanel("?health_page=3", onUrlUpdate);

    await waitFor(() => expect(onUrlUpdate).toHaveBeenCalled());
    expect(modelInfoCall.mock.calls[0][PAGE_ARG]).toBe(3);
    expect(onUrlUpdate.mock.calls.at(-1)?.[0].searchParams.has("health_page")).toBe(false);
  });

  it("keeps the page in the URL when fetching it fails", async () => {
    modelInfoCall.mockRejectedValue(new Error("proxy unavailable"));
    const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();

    const queryClient = renderPanel("?health_page=3", onUrlUpdate);

    await waitFor(() => expect(queryClient.getQueryCache().getAll()[0]?.state.status).toBe("error"));
    await flushUrlUpdates();

    expect(modelInfoCall.mock.calls[0][PAGE_ARG]).toBe(3);
    expect(onUrlUpdate).not.toHaveBeenCalled();
    expect(screen.queryByText("No models found")).not.toBeInTheDocument();
  });
});
