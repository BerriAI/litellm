/* @vitest-environment jsdom */
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { modelCostMapKeys } from "../../hooks/models/useModelCostMap";
import PriceDataManagementTab from "./PriceDataManagementTab";

vi.mock("@/components/price_data_reload", () => ({
  default: ({ onReloadSuccess }: { onReloadSuccess: () => void }) => <button onClick={onReloadSuccess}>reload</button>,
}));
vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({ default: () => ({ accessToken: "sk-test" }) }));

const renderTab = (queryClient: QueryClient) =>
  render(
    <QueryClientProvider client={queryClient}>
      <PriceDataManagementTab />
    </QueryClientProvider>,
  );

describe("PriceDataManagementTab", () => {
  it("renders its content standalone, without a tab-panel ancestor", () => {
    renderTab(new QueryClient());
    expect(screen.getByText("Price Data Management")).toBeInTheDocument();
  });

  it("invalidates both the live and the catalog-only cost map after a reload", () => {
    const queryClient = new QueryClient();
    const liveKey = modelCostMapKeys.list({});
    const catalogKey = modelCostMapKeys.list({ filters: { catalog_only: "true" } });
    queryClient.setQueryData(liveKey, {});
    queryClient.setQueryData(catalogKey, {});
    renderTab(queryClient);

    fireEvent.click(screen.getByRole("button", { name: "reload" }));

    expect(queryClient.getQueryState(liveKey)?.isInvalidated).toBe(true);
    expect(queryClient.getQueryState(catalogKey)?.isInvalidated).toBe(true);
  });
});
