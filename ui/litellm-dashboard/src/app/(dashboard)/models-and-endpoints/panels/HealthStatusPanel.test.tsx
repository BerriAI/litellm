/* @vitest-environment jsdom */
import type { OnChangeFn, PaginationState } from "@tanstack/react-table";
import { act, render, waitFor } from "@testing-library/react";
import { type OnUrlUpdateFunction, withNuqsTestingAdapter } from "nuqs/adapters/testing";
import { beforeEach, describe, expect, it, vi } from "vitest";
import HealthStatusPanel from "./HealthStatusPanel";

interface CapturedHealthCheckProps {
  all_models_on_proxy?: string[];
  pagination?: PaginationState;
  onPaginationChange?: OnChangeFn<PaginationState>;
}

const mockHealthCheckComponent = vi.fn((_props: CapturedHealthCheckProps) => null);
vi.mock("@/components/model_dashboard/HealthCheckComponent", () => ({
  default: (props: CapturedHealthCheckProps) => {
    mockHealthCheckComponent(props);
    return null;
  },
}));

const lastHealthCheckProps = (): CapturedHealthCheckProps => {
  const lastCall = mockHealthCheckComponent.mock.calls.at(-1);
  if (!lastCall) throw new Error("HealthCheckComponent was not rendered");
  return lastCall[0];
};

vi.mock("@/app/(dashboard)/models-and-endpoints/utils/modelDataTransformer", () => ({
  transformModelData: () => ({ data: [] }),
}));

const mockUseModelsInfo = vi.fn();
vi.mock("@/app/(dashboard)/hooks/models/useModels", () => ({
  useModelsInfo: (...args: unknown[]) => mockUseModelsInfo(...args),
}));
vi.mock("@/app/(dashboard)/hooks/models/useModelCostMap", () => ({ useModelCostMap: () => ({ data: {} }) }));
vi.mock("@/app/(dashboard)/hooks/teams/useTeams", () => ({ useTeams: () => ({ data: [] }) }));
vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({ default: () => ({ accessToken: "123" }) }));

describe("HealthStatusPanel", () => {
  beforeEach(() => {
    mockHealthCheckComponent.mockClear();
    mockUseModelsInfo.mockReset();
    mockUseModelsInfo.mockReturnValue({ data: undefined, isLoading: true });
  });

  it("fetches and shows the page and page size named in the URL", () => {
    render(<HealthStatusPanel />, {
      wrapper: withNuqsTestingAdapter({ searchParams: "?health_page=3&health_page_size=25" }),
    });

    expect(mockUseModelsInfo).toHaveBeenLastCalledWith(3, 25);
    expect(lastHealthCheckProps().pagination).toEqual({ pageIndex: 2, pageSize: 25 });
  });

  it("writes table page changes to the URL and refetches that page", async () => {
    const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
    render(<HealthStatusPanel />, { wrapper: withNuqsTestingAdapter({ onUrlUpdate, hasMemory: true }) });
    expect(mockUseModelsInfo).toHaveBeenLastCalledWith(1, 50);

    act(() => lastHealthCheckProps().onPaginationChange?.({ pageIndex: 1, pageSize: 50 }));

    await waitFor(() => expect(onUrlUpdate).toHaveBeenCalled());
    const params = onUrlUpdate.mock.calls.at(-1)?.[0].searchParams;
    expect(params?.get("health_page")).toBe("2");
    expect(params?.get("health_page_size")).toBeNull();
    await waitFor(() => expect(mockUseModelsInfo).toHaveBeenLastCalledWith(2, 50));
  });

  it("passes deployment ids (not model names) to HealthCheckComponent as all_models_on_proxy", () => {
    mockUseModelsInfo.mockReturnValue({
      data: {
        data: [
          { model_name: "gpt-4", model_info: { id: "deployment-id-1" } },
          { model_name: "gpt-4", model_info: { id: "deployment-id-2" } },
        ],
        total_count: 2,
      },
      isLoading: false,
    });

    render(<HealthStatusPanel />, { wrapper: withNuqsTestingAdapter() });

    expect(mockHealthCheckComponent).toHaveBeenCalled();
    const props = mockHealthCheckComponent.mock.calls[0][0];
    expect(props.all_models_on_proxy).toEqual(["deployment-id-1", "deployment-id-2"]);
    expect(props.all_models_on_proxy).not.toContain("gpt-4");
  });
});
