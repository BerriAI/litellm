/* @vitest-environment jsdom */
import { render } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import HealthStatusPage from "./page";

vi.mock("next/navigation", () => ({
  usePathname: () => "/models-and-endpoints/health",
  useRouter: () => ({ push: vi.fn(), replace: vi.fn() }),
  useSearchParams: () => new URLSearchParams(""),
}));

const mockHealthCheckComponent = vi.fn((_props: { all_models_on_proxy?: string[] }) => null);
vi.mock("@/components/model_dashboard/HealthCheckComponent", () => ({
  default: (props: { all_models_on_proxy?: string[] }) => {
    mockHealthCheckComponent(props);
    return null;
  },
}));

vi.mock("@/app/(dashboard)/models-and-endpoints/utils/modelDataTransformer", () => ({
  transformModelData: () => ({ data: [] }),
}));

const mockUseModelsInfo = vi.fn();
vi.mock("@/app/(dashboard)/hooks/models/useModels", () => ({ useModelsInfo: () => mockUseModelsInfo() }));
vi.mock("@/app/(dashboard)/hooks/models/useModelCostMap", () => ({ useModelCostMap: () => ({ data: {} }) }));
vi.mock("@/app/(dashboard)/hooks/teams/useTeams", () => ({ useTeams: () => ({ data: [] }) }));
vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({ default: () => ({ accessToken: "123" }) }));

describe("HealthStatusPage", () => {
  beforeEach(() => {
    mockHealthCheckComponent.mockClear();
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

    render(<HealthStatusPage />);

    expect(mockHealthCheckComponent).toHaveBeenCalled();
    const props = mockHealthCheckComponent.mock.calls[0][0];
    expect(props.all_models_on_proxy).toEqual(["deployment-id-1", "deployment-id-2"]);
    expect(props.all_models_on_proxy).not.toContain("gpt-4");
  });
});
