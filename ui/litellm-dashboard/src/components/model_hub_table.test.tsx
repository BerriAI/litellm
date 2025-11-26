import * as networking from "@/components/networking";
import { render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import ModelHubTable from "./model_hub_table";

vi.mock("@/components/networking", () => ({
  getUiConfig: vi.fn(),
  modelHubPublicModelsCall: vi.fn(),
  modelHubCall: vi.fn(),
  getConfigFieldSetting: vi.fn(),
  getProxyBaseUrl: vi.fn(() => "http://localhost:4000"),
  getAgentsList: vi.fn(),
  fetchMCPServers: vi.fn(),
}));

vi.mock("next/navigation", () => ({
  useRouter: () => ({
    replace: vi.fn(),
  }),
}));

vi.mock("./public_model_hub", () => ({
  default: () => <div>Public Model Hub</div>,
}));

describe("ModelHubTable", () => {
  afterEach(() => {
    vi.clearAllMocks();
  });

  it("should render", async () => {
    vi.mocked(networking.modelHubCall).mockResolvedValue({
      data: [],
    });
    vi.mocked(networking.getConfigFieldSetting).mockResolvedValue({
      field_value: false,
    });
    vi.mocked(networking.getAgentsList).mockResolvedValue({
      agents: [],
    });
    vi.mocked(networking.fetchMCPServers).mockResolvedValue([]);

    render(<ModelHubTable accessToken="test-token" publicPage={false} premiumUser={false} userRole={null} />);

    await waitFor(() => {
      expect(screen.getByText("AI Hub")).toBeInTheDocument();
    });
  });

  it("should call getUiConfig before modelHubPublicModelsCall when publicPage is true", async () => {
    const getUiConfigMock = vi.mocked(networking.getUiConfig);
    const modelHubPublicModelsCallMock = vi.mocked(networking.modelHubPublicModelsCall);

    getUiConfigMock.mockResolvedValue({ server_root_path: "/", proxy_base_url: "http://localhost:4000" });
    modelHubPublicModelsCallMock.mockResolvedValue([]);

    render(<ModelHubTable accessToken={null} publicPage={true} premiumUser={false} userRole={null} />);

    await waitFor(() => {
      expect(getUiConfigMock).toHaveBeenCalled();
      expect(modelHubPublicModelsCallMock).toHaveBeenCalled();
    });

    const getUiConfigCallOrder = getUiConfigMock.mock.invocationCallOrder[0];
    const modelHubPublicModelsCallOrder = modelHubPublicModelsCallMock.mock.invocationCallOrder[0];

    expect(getUiConfigCallOrder).toBeLessThan(modelHubPublicModelsCallOrder);
  });
});
