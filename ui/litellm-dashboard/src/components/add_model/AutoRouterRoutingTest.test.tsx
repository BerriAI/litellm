import { fireEvent, renderWithProviders, screen, waitFor } from "../../../tests/test-utils";
import userEvent from "@testing-library/user-event";
import { vi } from "vitest";
import AutoRouterRoutingTest from "./AutoRouterRoutingTest";
import { testAutoRouterRouting } from "../networking";
import { ComplexityRouterConfigPayload } from "./build_complexity_router_config";
vi.mock(
  "@/app/(dashboard)/hooks/autoRouter/useComplexityScorerDefaults",
  async () => await import("../../../tests/mocks/complexityScorerDefaults"),
);

vi.mock("../networking", () => ({
  testAutoRouterRouting: vi.fn(),
}));

const CONFIG = {
  tiers: { SIMPLE: ["cheap"], MEDIUM: ["mid"], COMPLEX: ["strong"], REASONING: ["o3"] },
  classifier_type: "heuristic",
} as unknown as ComplexityRouterConfigPayload;

const Harness = () => (
  <AutoRouterRoutingTest
    accessToken="token"
    config={CONFIG}
    defaultModel="mid"
    routerName="my-router"
    teamId={undefined}
  />
);

const expectedRequest = {
  prompt: "think step by step",
  complexity_router_config: CONFIG,
  default_model: "mid",
  router_name: "my-router",
};

const successResponse = {
  status: "success" as const,
  result: {
    routed_model: "o3",
    routed_model_configured: true,
    routing_decision: {
      router_model_name: "my-router",
      router_type: "complexity",
      routed_model: "o3",
      cause: "heuristic_scorer",
      tier: "REASONING",
      score: 0.91,
    },
  },
};

describe("AutoRouterRoutingTest", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("cannot send an empty prompt", () => {
    renderWithProviders(<Harness />);

    expect(screen.getByTestId("auto-router-routing-test-send")).toBeDisabled();
  });

  it("routes the typed prompt through the config being edited and shows where it landed", async () => {
    const user = userEvent.setup();
    vi.mocked(testAutoRouterRouting).mockResolvedValue(successResponse);
    renderWithProviders(<Harness />);

    fireEvent.change(screen.getByTestId("auto-router-routing-test-prompt"), {
      target: { value: "think step by step" },
    });
    await user.click(screen.getByTestId("auto-router-routing-test-send"));

    expect(testAutoRouterRouting).toHaveBeenCalledWith("token", expectedRequest);
    expect(await screen.findByTestId("auto-router-routing-test-routed-model")).toHaveTextContent("o3");
    expect(screen.getByText("REASONING")).toBeInTheDocument();
    expect(screen.queryByTestId("auto-router-routing-test-unconfigured")).not.toBeInTheDocument();
  });

  it("warns when the routed model is not a model group on this proxy", async () => {
    const user = userEvent.setup();
    vi.mocked(testAutoRouterRouting).mockResolvedValue({
      ...successResponse,
      result: { ...successResponse.result, routed_model_configured: false },
    });
    renderWithProviders(<Harness />);

    fireEvent.change(screen.getByTestId("auto-router-routing-test-prompt"), { target: { value: "hello" } });
    await user.click(screen.getByTestId("auto-router-routing-test-send"));

    expect(await screen.findByTestId("auto-router-routing-test-unconfigured")).toBeInTheDocument();
  });

  it("shows why a prompt could not be routed", async () => {
    const user = userEvent.setup();
    vi.mocked(testAutoRouterRouting).mockResolvedValue({ status: "error", error: "no tier has a model" });
    renderWithProviders(<Harness />);

    fireEvent.change(screen.getByTestId("auto-router-routing-test-prompt"), { target: { value: "hello" } });
    await user.click(screen.getByTestId("auto-router-routing-test-send"));

    expect(await screen.findByText("no tier has a model")).toBeInTheDocument();
    await waitFor(() => expect(screen.queryByTestId("auto-router-routing-test-result")).not.toBeInTheDocument());
  });
});
