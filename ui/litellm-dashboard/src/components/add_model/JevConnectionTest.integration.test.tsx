import { afterEach, describe, expect, it, vi } from "vitest";
import { fireEvent, renderWithProviders, screen, waitFor } from "../../../tests/test-utils";
import AutoRouterConnectionTest from "./auto_router_connection_test";
import AutoRouterRoutingTest from "./AutoRouterRoutingTest";
import { buildAutoRouterTestTargets } from "./build_auto_router_test_targets";
import {
  buildSavedJevConnectionTestRequest,
  JEV_CONNECTION_TEST_PROMPT,
} from "./build_auto_router_routing_test_request";
import { buildComplexityRouterConfig, type BuildComplexityRouterConfigParams } from "./build_complexity_router_config";

vi.mock(
  "@/app/(dashboard)/hooks/autoRouter/useComplexityScorerDefaults",
  async () => await import("../../../tests/mocks/complexityScorerDefaults"),
);

const configParams: BuildComplexityRouterConfigParams = {
  classifierType: "jev",
  jevClassifierConfig: { model: "jev-latest", timeout_ms: 3000 },
  tiers: { SIMPLE: ["fast"], MEDIUM: ["mid"], COMPLEX: ["strong"], REASONING: ["reasoner"] },
  defaultModel: undefined,
  planModeMinTier: undefined,
  tierLabels: undefined,
  classifierLlmConfig: undefined,
  classifierContextWindowSize: undefined,
  classifierContextBudgetChars: undefined,
  classifierContextIncludeAssistantTurns: undefined,
  classifierFallback: undefined,
  classificationPrompt: undefined,
  classificationExamples: undefined,
  heuristicFirstMaxTier: undefined,
  classificationMode: undefined,
  sessionAffinity: false,
  deploymentAffinity: true,
  customTechnicalKeywords: [],
  keywordTierRules: [],
  semanticMatchingEnabled: false,
  embeddingModel: undefined,
  matchThreshold: 0.5,
  escalationKeywords: [],
  adaptive: false,
  adaptiveWeights: { quality: 0.3, cost: 0.7 },
  tierDistancePenalty: 0.5,
  adaptiveEligible: "all",
  returnRawModelName: false,
};
const config = buildComplexityRouterConfig(configParams);
const request = buildSavedJevConnectionTestRequest(JSON.stringify(config), "fast", "my-router");
const targets = buildAutoRouterTestTargets({
  tiers: Object.entries(config.tiers),
  semanticMatchingEnabled: false,
  embeddingModel: undefined,
});
const response = (cause: string) => ({
  routed_model: "fast",
  routed_model_configured: true,
  routing_decision: {
    cause,
    tier: "SIMPLE",
    classifier_model: "jev-latest",
    classifier_confidence: 0.8,
    classifier_probabilities: { SIMPLE: 0.8, REASONING: 0.2 },
    classifier_cost: 0.00001234,
  },
});

afterEach(() => vi.unstubAllGlobals());

describe("JEV network probes", () => {
  it.each(["jev_classifier", "classifier_fallback", "default_model_fallback", "keyword_match"])(
    "probes the routing endpoint independently of tier models and checks the cause %s",
    async (cause) => {
      const fetchMock = vi.fn<typeof fetch>(
        async (input) =>
          new Response(JSON.stringify(String(input).endsWith("/auto_router/test_routing") ? response(cause) : {})),
      );
      vi.stubGlobal("fetch", fetchMock);
      const onTestComplete = vi.fn();
      renderWithProviders(
        <AutoRouterConnectionTest
          accessToken="test-token"
          targets={targets}
          jevRequest={request}
          onTestComplete={onTestComplete}
        />,
      );
      await waitFor(() => expect(onTestComplete).toHaveBeenCalledOnce());
      expect(fetchMock).toHaveBeenCalledWith(
        expect.stringContaining("/auto_router/test_routing"),
        expect.objectContaining({
          method: "POST",
          body: expect.any(String),
        }),
      );
      const routingCall = fetchMock.mock.calls.find(([url]) => String(url).endsWith("/auto_router/test_routing"));
      const expectedRequest = {
        prompt: JEV_CONNECTION_TEST_PROMPT,
        complexity_router_config: config,
        default_model: "fast",
        router_name: "my-router",
      };
      expect(JSON.parse(String(routingCall?.[1]?.body))).toEqual(expectedRequest);
      expect(fetchMock).toHaveBeenCalledTimes(5);
      expect(screen.getAllByTestId("test-status-success")).toHaveLength(4);
      expect(screen.getByRole("status", { name: "JEV connection" })).toHaveTextContent(
        cause === "jev_classifier"
          ? "JEV classification succeeded"
          : `JEV was not reached successfully (routing cause: ${cause})`,
      );
    },
  );

  it("shows routing diagnostics from the real networking response", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn<typeof fetch>(async () => new Response(JSON.stringify(response("jev_classifier")))),
    );
    renderWithProviders(
      <AutoRouterRoutingTest
        accessToken="token"
        config={config}
        defaultModel="fast"
        routerName="router"
        teamId={undefined}
      />,
    );
    fireEvent.change(screen.getByTestId("auto-router-routing-test-prompt"), { target: { value: "Hello" } });
    fireEvent.click(screen.getByTestId("auto-router-routing-test-send"));
    expect(await screen.findByText("JEV classifier")).toBeInTheDocument();
    expect(screen.getByText("jev-latest")).toBeInTheDocument();
    expect(screen.getByText("80.0%")).toBeInTheDocument();
    expect(screen.getByText("SIMPLE: 80.0%")).toBeInTheDocument();
    expect(screen.getByText("REASONING: 20.0%")).toBeInTheDocument();
    expect(screen.getByText("$0.00001234")).toBeInTheDocument();
  });

  it("reports a classifier endpoint error while still checking downstream models", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn<typeof fetch>(async (input) =>
        String(input).endsWith("/auto_router/test_routing")
          ? new Response(JSON.stringify({ detail: "JEV classifier unavailable" }), { status: 503 })
          : new Response("{}"),
      ),
    );
    renderWithProviders(<AutoRouterConnectionTest accessToken="token" targets={targets} jevRequest={request} />);
    expect(await screen.findByText("JEV classifier unavailable")).toBeInTheDocument();
    expect(screen.getAllByTestId("test-status-success")).toHaveLength(4);
  });
});
