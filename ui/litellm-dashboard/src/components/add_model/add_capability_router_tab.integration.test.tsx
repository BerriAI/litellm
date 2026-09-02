import userEvent from "@testing-library/user-event";
import { fireEvent, renderWithProviders, screen, testQueryClient, waitFor } from "../../../tests/test-utils";
import { beforeEach, describe, expect, it, vi } from "vitest";

import AddAutoRouterTab from "./add_auto_router_tab";
import { handleAddAutoRouterSubmit } from "./handle_add_auto_router_submit";

vi.mock(
  "@/app/(dashboard)/hooks/autoRouter/useComplexityScorerDefaults",
  async () => await import("../../../tests/mocks/complexityScorerDefaults"),
);

const { mockFetchAvailableModels, mockFetchAllModelDeployments, validateAutoRouterConfig } = vi.hoisted(() => ({
  mockFetchAvailableModels: vi.fn(),
  mockFetchAllModelDeployments: vi.fn(),
  validateAutoRouterConfig: vi.fn().mockResolvedValue({ valid: true }),
}));

vi.mock("../networking", () => ({
  modelAvailableCall: vi.fn().mockResolvedValue({ data: [] }),
  validateAutoRouterConfig,
}));
vi.mock("@/components/llm_calls/fetch_models", () => ({ fetchAvailableModels: mockFetchAvailableModels }));
vi.mock("@/app/(dashboard)/hooks/models/useModels", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/app/(dashboard)/hooks/models/useModels")>();
  return { ...actual, fetchAllModelDeployments: mockFetchAllModelDeployments };
});
vi.mock("./handle_add_auto_router_submit", () => ({ handleAddAutoRouterSubmit: vi.fn() }));
vi.mock("./CapabilityRouterConfig", () => {
  const filledConfig = {
    candidates: [
      { model: "small", description: "Bounded extraction" },
      { model: "frontier", description: "Ambiguous multi-step work" },
    ],
    classifier: { model: "classifier", timeout_ms: 3000, max_output_tokens: 1024 },
    probability_threshold: 0.75,
    fallback_model: "frontier",
    estimated_output_tokens: 1000,
    cache_ttl_seconds: 3600,
  };
  return {
    default: ({ onChange }: { onChange: (value: unknown) => void }) => (
      <button type="button" onClick={() => onChange(filledConfig)}>
        Fill capability config
      </button>
    ),
  };
});

describe("AddAutoRouterTab capability flow", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    testQueryClient.clear();
    mockFetchAvailableModels.mockResolvedValue(
      ["small", "frontier", "classifier"].map((model_group) => ({ model_group, mode: "chat" })),
    );
    mockFetchAllModelDeployments.mockResolvedValue([]);
  });

  it("validates and submits the capability config shown in the form", async () => {
    const user = userEvent.setup();
    renderWithProviders(<AddAutoRouterTab handleOk={vi.fn()} accessToken="token" userRole="Admin" />);

    fireEvent.click(screen.getByTestId("router-type-selector"));
    await user.click(screen.getByRole("option", { name: "Capability" }));
    expect(screen.queryByTestId("template-selector")).not.toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Fill capability config" }));
    await user.type(screen.getByPlaceholderText(/smart_router/i), "cost-router");
    await user.click(screen.getByRole("button", { name: "Add Auto Router" }));

    await waitFor(() => expect(handleAddAutoRouterSubmit).toHaveBeenCalledOnce());
    expect(validateAutoRouterConfig).toHaveBeenCalledWith(
      "token",
      expect.objectContaining({ probability_threshold: 0.75, cache_ttl_seconds: 3600 }),
      undefined,
      "capability",
    );
    expect(vi.mocked(handleAddAutoRouterSubmit).mock.calls[0]?.[0]).toMatchObject({
      auto_router_name: "cost-router",
      model_type: "capability_router",
      capability_router_config: {
        fallback_model: "frontier",
        probability_threshold: 0.75,
        cache_ttl_seconds: 3600,
      },
    });
  });
});
